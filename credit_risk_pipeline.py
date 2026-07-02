"""
Credit Risk Extraction Pipeline
================================
Automated NLP system for quantitative credit research at Joywin International.
Scans financial news and extracts events material to a company's S&P credit rating.

Architecture
------------
Phase 1 -- Data Ingestion & Dynamic Routing
    Fetches company news via Finnhub API. Maps Yahoo Finance industry to the
    correct S&P sector using a curated override dictionary and word-overlap scoring.

Phase 2 -- Cross-Encoder Relevance Filter (judge path)
    Scores every article headline+summary against the sector's S&P criteria using
    a cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2). Unlike cosine
    similarity, the cross-encoder reads both texts together and scores how well the
    headline relates to the specific criterion -- much better signal-to-noise.
    Only the top CROSSENCODER_TOP_N articles are carried forward; everything else
    is discarded before any scraping happens.

Phase 3 -- Selective Full-Text Scraping
    Downloads complete article text only for the articles that passed Phase 2.
    Uses requests + newspaper3k with a hard 8-second timeout (bypasses newspaper's
    own HTTP client which has no timeout and hangs on paywalls/slow CDNs).

Phase 4 -- Local-Claude Judge ("The Analyst")
    The real relevance decision. Each filtered article is judged by the local Claude
    Code CLI (claude -p, your subscription -- no API key): is this a material credit
    event, good or bad for the bond, which S&P factor, and why. Claude receives the
    full article text and the best-matching S&P criterion as context. Returns a clean
    event_summary, direction, confidence, and rationale per article.

Phase 4b -- FinBERT Tone (secondary signal only)
    Runs FinBERT on the judge's event_summary to add a tone label (positive/negative/
    neutral). Shown next to the judge's credit direction -- flags divergent cases where
    upbeat-sounding news is actually bad for credit, or vice versa.
    NOTE: Claude's credit direction (Phase 4) is the authoritative signal. FinBERT is a
    lexical sentiment model with no understanding of credit context; it cannot distinguish
    debt issuance (bad for bondholders, sounds routine) from genuine recovery. Use
    tone_alignment = divergent as a prompt to re-read the article, not as a correction.

Phase 5 -- Deduplication & Ranking
    Removes near-duplicate signals from multiple outlets covering the same event.
    Ranks by judge confidence and exports a structured JSON report.

A/B switch: set USE_LLM_JUDGE = False to revert to the legacy cosine-only path
(spaCy entity filter + bi-encoder paragraph scoring) for direct comparison.

Usage
-----
    1. Set TICKER, COMPANY_NAME, and the date window in the CONFIGURATION block.
    2. Set your Finnhub API key as an environment variable:
           set FINNHUB_API_KEY=your_key_here        (Windows CMD)
           $env:FINNHUB_API_KEY="your_key_here"     (PowerShell)
    3. Run:  python credit_risk_pipeline.py

Dependencies
------------
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm   # only needed for legacy path
    # The judge (Phase 4) calls your local Claude Code CLI -- install it once via
    #   irm https://claude.ai/install.ps1 | iex   (then run `claude` to log in)
"""

import os
import re
import sys
import subprocess
import requests
import yfinance as yf
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import datetime
import torch


# ==========================================
# CONFIGURATION  -- edit these before running
# ==========================================

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY_HERE")
TICKER          = "BA"
COMPANY_NAME    = "Boeing"

START_DATE      = "2026-05-01"
END_DATE        = "2026-06-30"

# --- Cross-encoder relevance filter (judge path, Phase 2) ---
CROSSENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSSENCODER_TOP_N  = 25    # articles forwarded to Claude after CE filter
MAX_ARTICLE_CHARS   = 4000  # character limit for article text sent to the judge

# --- Legacy cosine gate (used ONLY when USE_LLM_JUDGE = False) ---
SIMILARITY_THRESHOLD    = 0.42
MIN_CONFIDENCE_GAP      = 0.04
MIN_PARAGRAPH_WORDS     = 20

# --- Shared pipeline knobs ---
MAX_SIGNALS_PER_ARTICLE = 2     # cap per article (legacy path)
DEDUP_THRESHOLD         = 0.88  # cosine score above which two signals are near-duplicates
TOP_N_OUTPUT            = 25    # maximum signals in the final report
SCRAPE_WORKERS          = 15    # parallel threads for HTTP scraping
MAX_ARTICLES            = 300   # cap on Finnhub articles fetched; None = no cap

# --- LLM judge (Phase 4; uses your local Claude subscription) ---
USE_LLM_JUDGE        = True
JUDGE_MODEL          = "haiku"
JUDGE_WORKERS        = 4
JUDGE_TIMEOUT        = 120
MIN_JUDGE_CONFIDENCE = 0.6

# --- Credit score bands (5-tier; calibrate after more backtests) ---
# Each tier maps to a likely S&P action type based on observed backtest scores.
# STLA downgrade = -0.743, BA outlook-stable = +0.304 — use as anchors.
SCORE_BANDS = [
    (-1.00, -0.50, "DOWNGRADE WATCH",  "Likely downgrade or outlook negative"),
    (-0.50, -0.20, "CAUTION",          "Deteriorating — monitor closely"),
    (-0.20, +0.20, "STABLE",           "No material credit action expected"),
    (+0.20, +0.50, "IMPROVING",        "Possible outlook positive or stable revision"),
    (+0.50, +1.00, "UPGRADE WATCH",    "Likely upgrade or outlook positive"),
]

# --- FinBERT tone (Phase 4b; secondary signal — Claude's direction takes precedence) ---
# FinBERT is a lexical sentiment model; it has no credit context. Trust Claude's judgment.
USE_FINBERT   = True
FINBERT_MODEL = "ProsusAI/finbert"

# --- Claude CLI location ---
def _default_claude_bin():
    local = Path(os.path.expanduser("~")) / ".local" / "bin" / "claude.exe"
    return str(local) if local.exists() else "claude"

CLAUDE_BIN = _default_claude_bin()


# ==========================================
# SECTOR OVERRIDES  (Yahoo Finance -> S&P)
# ==========================================

SECTOR_OVERRIDES = {
    "Advertising Agencies":                "Media And Entertainment",
    "Aerospace & Defense":                 "Aerospace And Defense",
    "Agricultural Inputs":                 "Agribusiness, Commodity Foods, And Agricultural Cooperatives",
    "Airlines":                            "Transportation Cyclical",
    "Airports & Air Services":             "Transportation Infrastructure",
    "Apparel Manufacturing":               "Consumer Staples And Branded Nondurables",
    "Apparel Retail":                      "Retail And Restaurants",
    "Asset Management":                    "Asset Managers",
    "Auto & Truck Dealerships":            "Retail And Restaurants",
    "Auto Manufacturers":                  "Auto And Commercial Vehicle Manufacturing",
    "Auto Parts":                          "Auto Suppliers",
    "Banks - Diversified":                 "Financial Services Finance Companies",
    "Banks - Regional":                    "Financial Services Finance Companies",
    "Beverages - Brewers":                 "Consumer Staples And Branded Nondurables",
    "Beverages - Non-Alcoholic":           "Consumer Staples And Branded Nondurables",
    "Beverages - Wineries & Distilleries": "Consumer Staples And Branded Nondurables",
    "Biotechnology":                       "Pharmaceuticals",
    "Broadcasting":                        "Media And Entertainment",
    "Building Products & Equipment":       "Building Materials",
    "Capital Markets":                     "Financial Market Infrastructure",
    "Chemicals":                           "Commodity Chemicals",
    "Communication Equipment":             "Technology Hardware And Semiconductors",
    "Computer Hardware":                   "Technology Hardware And Semiconductors",
    "Confectioners":                       "Consumer Staples And Branded Nondurables",
    "Conglomerates":                       "Business And Consumer Services",
    "Consumer Electronics":                "Technology Hardware And Semiconductors",
    "Copper":                              "Mining",
    "Credit Services":                     "Financial Services Finance Companies",
    "Department Stores":                   "Retail And Restaurants",
    "Diagnostics & Research":              "Health Care Equipment",
    "Discount Stores":                     "Retail And Restaurants",
    "Drug Manufacturers - General":        "Pharmaceuticals",
    "Drug Manufacturers - Specialty & Generic": "Pharmaceuticals",
    "Education & Training Services":       "Business And Consumer Services",
    "Electronic Components":               "Technology Hardware And Semiconductors",
    "Electronic Gaming & Multimedia":      "Media And Entertainment",
    "Engineering & Construction":          "Engineering And Construction",
    "Entertainment":                       "Media And Entertainment",
    "Farm & Heavy Construction Machinery": "Capital Goods",
    "Farm Products":                       "Agribusiness, Commodity Foods, And Agricultural Cooperatives",
    "Financial Data & Stock Exchanges":    "Financial Market Infrastructure",
    "Food Distribution":                   "Agribusiness, Commodity Foods, And Agricultural Cooperatives",
    "Footwear & Accessories":              "Consumer Staples And Branded Nondurables",
    "Furnishings, Fixtures & Appliances":  "Consumer Durables",
    "Gold":                                "Mining",
    "Grocery Stores":                      "Retail And Restaurants",
    "Health Information Services":         "Health Care Services",
    "Healthcare Plans":                    "Health Care Services",
    "Home Improvement Retail":             "Retail And Restaurants",
    "Household & Personal Products":       "Consumer Staples And Branded Nondurables",
    "Information Technology Services":     "Technology Software And Services",
    "Insurance - Diversified":             "Financial Services Finance Companies",
    "Insurance - Life":                    "Financial Services Finance Companies",
    "Insurance - Property & Casualty":     "Financial Services Finance Companies",
    "Insurance Brokers":                   "Financial Services Finance Companies",
    "Integrated Freight & Logistics":      "Railroad, Package Express, And Logistics",
    "Internet Content & Information":      "Media And Entertainment",
    "Internet Retail":                     "Retail And Restaurants",
    "Leisure":                             "Leisure And Sports",
    "Lodging":                             "Leisure And Sports",
    "Lumber & Wood Production":            "Forest And Paper Products",
    "Luxury Goods":                        "Consumer Staples And Branded Nondurables",
    "Marine Shipping":                     "Transportation Cyclical",
    "Medical Care Facilities":             "Health Care Services",
    "Medical Devices":                     "Health Care Equipment",
    "Medical Instruments & Supplies":      "Health Care Equipment",
    "Mortgage Finance":                    "Financial Services Finance Companies",
    "Oil & Gas E&P":                       "Oil And Gas Exploration And Production",
    "Oil & Gas Equipment & Services":      "Oilfield Services And Equipment",
    "Oil & Gas Integrated":                "Oil And Gas Exploration And Production",
    "Oil & Gas Midstream":                 "Midstream Energy",
    "Oil & Gas Refining & Marketing":      "Refining And Marketing",
    "Other Precious Metals & Mining":      "Mining",
    "Packaging & Containers":              "Containers And Packaging",
    "Packaged Foods":                      "Consumer Staples And Branded Nondurables",
    "Paper & Paper Products":              "Forest And Paper Products",
    "Pollution & Treatment Controls":      "Environmental Services",
    "Publishing":                          "Media And Entertainment",
    "Railroads":                           "Railroad, Package Express, And Logistics",
    "Real Estate - Development":           "Homebuilders And Real Estate Developers",
    "Real Estate Services":                "Financial Services Finance Companies",
    "REIT - Diversified":                  "Homebuilders And Real Estate Developers",
    "REIT - Healthcare Facilities":        "Homebuilders And Real Estate Developers",
    "REIT - Hotel & Motel":                "Homebuilders And Real Estate Developers",
    "REIT - Industrial":                   "Homebuilders And Real Estate Developers",
    "REIT - Office":                       "Homebuilders And Real Estate Developers",
    "REIT - Residential":                  "Homebuilders And Real Estate Developers",
    "REIT - Retail":                       "Homebuilders And Real Estate Developers",
    "REIT - Specialty":                    "Homebuilders And Real Estate Developers",
    "Rental & Leasing Services":           "Business And Consumer Services",
    "Resorts & Casinos":                   "Leisure And Sports",
    "Restaurants":                         "Retail And Restaurants",
    "Security & Protection Services":      "Business And Consumer Services",
    "Semiconductor Equipment & Materials": "Technology Hardware And Semiconductors",
    "Semiconductors":                      "Technology Hardware And Semiconductors",
    "Software - Application":              "Technology Software And Services",
    "Software - Infrastructure":           "Technology Software And Services",
    "Solar":                               "Unregulated Power And Gas",
    "Specialty Business Services":         "Business And Consumer Services",
    "Specialty Chemicals":                 "Specialty Chemicals",
    "Specialty Industrial Machinery":      "Capital Goods",
    "Specialty Retail":                    "Retail And Restaurants",
    "Staffing & Employment Services":      "Business And Consumer Services",
    "Steel":                               "Metals Production And Processing",
    "Telecom Services":                    "Telecommunications",
    "Tobacco":                             "Consumer Staples And Branded Nondurables",
    "Travel Services":                     "Leisure And Sports",
    "Trucking":                            "Transportation Cyclical",
    "Uranium":                             "Mining",
    "Utilities - Diversified":             "Regulated Utilities",
    "Utilities - Independent Power Producers": "Unregulated Power And Gas",
    "Utilities - Regulated Electric":      "Regulated Utilities",
    "Utilities - Regulated Gas":           "Regulated Utilities",
    "Utilities - Regulated Water":         "Regulated Utilities",
    "Utilities - Renewable":               "Unregulated Power And Gas",
    "Waste Management":                    "Environmental Services",
}


# ==========================================
# PHASE 1 HELPER: SECTOR ROUTING
# ==========================================

def match_sp_sector(industry, sp_criteria_master):
    mapped      = SECTOR_OVERRIDES.get(industry, industry)
    best_score  = 0
    best_sector = sp_criteria_master["sectors"][0]

    for sector in sp_criteria_master["sectors"]:
        yf_words = set(mapped.replace("&", "").replace(",", "").split())
        sp_words = set(sector["sector_name"].replace("And", "").replace(",", "").split())
        score    = len(yf_words & sp_words)
        if score > best_score:
            best_score  = score
            best_sector = sector

    return best_sector


# ==========================================
# PHASE 2 HELPER: CROSS-ENCODER RELEVANCE FILTER
# ==========================================

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(CROSSENCODER_MODEL)
    return _CROSS_ENCODER


def filter_with_crossencoder(news_data, criteria_texts, criteria_labels):
    """
    Scores each article's headline + summary against every S&P criterion for the
    sector using a cross-encoder. Returns the top CROSSENCODER_TOP_N articles by
    max criterion score, each annotated with matched_criterion and risk_category.

    Cross-encoders read both texts jointly (unlike cosine which compares embeddings
    in isolation), giving much better signal on whether the headline actually relates
    to the specific S&P criterion.
    """
    ce = _get_cross_encoder()
    n_criteria = len(criteria_texts)

    # Build all (headline+summary, criterion) pairs in one flat list for batch scoring.
    all_pairs = []
    for article in news_data:
        headline = article.get("headline", "")
        summary  = article.get("summary", "")[:300]
        text     = f"{headline} {summary}".strip()
        for crit in criteria_texts:
            all_pairs.append([text, crit])

    scores = ce.predict(all_pairs)  # raw logits; higher = more relevant

    results = []
    for i, article in enumerate(news_data):
        article_scores = scores[i * n_criteria : (i + 1) * n_criteria]
        best_idx       = int(article_scores.argmax())
        results.append({
            **article,
            "_ce_score":         float(article_scores[best_idx]),
            "matched_criterion": criteria_texts[best_idx],
            "risk_category":     criteria_labels[best_idx],
        })

    results.sort(key=lambda x: x["_ce_score"], reverse=True)
    return results[:CROSSENCODER_TOP_N]


# ==========================================
# PHASE 3 HELPERS: FULL-TEXT SCRAPING
# ==========================================

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_full_text(url, fallback_summary):
    """
    Downloads raw HTML via requests (8-second hard timeout) then passes it to
    newspaper3k for local parsing. Returns the Finnhub summary fallback if scraping
    fails or yields fewer than 100 words.
    """
    try:
        response = requests.get(url, timeout=8, headers=_SCRAPE_HEADERS)
        response.raise_for_status()
        article = Article(url)
        article.set_html(response.text)
        article.parse()
        if len(article.text.split()) > 100:
            return article.text
    except Exception:
        pass
    return fallback_summary


def extract_paragraphs(text):
    """Splits text into substantive paragraphs (legacy path only)."""
    return [
        p.strip()
        for p in text.split("\n")
        if len(p.strip().split()) >= MIN_PARAGRAPH_WORDS
    ]


# ==========================================
# PHASE 3 LEGACY HELPER: ENTITY FILTERING
# ==========================================

def is_primary_subject(paragraph, company_name, doc):
    """
    Returns True if the company is a primary actor in this paragraph (legacy path).
    Accepts if: >=2 mentions, OR first ORG entity, OR grammatical subject/object role.
    """
    company_lower = company_name.lower()
    if company_lower not in paragraph.lower():
        return False
    if paragraph.lower().count(company_lower) >= 2:
        return True
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    if orgs and company_lower in orgs[0].text.lower():
        return True
    primary_deps = {"nsubj", "nsubjpass", "dobj", "attr"}
    for ent in doc.ents:
        if ent.label_ == "ORG" and company_lower in ent.text.lower():
            if ent.root.dep_ in primary_deps:
                return True
    return False


# ==========================================
# PHASE 4 HELPERS: LOCAL-CLAUDE JUDGE
# ==========================================

_JUDGE_INSTRUCTIONS = (
    "You are a senior S&P credit analyst. Decide whether this NEWS ARTICLE describes a "
    "MATERIAL credit-rating event for the TARGET COMPANY, judged against the S&P sector "
    "criterion provided. Be a strict skeptic: if the company is only mentioned in passing, "
    "if it is generic market/stock commentary, or if you are unsure, set material=false.\n"
    "Judge credit DIRECTION from the bondholder's view, NOT general tone -- e.g. new debt "
    "issuance or a debt-funded acquisition is usually 'negative' for credit even if upbeat.\n"
    "Base 'sp_factor' on the provided criterion. If it fits neither business nor financial "
    "risk, return risk_category='Neither' and material=false.\n"
    "'event_summary' must be ONE clear, self-contained sentence stating what actually "
    "happened (the event itself), readable without the original article.\n"
    "Respond with ONLY a JSON object, no preamble, no markdown fences:\n"
    '{"material": true/false, "risk_category": "Business Risk|Financial Risk|Neither", '
    '"sp_factor": "<short factor label>", "direction": "positive|negative|neutral", '
    '"confidence": 0.0-1.0, "event_summary": "<one sentence: what happened>", '
    '"rationale": "<one sentence: why it is credit-material>"}'
)


def build_judge_prompt(company, sector_name, matched_criterion, headline, article_text):
    return (
        f"{_JUDGE_INSTRUCTIONS}\n\n"
        f"TARGET COMPANY: {company}\n"
        f"S&P SECTOR: {sector_name}\n"
        f"MATCHED S&P CRITERION:\n{matched_criterion}\n\n"
        f"NEWS HEADLINE: {headline}\n"
        f"FULL ARTICLE:\n{article_text[:MAX_ARTICLE_CHARS]}"
    )


def _parse_judge_json(raw):
    outer  = json.loads(raw)
    result = outer.get("result", raw) if isinstance(outer, dict) else raw
    if isinstance(result, dict):
        return result
    text  = str(result).strip()
    fence = re.search(r"\{.*\}", text, re.DOTALL)
    if not fence:
        return None
    return json.loads(fence.group(0))


def judge_article(company, sector_name, matched_criterion, headline, article_text):
    """
    Run one local-Claude judgment on the full article text. Returns the verdict dict,
    or None on failure (skipped rather than crashing the run). One retry.
    """
    prompt = build_judge_prompt(company, sector_name, matched_criterion, headline, article_text)
    cmd    = [CLAUDE_BIN, "-p", prompt, "--model", JUDGE_MODEL,
              "--output-format", "json", "--allowedTools", ""]

    for attempt in (1, 2):
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                stdin=subprocess.DEVNULL, timeout=JUDGE_TIMEOUT,
            )
            if res.returncode != 0:
                raise RuntimeError((res.stderr or res.stdout or "non-zero exit").strip()[:200])
            verdict = _parse_judge_json(res.stdout.strip())
            if verdict is None:
                raise ValueError("no JSON object in judge output")
            return verdict
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"    [judge skipped] {str(exc).splitlines()[0][:120]}")
                return None
    return None


def run_judge_articles(articles, company, sector_name):
    """
    Judge all filtered articles in parallel. Each article dict must have full_text,
    matched_criterion, headline keys. Returns the subset the judge marks material
    and confident, with verdict fields merged in.
    """
    def work(article):
        verdict = judge_article(
            company, sector_name,
            article["matched_criterion"],
            article.get("headline", ""),
            article.get("full_text", ""),
        )
        if not verdict or not verdict.get("material"):
            return None
        try:
            confidence = float(verdict.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < MIN_JUDGE_CONFIDENCE:
            return None
        out = dict(article)
        out.update({
            "material":      True,
            "risk_category": verdict.get("risk_category", "Neither"),
            "sp_factor":     verdict.get("sp_factor", ""),
            "direction":     verdict.get("direction", "neutral"),
            "confidence":    round(confidence, 3),
            "event_summary": verdict.get("event_summary", ""),
            "rationale":     verdict.get("rationale", ""),
        })
        return out

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        verdicts = list(pool.map(work, articles))
    return [v for v in verdicts if v is not None]


# ==========================================
# PHASE 4b HELPER: FINBERT TONE
# ==========================================

_FINBERT_PIPE = None


def _get_finbert():
    global _FINBERT_PIPE
    if _FINBERT_PIPE is None:
        from transformers import pipeline
        _FINBERT_PIPE = pipeline("sentiment-analysis", model=FINBERT_MODEL,
                                 truncation=True, max_length=512)
    return _FINBERT_PIPE


def add_finbert_tone(signals):
    """
    Adds FinBERT tone label + score to each signal using the event_summary.
    Flags tone_alignment = 'divergent' when FinBERT tone disagrees with the judge's
    credit direction (e.g. upbeat tone on a debt issuance that is bad for bondholders).

    Claude's direction is authoritative. FinBERT cannot reason about credit context —
    divergent is a flag to re-read, not a reason to override the judge's verdict.
    """
    if not signals:
        return signals
    try:
        pipe = _get_finbert()
    except Exception as exc:  # noqa: BLE001
        print(f"    [FinBERT unavailable, skipping tone] {str(exc).splitlines()[0][:120]}")
        return signals

    # Use event_summary (judge-written, clean) for tone scoring.
    texts   = [s.get("event_summary", s.get("extracted_chunk", ""))[:512] for s in signals]
    results = pipe(texts)
    for sig, r in zip(signals, results):
        tone = r["label"].lower()
        sig["finbert_tone"]  = tone
        sig["finbert_score"] = round(float(r["score"]), 3)
        direction = sig.get("direction", "neutral")
        if tone in ("positive", "negative") and direction in ("positive", "negative"):
            sig["tone_alignment"] = "aligned" if tone == direction else "divergent"
        else:
            sig["tone_alignment"] = "mixed"
    return signals


# ==========================================
# PHASE 4 LEGACY HELPER: PER-ARTICLE CAP
# ==========================================

def cap_per_article(signals, max_per_article):
    """Keep at most max_per_article signals per URL (legacy path; judge path is 1 by design)."""
    ranked = sorted(signals, key=lambda s: s.get("confidence", s.get("similarity_score", 0)),
                    reverse=True)
    counts, kept = {}, []
    for s in ranked:
        url = s.get("url", "")
        if counts.get(url, 0) >= max_per_article:
            continue
        counts[url] = counts.get(url, 0) + 1
        kept.append(s)
    return kept


# ==========================================
# PHASE 5 HELPERS: DEDUPLICATION & RANKING
# ==========================================

def deduplicate(signals, encoder):
    """
    Removes near-duplicate signals (multiple outlets covering the same event).
    In judge mode, compares event_summary strings; in legacy mode, uses extracted_chunk.
    """
    if len(signals) < 2:
        return signals

    texts = [s.get("event_summary") or s.get("extracted_chunk", "") for s in signals]
    vecs  = encoder.encode(texts, convert_to_tensor=True)
    keep  = [True] * len(signals)

    score_key = "confidence" if "confidence" in signals[0] else "similarity_score"
    for i in range(len(signals)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(signals)):
            if not keep[j]:
                continue
            if util.cos_sim(vecs[i], vecs[j]).item() >= DEDUP_THRESHOLD:
                if signals[i].get(score_key, 0) >= signals[j].get(score_key, 0):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [s for i, s in enumerate(signals) if keep[i]]


# ==========================================
# CONSOLE REPORT
# ==========================================

def print_report(signals, ticker, company_name, sector_name):
    print("\n" + "=" * 65)
    print("  CREDIT RISK SIGNAL REPORT")
    print(f"  {company_name} ({ticker})  |  S&P Sector: {sector_name}")
    print("=" * 65)

    for category in ["Business Risk", "Financial Risk"]:
        cat_signals = [s for s in signals if s.get("risk_category") == category]
        if not cat_signals:
            continue
        print(f"\n{'-' * 65}")
        print(f"  {category.upper()}  ({len(cat_signals)} signals)")
        print(f"{'-' * 65}")
        for i, sig in enumerate(cat_signals, 1):
            if "direction" in sig:  # judge mode
                arrow = {"positive": "[+] POSITIVE", "negative": "[-] NEGATIVE"}.get(
                    sig["direction"], "[=] NEUTRAL")
                tone  = sig.get("finbert_tone", "")
                align = sig.get("tone_alignment", "")
                tone_str = (
                    f"  |  FinBERT: {tone}"
                    + (f" (DIVERGENT)" if align == "divergent" else "")
                ) if tone else ""
                print(f"\n  [{i}]  {sig['date']}  |  Credit: {arrow}"
                      f"  |  Confidence: {sig['confidence']}{tone_str}")
                print(f"  Event     : {sig.get('event_summary', '')[:300]}")
                print(f"  S&P factor: {sig.get('sp_factor', '')[:120]}")
                print(f"  Why       : {sig.get('rationale', '')[:300]}")
                print(f"  Headline  : {sig.get('headline', '')[:120]}")
            else:                   # legacy cosine mode
                print(f"\n  [{i}]  {sig['date']}  |  Score: {sig['similarity_score']}"
                      f"  |  Gap: {sig['confidence_gap']}")
                print(f"  Headline  : {sig['headline'][:90]}")
                print(f"  Criterion : {sig['matched_criterion'][:110]}...")
                print(f"  Extract   : {sig['extracted_chunk'][:220]}...")
            if sig.get("url"):
                print(f"  Source    : {sig['url'][:90]}")
    print()


# ==========================================
# CREDIT SCORING
# ==========================================

def compute_credit_score(signals, end_date_str):
    """
    Compute a normalized credit score in [-1.0, +1.0] from judged signals.

    Negative = credit deteriorating  (watch for downgrade)
    Positive = credit improving      (watch for upgrade)

    Each signal is weighted by:
      confidence  x  risk_weight  x  recency_weight  x  direction_sign

    risk_weight  : Financial Risk = 1.5, Business Risk = 1.0
                   Financial deterioration is a stronger predictor than operational stress.
    recency_weight: 0-30 days before end = 1.0, 31-60 days = 0.8, 61-90 days = 0.6
                   Events closer to the evaluation date matter more.
    direction_sign: negative = -1, positive = +1, neutral = 0

    Final score is normalized by the theoretical maximum (all signals pointing same way)
    so results are always in [-1, +1] regardless of signal count.
    """
    direction_map = {"negative": -1, "positive": +1, "neutral": 0}
    risk_weights  = {"Financial Risk": 1.5, "Business Risk": 1.0}

    try:
        end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        end_dt = datetime.datetime.now()

    raw_total = 0.0;  max_total = 0.0
    br_raw    = 0.0;  br_max    = 0.0
    fr_raw    = 0.0;  fr_max    = 0.0
    neg_count = 0;    pos_count = 0;    neu_count = 0

    for sig in signals:
        d_sign = direction_map.get(sig.get("direction", "neutral"), 0)
        conf   = float(sig.get("confidence", sig.get("similarity_score", 0)) or 0)
        cat    = sig.get("risk_category", "Business Risk")
        r_wt   = risk_weights.get(cat, 1.0)

        try:
            days_back = (end_dt - datetime.datetime.strptime(sig["date"], "%Y-%m-%d")).days
        except (KeyError, ValueError):
            days_back = 45
        rec_wt = 1.0 if days_back <= 30 else (0.8 if days_back <= 60 else 0.6)

        w = conf * r_wt * rec_wt
        raw_total += d_sign * w
        max_total += w

        if cat == "Financial Risk":
            fr_raw += d_sign * w;  fr_max += w
        else:
            br_raw += d_sign * w;  br_max += w

        if d_sign < 0:   neg_count += 1
        elif d_sign > 0: pos_count += 1
        else:            neu_count += 1

    score    = round(raw_total / max_total, 3) if max_total > 0 else 0.0
    br_score = round(br_raw / br_max, 3)       if br_max    > 0 else None
    fr_score = round(fr_raw / fr_max, 3)       if fr_max    > 0 else None

    verdict = SCORE_BANDS[-1][2]
    description = SCORE_BANDS[-1][3]
    for lo, hi, label, desc in SCORE_BANDS:
        if lo <= score <= hi:
            verdict = label
            description = desc
            break

    total_signals = neg_count + pos_count + neu_count
    if total_signals >= 8:
        conviction = "HIGH"
    elif total_signals >= 4:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    return {
        "score":                score,
        "verdict":              verdict,
        "description":          description,
        "conviction":           conviction,
        "business_risk_score":  br_score,
        "financial_risk_score": fr_score,
        "negative_signals":     neg_count,
        "positive_signals":     pos_count,
        "neutral_signals":      neu_count,
    }


def print_score_summary(result, ticker, company_name):
    """Print a visual credit score summary to the console."""
    score   = result["score"]
    verdict = result["verdict"]

    # ASCII position bar: maps [-1, +1] to a 42-char bar
    bar_len = 42
    bar_pos = max(0, min(bar_len - 1, int((score + 1) / 2 * bar_len)))
    bar     = "-" * bar_pos + "O" + "-" * (bar_len - 1 - bar_pos)

    print("\n" + "=" * 65)
    print("  CREDIT SCORE SUMMARY")
    print(f"  {company_name} ({ticker})")
    print("=" * 65)
    print(f"\n  Overall score  : {score:+.3f}   [{verdict}]")
    print(f"  Interpretation : {result.get('description', '')}")
    print(f"  [-1.0 NEGATIVE |{bar}| POSITIVE +1.0]")

    br = result["business_risk_score"]
    fr = result["financial_risk_score"]
    print(f"\n  Business Risk  : {f'{br:+.3f}' if br is not None else 'N/A'}")
    print(f"  Financial Risk : {f'{fr:+.3f}' if fr is not None else 'N/A'}")
    conviction = result.get("conviction", "N/A")
    conv_note  = {"HIGH": "8+ signals", "MEDIUM": "4-7 signals", "LOW": "1-3 signals"}.get(conviction, "")
    print(f"\n  Signals        : {result['negative_signals']} negative  |  "
          f"{result['positive_signals']} positive  |  "
          f"{result['neutral_signals']} neutral")
    print(f"  Conviction     : {conviction}  ({conv_note})")
    print(f"\n  Score bands :")
    for lo, hi, label, desc in SCORE_BANDS:
        marker = " <--" if verdict == label else ""
        print(f"    [{lo:+.2f} to {hi:+.2f}]  {label}{marker}")
    print("=" * 65 + "\n")


# ==========================================
# MAIN PIPELINE
# ==========================================

def _ask(prompt, default):
    """Print a prompt with a default value and return the user's input (or default)."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def run_pipeline():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    global TICKER, COMPANY_NAME, START_DATE, END_DATE, JUDGE_MODEL

    print("=" * 65)
    print("  CREDIT RISK EXTRACTION PIPELINE  -  Joywin International")
    print("=" * 65)
    print("\n  Press Enter to keep the default shown in [brackets].\n")

    TICKER       = _ask("Ticker symbol",  TICKER).upper()
    COMPANY_NAME = _ask("Company name",   COMPANY_NAME)
    START_DATE   = _ask("Start date (YYYY-MM-DD)", START_DATE)
    END_DATE     = _ask("End date   (YYYY-MM-DD)", END_DATE)
    JUDGE_MODEL  = _ask("Claude model (haiku / sonnet / opus)", JUDGE_MODEL)

    print()
    print(f"  Target : {COMPANY_NAME} ({TICKER})")
    print(f"  Period : {START_DATE}  ->  {END_DATE}")
    print(f"  Model  : {JUDGE_MODEL}\n")

    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        print("ERROR: Set the FINNHUB_API_KEY environment variable before running.")
        return

    print("Loading NLP models...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    with open("sector_risk_kw_new.json", "r", encoding="utf-8") as f:
        sp_criteria_master = json.load(f)

    # ── Phase 1: Routing ──────────────────────────────────────────
    print("\n[Phase 1] Data ingestion & sector routing...")

    yf_ticker = yf.Ticker(TICKER)
    industry  = yf_ticker.info.get("industry", "Unknown")
    print(f"  Yahoo Finance industry : {industry}")

    target_sector = match_sp_sector(industry, sp_criteria_master)
    print(f"  Matched S&P sector     : {target_sector['sector_name']}")

    criteria_labels, criteria_texts = [], []
    for kw in target_sector.get("business_risk_keywords", []):
        criteria_labels.append("Business Risk")
        criteria_texts.append(kw)
    for kw in target_sector.get("financial_risk_keywords", []):
        criteria_labels.append("Financial Risk")
        criteria_texts.append(kw)

    finnhub_url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={TICKER}&from={START_DATE}&to={END_DATE}&token={FINNHUB_API_KEY}"
    )
    news_data = requests.get(finnhub_url).json()
    print(f"  Finnhub articles fetched : {len(news_data)}")

    if MAX_ARTICLES and len(news_data) > MAX_ARTICLES:
        news_data = sorted(news_data, key=lambda a: a.get("datetime", 0),
                           reverse=True)[:MAX_ARTICLES]
        print(f"  Capped to most-recent    : {len(news_data)}")

    if USE_LLM_JUDGE:
        # ── Phase 2: Cross-encoder relevance filter ───────────────
        print(f"\n[Phase 2] Cross-encoder relevance filter ({len(news_data)} articles -> top {CROSSENCODER_TOP_N})...")
        print("  Loading cross-encoder model...")
        filtered = filter_with_crossencoder(news_data, criteria_texts, criteria_labels)
        print(f"  Kept : {len(filtered)} articles (highest criterion relevance)")

        # ── Phase 3: Scrape only the filtered articles ────────────
        print(f"\n[Phase 3] Scraping {len(filtered)} selected articles ({SCRAPE_WORKERS} threads)...")

        def fetch(article):
            url     = article.get("url", "")
            summary = article.get("summary", "")
            text    = scrape_full_text(url, summary) if url else summary
            scraped = text != summary
            article = dict(article)
            article["full_text"] = text
            article["date"] = datetime.datetime.fromtimestamp(
                article.get("datetime", 0)).strftime("%Y-%m-%d")
            return article, scraped

        with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as pool:
            fetched_pairs = list(pool.map(fetch, filtered))

        scraped_count  = sum(1 for _, s in fetched_pairs if s)
        fallback_count = len(fetched_pairs) - scraped_count
        fetched_articles = [a for a, _ in fetched_pairs]
        print(f"  Full-text scraped : {scraped_count} | Summary fallback : {fallback_count}")

        # ── Phase 4: Local-Claude judge (one call per article) ────
        print(f"\n[Phase 4] Local-Claude judge ({JUDGE_MODEL}, {JUDGE_WORKERS} workers)...")
        all_signals = run_judge_articles(fetched_articles, COMPANY_NAME, target_sector["sector_name"])
        print(f"  Judged material : {len(all_signals)}")

        if USE_FINBERT and all_signals:
            print("\n[Phase 4b] FinBERT tone (second opinion)...")
            all_signals = add_finbert_tone(all_signals)

        # Clean internal scratch fields.
        for s in all_signals:
            s.pop("_ce_score", None)
            s.pop("full_text", None)

        rank_key = "confidence"

    else:
        # ── Legacy cosine-only path (A/B baseline) ────────────────
        import spacy
        nlp = spacy.load("en_core_web_sm")

        criteria_vectors = encoder.encode(criteria_texts, convert_to_tensor=True)

        print(f"\n[Phase 2] Parallel article scraping ({SCRAPE_WORKERS} threads)...")

        def fetch_legacy(meta):
            url     = meta.get("url", "")
            summary = meta.get("summary", "")
            text    = scrape_full_text(url, summary) if url else summary
            return meta, text

        with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as pool:
            fetched = list(pool.map(fetch_legacy, news_data))

        scraped_count  = sum(1 for _, t in fetched if t)
        fallback_count = 0

        print("\n[Phase 3] Entity filtering (spaCy)...")
        candidates = []
        for meta, raw_text in fetched:
            for para in extract_paragraphs(raw_text):
                doc = nlp(para)
                if is_primary_subject(para, COMPANY_NAME, doc):
                    candidates.append((para, meta))
        print(f"  Candidate paragraphs : {len(candidates)}")
        if not candidates:
            print("  No relevant paragraphs found. Try widening the date range.")
            return

        print("\n[Phase 4] Batch semantic encoding & matching...")
        para_texts   = [p for p, _ in candidates]
        para_vectors = encoder.encode(para_texts, batch_size=64, convert_to_tensor=True,
                                      show_progress_bar=True)
        sim_matrix = util.cos_sim(para_vectors, criteria_vectors)

        article_signal_counts = {}
        all_signals = []
        for i, (paragraph, meta) in enumerate(candidates):
            scores = sim_matrix[i]
            k = min(2, len(scores))
            top_scores, top_indices = torch.topk(scores, k=k)
            best_score   = top_scores[0].item()
            best_idx     = top_indices[0].item()
            second_score = top_scores[1].item() if k > 1 else 0.0
            if best_score < SIMILARITY_THRESHOLD:
                continue
            if (best_score - second_score) < MIN_CONFIDENCE_GAP:
                continue
            url = meta.get("url", "")
            if article_signal_counts.get(url, 0) >= MAX_SIGNALS_PER_ARTICLE:
                continue
            article_signal_counts[url] = article_signal_counts.get(url, 0) + 1
            all_signals.append({
                "date":              datetime.datetime.fromtimestamp(
                                         meta["datetime"]).strftime("%Y-%m-%d"),
                "headline":          meta.get("headline", ""),
                "url":               url,
                "risk_category":     criteria_labels[best_idx],
                "matched_criterion": criteria_texts[best_idx],
                "similarity_score":  round(best_score, 4),
                "confidence_gap":    round(best_score - second_score, 4),
                "extracted_chunk":   paragraph,
            })

        rank_key = "similarity_score"
        print(f"  Raw signals : {len(all_signals)}")

    # ── Phase 5: Deduplication & ranking ─────────────────────────
    print("\n[Phase 5] Deduplication & ranking...")

    unique_signals = deduplicate(all_signals, encoder)
    unique_signals.sort(key=lambda x: x.get(rank_key, 0), reverse=True)
    top_signals = unique_signals[:TOP_N_OUTPUT]

    print(f"  Unique signals : {len(unique_signals)} | Reporting : {len(top_signals)}")

    # ── Output ────────────────────────────────────────────────────
    print_report(top_signals, TICKER, COMPANY_NAME, target_sector["sector_name"])

    score_result = compute_credit_score(top_signals, END_DATE)
    print_score_summary(score_result, TICKER, COMPANY_NAME)

    output = {
        "ticker":            TICKER,
        "company_name":      COMPANY_NAME,
        "applied_sp_sector": target_sector["sector_name"],
        "period":            {"from": START_DATE, "to": END_DATE},
        "run_timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode":              "llm_judge" if USE_LLM_JUDGE else "cosine_only",
        "configuration": {
            "use_llm_judge":        USE_LLM_JUDGE,
            "judge_model":          JUDGE_MODEL if USE_LLM_JUDGE else None,
            "min_judge_confidence": MIN_JUDGE_CONFIDENCE if USE_LLM_JUDGE else None,
            "crossencoder_model":   CROSSENCODER_MODEL if USE_LLM_JUDGE else None,
            "crossencoder_top_n":   CROSSENCODER_TOP_N if USE_LLM_JUDGE else None,
            "use_finbert":          USE_FINBERT if USE_LLM_JUDGE else False,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "dedup_threshold":      DEDUP_THRESHOLD,
        },
        "stats": {
            "articles_fetched":    len(news_data),
            "after_ce_filter":     len(filtered) if USE_LLM_JUDGE else None,
            "full_text_scraped":   scraped_count,
            "summary_fallback":    fallback_count,
            "judged_material":     len(all_signals) if USE_LLM_JUDGE else None,
            "unique_signals":      len(unique_signals),
            "reported_signals":    len(top_signals),
        },
        "credit_score": score_result,
        "signals": top_signals,
    }

    os.makedirs("results", exist_ok=True)
    output_filename = os.path.join("results", f"credit_signals_{TICKER}_{END_DATE}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Full output written to: {output_filename}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
