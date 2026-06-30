"""
Credit Risk Extraction Pipeline
================================
Automated NLP system for quantitative credit research at Joywin International.
Scans financial news and extracts events material to a company's S&P credit rating.

Architecture
------------
Phase 1 — Data Ingestion & Dynamic Routing
    Fetches company news via Finnhub API. Maps Yahoo Finance industry to the
    correct S&P sector using a curated override dictionary and word-overlap scoring.

Phase 2 — Parallel Full-Text Article Scraping
    Downloads complete article text across multiple concurrent threads via
    requests + newspaper3k. Falls back to the Finnhub API summary on failure.

Phase 3 — Entity Filtering ("The Bouncer")
    Uses spaCy NLP to discard paragraphs where the target company is not a
    primary actor. Distinguishes between central subjects and attribution quotes.

Phase 4 — Batch Semantic Signal Extraction ("The Analyst")
    Encodes all candidate paragraphs in a single batched call (all-MiniLM-L6-v2),
    then computes the full similarity matrix against S&P criteria in one operation.
    A confidence-gap filter rejects ambiguous low-discrimination matches.

Phase 5 — Deduplication & Ranking
    Removes near-duplicate signals from multiple outlets covering the same event.
    Ranks remaining signals by relevance score and exports a structured JSON report.

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
    python -m spacy download en_core_web_sm
"""

import os
import requests
import yfinance as yf
import spacy
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
import json
import datetime
import torch


# ==========================================
# CONFIGURATION  — edit these before running
# ==========================================

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY_HERE")
TICKER          = "BAC"
COMPANY_NAME    = "Bank of America"

START_DATE      = "2026-05-01"
END_DATE        = "2026-06-30"

# --- Relevance thresholds (tune to adjust precision vs. recall) ---
SIMILARITY_THRESHOLD    = 0.42  # Minimum cosine score for a signal to be flagged
MIN_CONFIDENCE_GAP      = 0.04  # Top-1 score must exceed Top-2 by this margin
MIN_PARAGRAPH_WORDS     = 20    # Discard very short paragraph fragments
MAX_SIGNALS_PER_ARTICLE = 2     # Cap signals per article to prevent one event dominating
DEDUP_THRESHOLD         = 0.88  # Cosine score above which two signals are near-duplicates
TOP_N_OUTPUT            = 25    # Maximum signals to include in the final report
SCRAPE_WORKERS          = 15    # Parallel threads for HTTP article scraping


# ==========================================
# SECTOR OVERRIDES  (Yahoo Finance → S&P)
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
    """
    Maps a Yahoo Finance industry string to the best-matching S&P sector
    using word-overlap scoring. Falls back to the first sector if no match found.
    """
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
# PHASE 2 HELPERS: FULL-TEXT SCRAPING
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
    newspaper3k for local parsing — bypassing newspaper's own HTTP client which
    has no timeout and hangs indefinitely on paywalls or slow CDNs.

    Returns the Finnhub summary fallback if scraping fails or yields < 100 words.
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
    """Splits text into substantive paragraphs, discarding short fragments."""
    return [
        p.strip()
        for p in text.split("\n")
        if len(p.strip().split()) >= MIN_PARAGRAPH_WORDS
    ]


# ==========================================
# PHASE 3 HELPER: ENTITY FILTERING
# ==========================================

def is_primary_subject(paragraph, company_name, doc):
    """
    Returns True if the company is a PRIMARY actor in this paragraph — not
    merely an attribution source ("BofA analysts said...") or a passing mention.

    Accepts if ANY of the following hold:
      1. Company name appears >= 2 times (central topic).
      2. Company is the first ORG entity recognised by spaCy (lead actor).
      3. Company occupies a grammatical subject or direct-object role.
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
# PHASE 5 HELPERS: DEDUPLICATION & RANKING
# ==========================================

def deduplicate(signals, encoder):
    """
    Removes near-duplicate signals caused by multiple outlets reporting the same
    underlying event. When two signals exceed DEDUP_THRESHOLD in cosine similarity,
    only the higher-scoring one is kept.
    """
    if len(signals) < 2:
        return signals

    texts = [s["extracted_chunk"] for s in signals]
    vecs  = encoder.encode(texts, convert_to_tensor=True)
    keep  = [True] * len(signals)

    for i in range(len(signals)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(signals)):
            if not keep[j]:
                continue
            if util.cos_sim(vecs[i], vecs[j]).item() >= DEDUP_THRESHOLD:
                if signals[i]["similarity_score"] >= signals[j]["similarity_score"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [s for i, s in enumerate(signals) if keep[i]]


# ==========================================
# CONSOLE REPORT
# ==========================================

def print_report(signals, ticker, company_name, sector_name):
    """Prints a structured, analyst-readable summary of the extracted signals."""
    print("\n" + "=" * 65)
    print("  CREDIT RISK SIGNAL REPORT")
    print(f"  {company_name} ({ticker})  |  S&P Sector: {sector_name}")
    print("=" * 65)

    for category in ["Business Risk", "Financial Risk"]:
        cat_signals = [s for s in signals if s["risk_category"] == category]
        if not cat_signals:
            continue
        print(f"\n{'─' * 65}")
        print(f"  {category.upper()}  ({len(cat_signals)} signals)")
        print(f"{'─' * 65}")
        for i, sig in enumerate(cat_signals, 1):
            print(f"\n  [{i}]  {sig['date']}  |  Score: {sig['similarity_score']}"
                  f"  |  Gap: {sig['confidence_gap']}")
            print(f"  Headline  : {sig['headline'][:90]}")
            print(f"  Criterion : {sig['matched_criterion'][:110]}...")
            print(f"  Extract   : {sig['extracted_chunk'][:220]}...")
            if sig.get("url"):
                print(f"  Source    : {sig['url'][:90]}")
    print()


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_pipeline():
    print("=" * 65)
    print("  CREDIT RISK EXTRACTION PIPELINE  —  Joywin International")
    print("=" * 65)
    print(f"\n  Target : {COMPANY_NAME} ({TICKER})")
    print(f"  Period : {START_DATE}  ->  {END_DATE}\n")

    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        print("ERROR: Set the FINNHUB_API_KEY environment variable before running.")
        return

    print("Loading NLP models...")
    nlp     = spacy.load("en_core_web_sm")
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

    criteria_vectors = encoder.encode(criteria_texts, convert_to_tensor=True)

    finnhub_url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={TICKER}&from={START_DATE}&to={END_DATE}&token={FINNHUB_API_KEY}"
    )
    news_data = requests.get(finnhub_url).json()
    print(f"  Finnhub articles fetched : {len(news_data)}")

    # ── Phase 2: Parallel full-text scraping ──────────────────────
    print(f"\n[Phase 2] Parallel article scraping ({SCRAPE_WORKERS} threads)...")

    def fetch(meta):
        url     = meta.get("url", "")
        summary = meta.get("summary", "")
        if url:
            text = scrape_full_text(url, summary)
            return meta, text, text != summary
        return meta, summary, False

    with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as pool:
        fetched = list(pool.map(fetch, news_data))

    scraped_count  = sum(1 for _, _, full in fetched if full)
    fallback_count = len(fetched) - scraped_count
    print(f"  Full-text scraped : {scraped_count} | Summary fallback : {fallback_count}")

    # ── Phase 3: Entity filtering ─────────────────────────────────
    print("\n[Phase 3] Entity filtering (spaCy)...")

    candidates = []
    for meta, raw_text, _ in fetched:
        for para in extract_paragraphs(raw_text):
            doc = nlp(para)
            if is_primary_subject(para, COMPANY_NAME, doc):
                candidates.append((para, meta))

    print(f"  Candidate paragraphs : {len(candidates)}")

    if not candidates:
        print("  No relevant paragraphs found. Try widening the date range.")
        return

    # ── Phase 4: Batch encoding + matrix cosine similarity ────────
    # All paragraphs encoded in one call; one matrix op replaces N encode() calls.
    print("\n[Phase 4] Batch semantic encoding & matching...")

    para_texts   = [p for p, _ in candidates]
    para_vectors = encoder.encode(
        para_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True
    )
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

        url   = meta.get("url", "")
        count = article_signal_counts.get(url, 0)
        if count >= MAX_SIGNALS_PER_ARTICLE:
            continue
        article_signal_counts[url] = count + 1

        pub_date = datetime.datetime.fromtimestamp(
            meta["datetime"]
        ).strftime("%Y-%m-%d")

        all_signals.append({
            "date":              pub_date,
            "headline":          meta.get("headline", ""),
            "url":               url,
            "risk_category":     criteria_labels[best_idx],
            "matched_criterion": criteria_texts[best_idx],
            "similarity_score":  round(best_score, 4),
            "confidence_gap":    round(best_score - second_score, 4),
            "extracted_chunk":   paragraph,
        })

    print(f"  Raw signals : {len(all_signals)}")

    # ── Phase 5: Deduplication & ranking ─────────────────────────
    print("\n[Phase 5] Deduplication & ranking...")

    unique_signals = deduplicate(all_signals, encoder)
    unique_signals.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_signals = unique_signals[:TOP_N_OUTPUT]

    print(f"  Unique signals : {len(unique_signals)} | Reporting : {len(top_signals)}")

    # ── Output ────────────────────────────────────────────────────
    print_report(top_signals, TICKER, COMPANY_NAME, target_sector["sector_name"])

    output = {
        "ticker":            TICKER,
        "company_name":      COMPANY_NAME,
        "applied_sp_sector": target_sector["sector_name"],
        "period":            {"from": START_DATE, "to": END_DATE},
        "run_timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "configuration": {
            "similarity_threshold":    SIMILARITY_THRESHOLD,
            "min_confidence_gap":      MIN_CONFIDENCE_GAP,
            "max_signals_per_article": MAX_SIGNALS_PER_ARTICLE,
            "dedup_threshold":         DEDUP_THRESHOLD,
        },
        "stats": {
            "articles_fetched":  len(news_data),
            "full_text_scraped": scraped_count,
            "summary_fallback":  fallback_count,
            "raw_signals":       len(all_signals),
            "unique_signals":    len(unique_signals),
            "reported_signals":  len(top_signals),
        },
        "signals": top_signals,
    }

    output_filename = f"credit_signals_{TICKER}_{END_DATE}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Full output written to: {output_filename}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
