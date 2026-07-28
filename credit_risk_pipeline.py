"""
Credit Risk Extraction Pipeline
================================
Automated NLP system for quantitative credit research at Joywin International.
Scans financial news and extracts events material to a company's S&P credit rating.

Architecture
------------
Phase 1 -- Ingestion, Routing & Entity Gate
    Fetches news from Finnhub + GDELT (merged, de-duplicated by normalized headline).
    Routes the issuer to the correct S&P sector with an LLM sector router (Yahoo
    industry + business summary -> S&P sector; embedding/override fallback). An entity
    gate drops articles that never name the issuer, before any expensive step.

Phase 2 -- LLM Credit-Materiality Triage
    A cheap LLM (TRIAGE_MODEL via OpenRouter) scores every on-topic article 0-10 for
    credit materiality; articles scoring >= TRIAGE_MIN_SCORE are forwarded, capped at
    TRIAGE_MAX_KEEP. Replaces the cross-encoder, which rewarded generic money-language
    and buried real stories. Set USE_LLM_TRIAGE = False to fall back to the cross-encoder
    (cross-encoder/ms-marco-MiniLM-L6-v2, top CROSSENCODER_TOP_N).

Phase 3 -- Selective Full-Text Scraping
    Downloads full text only for the filtered articles: trafilatura -> newspaper3k ->
    Finnhub summary fallback, with a hard timeout. Resolves Finnhub redirect URLs to the
    real publisher page so stored links open correctly.

Phase 4 -- The Judge ("The Analyst")
    The real relevance decision. Each filtered article is judged against the sector's S&P
    criteria: is this a material credit event, good or bad for the bond, which S&P factor,
    and why. Returns event_summary, direction, confidence, rationale, and verbatim
    key_figures (verify_figures drops any number not present in the source text). Backend
    chosen at startup: OpenRouter (default, e.g. Haiku 4.5) or the local Claude Code CLI.

Phase 4b -- FinBERT Tone (secondary signal only)
    Adds a tone label on the judge's event_summary and flags tone_alignment = divergent.
    NOTE: the judge's credit direction (Phase 4) is authoritative. FinBERT has no credit
    context; it cannot distinguish debt issuance (bad for bondholders, sounds routine)
    from genuine recovery. Treat divergence as a prompt to re-read, not a correction.

Phase 5 -- Event Dedup, Scoring & Report
    Clusters same-event signals (reworded duplicates + cross-category splits) and
    reconciles them to one vote per event; conflicting reads net out; outlet count feeds
    a mild capped coverage weight. Computes a normalized [-1,+1] score with evidence
    shrinkage n/(n+SCORE_SHRINKAGE_K), a 5-band verdict + conviction label, and a
    bootstrap uncertainty band (>=4 signals). Also: market snapshot / priced-in check
    (5b), deterministic fundamentals signal from filings (5c), always-on financial panel,
    developing-news section, and a substance-ranked equity digest. Exports JSON.

A/B switch: set USE_LLM_JUDGE = False to revert to the legacy cosine-only path
(spaCy entity filter + bi-encoder paragraph scoring) for direct comparison.

Usage
-----
    Interactive:  python credit_risk_pipeline.py   (prompts for ticker/dates/judge)
    Batch:        python credit_risk_pipeline.py TICKER "Company" START END [JUDGE]
    Validation:   python backtest_harness.py       (labelled cases -> scoreboard)

Dependencies
------------
    pip install -r requirements.txt
    # Keys (env or .env): FINNHUB_API_KEY, plus OPENROUTER_API_KEY for the default
    #   triage + judge path. GDELT needs no key.
    # Optional Claude CLI judge backend: irm https://claude.ai/install.ps1 | iex
    # python -m spacy download en_core_web_sm   # only needed for the legacy cosine path
"""

import os
import re
import sys
import textwrap
import subprocess
import requests
import yfinance as yf
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import random
import datetime
import time
import torch

# Script directory. Resolve data/output paths against this, NOT the current working
# directory, so the pipeline works when launched from anywhere (IDE, cron, another cwd).
HERE = Path(__file__).resolve().parent


def _load_dotenv():
    """
    Minimal .env loader (no external dependency). Reads KEY=VALUE lines from a
    .env file in the script directory and populates os.environ for any key not
    already set in the real environment. Silently does nothing if .env is absent.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


# ==========================================
# CONFIGURATION  -- edit these before running
# ==========================================

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY_HERE")
TICKER          = "BA"
COMPANY_NAME    = "Boeing"

START_DATE      = "2026-05-01"
END_DATE        = "2026-06-30"

# --- Phase 2 relevance filter: LLM triage (default) or cross-encoder (fallback) ---
# The cross-encoder (ms-marco) ranked headline+summary against dense S&P prose criteria by
# SURFACE similarity -- it rewarded generic money/dollar language and was nearly insensitive
# to which criteria it was given (a "Crusoe $30B, $3B debt" headline out-ranked "Meta will
# spend $135B on AI" no matter the sector). Judging credit-materiality needs REASONING
# (capex -> leverage -> credit-negative), which only an LLM can do. So Phase 2 now runs a
# cheap LLM that scores each on-topic article 0-10 for credit-materiality; the survivors go
# to the full judge. Falls back to the cross-encoder if OpenRouter is unavailable.
USE_LLM_TRIAGE      = True
TRIAGE_MODEL        = "google/gemini-2.5-flash-lite"  # cheap OpenRouter model for triage
TRIAGE_MIN_SCORE    = 4      # keep articles scoring >= this (0-10) for the judge
                             # (4 = wider net; lets borderline credit-relevant stories reach
                             #  the judge, esp. on thin-coverage names. Capped by TRIAGE_MAX_KEEP.)
TRIAGE_MAX_KEEP     = 40     # hard cap forwarded to the judge (cost ceiling)
TRIAGE_BATCH_SIZE   = 60     # articles scored per triage call (headline+short summary each, so
                             # a big batch fits easily; halves triage calls on high-volume names)

CROSSENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSSENCODER_TOP_N  = 40    # articles forwarded to Claude after CE filter (fallback path)
                            # (raised 25->40: 2-3 material signals per run was too thin
                            #  for a reliable score; ~$0.05/issuer at Haiku-4.5 rates)
MAX_ARTICLE_CHARS   = 4000  # character limit for article text sent to the judge

# --- Legacy cosine gate (used ONLY when USE_LLM_JUDGE = False) ---
SIMILARITY_THRESHOLD    = 0.42
MIN_CONFIDENCE_GAP      = 0.04
MIN_PARAGRAPH_WORDS     = 20

# --- Shared pipeline knobs ---
MAX_SIGNALS_PER_ARTICLE = 2     # cap per article (legacy path)
DEDUP_THRESHOLD         = 0.88  # cosine score above which two signals are near-duplicates
TOP_N_OUTPUT            = 25    # maximum signals in the final report

# --- Event-level dedup (judge path): one event -> one vote, weighted by coverage ---
# Reworded copies from many outlets, and the same event split across Business/Financial
# Risk, previously each voted separately and inflated the score (the CE / UPS misses).
# We cluster same-event signals and collapse them to ONE reconciled vote; coverage (how
# many outlets carried it) becomes a mild weight, so a widely-covered event still counts
# more without being counted multiple times.
EVENT_CLUSTER_SIM   = 0.50   # event_summary cosine >= this (within the day window) = same event
EVENT_CLUSTER_DAYS  = 3      # only merge same-event reads dated within N days of each other
COVERAGE_WEIGHT_K   = 0.25   # coverage multiplier w = 1 + K*ln(outlets); 5 outlets ~ x1.40 (mild)
COVERAGE_WEIGHT_MAX = 1.75   # cap so one viral story cannot dominate the score
SCRAPE_WORKERS          = 8     # parallel threads for HTTP scraping. Lowered 15->8: trafilatura/
                                # newspaper both parse HTML via lxml, and heavy concurrent lxml
                                # parsing can SEGFAULT (native crash, uncatchable in Python). Fewer
                                # threads cuts that risk; the harness retry covers the rare case.
MAX_ARTICLES            = 300   # cap on Finnhub articles fetched; None = no cap

# --- Supplementary news sources (free, no API key), merged with the Finnhub feed ---
# GDELT DOC API is the recommended second source: query-scoped to the company (less noisy
# than Finnhub's ticker firehose) and returns REAL publisher URLs that scrape to full text.
# Google News RSS is available but OFF by default -- its base64 links don't resolve (judge
# sees only the snippet) and it is heavy on the company's own newsroom PR.
USE_FINNHUB     = True
USE_GDELT       = True
USE_GOOGLE_NEWS = False
GDELT_MAX       = 100   # cap on GDELT items pulled per run (API max 250)
GOOGLE_NEWS_MAX = 100   # cap on Google News items pulled per run
GDELT_CACHE_TTL = 43200 # seconds (12h) to reuse a cached GDELT response for the same
                        # ticker+window, so repeated runs don't re-hit GDELT's 1-req/5s limit
# GDELT returns at most 250 records PER request, but covers 2017->present with no total cap.
# Finnhub, by contrast, caps at 250 and refuses windows older than ~1 year. So for long or
# historical windows we slice the date range into GDELT_CHUNK_DAYS sub-windows and fetch each
# (250 each), merging -- far more than 250, evenly spread, as far back as needed. GDELT throttles
# to ~1 req/5s, so each extra slice adds ~5s; GDELT_TOTAL_MAX caps the merged result.
GDELT_PAGINATE   = True
GDELT_CHUNK_DAYS = 30   # slice size; a window longer than this is fetched in multiple requests
GDELT_TOTAL_MAX  = 300  # overall cap on merged GDELT items across all slices (runaway guard)

# --- Bootstrap uncertainty band (Phase 5 scoring) ---
BOOTSTRAP_RESAMPLES     = 1000  # resamples of the signal set for the score band
MIN_BOOTSTRAP_SIGNALS   = 4     # below this the band is not statistically meaningful
BOOTSTRAP_SEED          = 42    # fixed seed -> reproducible band for the same signals

# --- Market snapshot as of end date (Phase 5b) ---
USE_MARKET_SNAPSHOT     = True
MARKET_BENCHMARK        = "^GSPC"   # benchmark for the window abnormal-return calc

# --- Deterministic fundamentals signal from quarterly filings (Phase 5c) ---
# A credit signal built straight from yfinance quarterly statements -- no news, no LLM.
# Votes across revenue / EBITDA margin / free cash flow / net leverage (see
# compute_fundamentals_signal): NEGATIVE on rising leverage, POSITIVE on broad corroborated
# improvement. Fires even when the news channel returns nothing, so fundamentally strong but
# quiet names (e.g. SanDisk) are not scored blank.
USE_FILINGS_FR          = True

# --- Always-on quarterly fundamentals panel (descriptive, not scored) ---
# Prints the key credit metrics and their trend every run (revenue, net income, margins,
# EBITDA, free cash flow, debt, net debt) from yfinance quarterly statements -- so the
# analyst always sees the financial picture even when no signal fires.
USE_FINANCIAL_PANEL      = True
FINANCIAL_PANEL_QUARTERS = 4     # number of recent quarters shown side by side

# --- Point-in-time financials (Phase 1) ---
# Use the latest quarterly statement whose period-end is on/before END_DATE. yfinance
# only returns quarters that have actually been reported, so for a live run this is the
# newest public filing (e.g. Micron Q3 once earnings are out). Set >0 to re-impose a
# reporting-lag guard for strict point-in-time backtests (accepts mild look-ahead at 0).
FINANCIALS_REPORTING_LAG_DAYS = 0

# --- LLM judge (Phase 4) ---
# The judge can run through three backends, chosen at runtime:
#   "claude"     -> local Claude Code CLI (claude -p); uses your subscription allowance
#   "openrouter" -> OpenRouter HTTP API (pay-per-token); key from OPENROUTER_API_KEY
#   "gemini"     -> local Gemini CLI (gemini -m); $0 marginal cost on the user's OAuth
#                   Google quota (no API key). Also serves triage + routing on this path,
#                   so a gemini run needs no OPENROUTER_API_KEY at all. See _gemini_chat.
USE_LLM_JUDGE        = True
JUDGE_BACKEND        = "claude"   # set interactively at startup
JUDGE_MODEL          = "haiku"    # meaning depends on backend (CLI alias or OpenRouter slug)
JUDGE_WORKERS        = 6    # raised 4->6 to keep latency flat with TOP_N at 40
JUDGE_TIMEOUT        = 120
MIN_JUDGE_CONFIDENCE = 0.65  # was 0.72; 0.65 keeps the credit signal rigorous, slightly more inclusive

# Verdict cache -- the reliability lever. Each article's judge verdict is persisted, so re-running
# the same issuer/window reuses prior judgments: IDENTICAL output run-to-run AND no repeat LLM
# calls (also saves quota). Keyed by company + article + criterion + judge model + prompt version.
# Bump _JUDGE_PROMPT_VERSION to invalidate every cached verdict after a prompt/rubric change.
VERDICT_CACHE         = True
_JUDGE_PROMPT_VERSION = "v2"   # v2: materiality gated on mapping to an S&P rating factor
                               # (drops lawsuits/activist/governance noise). Bumping invalidates v1.

# --- Recall-recovery loop (Phases 2-4) ---
# The single-pass pipeline scored blank on quiet issuers (the backtest's upgrade-side misses
# were all thin-signal: CQP n=0, MOG.A/WBD n=1, RBLX/LADR n=2). When the judge returns fewer
# than LOOP_MIN_MATERIAL material signals, re-search the news feed with expanded credit-risk
# terms and judge ONLY the newly found articles, up to LOOP_MAX_ROUNDS extra rounds. This is a
# general RECALL improvement (find more real news), not tuning toward any backtest label.
# Stops early when a round adds nothing new, so a genuinely quiet name doesn't loop forever.
USE_RECALL_LOOP     = True
LOOP_MIN_MATERIAL   = 4      # loop while material-signal count < this
LOOP_MAX_ROUNDS     = 2      # extra search+judge rounds after the first pass
LOOP_RELAX_TRIAGE   = 3      # from round 2 on, admit triage score >= this (vs TRIAGE_MIN_SCORE)

# --- Adversarial verification (Phase 4c) ---
# Precision guard for the loop: after signals are gathered, a skeptic pass asks the judge to
# REFUTE the credit-materiality of each material signal. Signals that fail the challenge are
# demoted to near-misses rather than scored. Keeps the wider recall net from admitting noise.
USE_ADVERSARIAL_VERIFY = False   # DISABLED: the skeptic pass proved volatile (demoted 4 signals
                                 # one run, 0 the next -> score swung 0.15) and it fought the recall
                                 # loop by dropping signals it had just found. The judge's 0.65
                                 # confidence bar is the precision gate. Matches the earlier finding
                                 # that a down-biasing self-critique hurt more than it helped.

# Broad credit-risk vocabulary for the recall loop's expanded news search (OR-joined with the
# company name). General bondholder-relevant terms; sector-specific terms are mined separately
# from the routed sector's S&P risk prose (see _sector_search_terms).
CREDIT_RISK_SEARCH_TERMS = [
    "credit rating", "downgrade", "upgrade", "outlook", "creditwatch",
    "refinancing", "debt maturity", "covenant", "leverage", "liquidity",
    "restructuring", "default", "bond", "notes offering", "cash flow",
    "guidance", "impairment", "acquisition", "dividend", "buyback",
]

# --- Sector routing ---
# Ask the judge LLM to map the Yahoo Finance industry (+ business summary) to the single
# best S&P sector. Replaces the embedding match that mis-routed some names on superficial
# word overlap (Stanley Black & Decker -> "Forest And Paper Products"; Brown-Forman likewise
# via cooperage/barrel language). Falls back to the embedding/override router if the LLM is
# unavailable or returns an off-list answer. One cheap extra LLM call per run.
USE_LLM_SECTOR_ROUTING = True

# SECONDARY equity/market news section only (NOT credit-scored). Credit-relevant news IS the
# primary BR/FR signal report; it is not repeated here. Shows the top few equity stories by
# coverage, each with a one-line summary. SHOW_EQUITY_NEWS hides it (CLI 6th arg "noequity").
SHOW_EQUITY_NEWS     = True
DIGEST_TOP_EQUITY    = 5      # equity/market stories shown (secondary), ranked by substance
DIGEST_CLUSTER_SIM   = 0.60   # headline cosine >= this = same story, different outlet

# "Developing / reviewed" section: credit-relevant stories the judge read but that were not
# (yet) a rating event -- the forward-looking business news (capex plans, strategic shifts,
# deals in progress). Shown on EVERY run so a report always surfaces >= this many stories,
# even at zero material signals. Context only -- these do NOT enter the credit score.
DEVELOPING_NEWS_N    = 3

# Runtime judge menu. Each entry: (label, backend, model_id).
# OpenRouter cost estimates are per issuer (~25 articles); see README for method.
JUDGE_CHOICES = [
    ("Claude Haiku 4.5  -- OpenRouter, ~$0.03/issuer  (recommended)", "openrouter", "anthropic/claude-haiku-4.5"),
    ("Claude Haiku      -- CLI, subscription",                        "claude",     "haiku"),
    ("Claude Sonnet     -- CLI, subscription (higher accuracy)",      "claude",     "sonnet"),
    ("Claude 3 Haiku    -- OpenRouter, ~$0.014/issuer (older model)", "openrouter", "anthropic/claude-3-haiku"),
    ("DeepSeek V3.1     -- OpenRouter, ~$0.011/issuer",               "openrouter", "deepseek/deepseek-chat-v3.1"),
    ("Gemini 2.5 Flash-Lite -- OpenRouter, ~$0.005/issuer",          "openrouter", "google/gemini-2.5-flash-lite"),
    ("GPT-4o-mini       -- OpenRouter, ~$0.008/issuer",              "openrouter", "openai/gpt-4o-mini"),
    ("Gemini 2.5 Pro    -- local CLI, $0 (OAuth quota; needs gemini CLI + GOOGLE_CLOUD_PROJECT)",
                                                                     "gemini",     "gemini-2.5-pro"),
]

# --- OpenRouter API (used when JUDGE_BACKEND == "openrouter") ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# --- Gemini CLI ($0 backend; used when JUDGE_BACKEND == "gemini") ---
# Inference runs on the user's OAuth Google login via the gemini CLI subprocess (no API key,
# no per-token bill -- quota/rate-limited). Judge uses GEMINI_JUDGE_MODEL; the cheaper
# triage/routing calls use GEMINI_TRIAGE_MODEL. See memory: gemini-cli-zero-cost-backend.
GEMINI_JUDGE_MODEL  = "gemini-2.5-pro"     # carried via JUDGE_MODEL when the CLI entry is chosen
GEMINI_TRIAGE_MODEL = "gemini-2.5-flash"   # cheaper/faster model for triage + sector routing
GEMINI_TIMEOUT      = 240                  # per-call subprocess timeout (s). Bounds a hung npm-shim
                                           # call to 4 min (was 600 = 10 min, a per-issuer time bomb);
                                           # measured Pro calls run ~30s, tail ~150s, so 240s is safe.
GEMINI_MAX_WORKERS  = 4                    # cap parallel gemini subprocesses (quota/rate friendly)
GEMINI_MAX_RETRIES  = 4                    # extra attempts on rate-limit/quota (429/RESOURCE_EXHAUSTED)
GEMINI_BACKOFF_BASE = 8                    # backoff seconds; wait = BASE * 2**attempt (8,16,32,64)
# Gemini is capped on NUMBER of requests, not tokens, so we pack many items per call. The judge
# (1 call/article) and adversarial verify (1 call/signal) are the big consumers -- batching them
# cuts per-issuer calls ~4-6x. Batch sizes are a quota-vs-per-item-attention trade; keep modest.
GEMINI_JUDGE_BATCH  = 3                    # articles judged per Gemini call (1 = per-article).
                                           # A/B tested on WBD: batch=3 keeps judge confidence
                                           # decisive (~0.95, finds all material signals) while
                                           # cutting calls ~3x; batch=8 collapsed confidence to
                                           # ~0.57 and MISSED material signals -- do NOT raise it.
GEMINI_VERIFY_BATCH = 12                   # signals challenged per adversarial-verify Gemini call

# --- Evidence shrinkage for the credit score ---
# score = raw_score x n/(n+K). Pulls the score toward 0 when it rests on few signals,
# so a unanimous 2-signal run no longer pins the -1.0/+1.0 ceiling. With K=2:
# n=2 -> x0.50, n=3 -> x0.60, n=8 -> x0.80. Fixes "LOW-conviction -1.0 looks as
# strong as HIGH-conviction -1.0" without touching the band boundaries.
SCORE_SHRINKAGE_K = 2

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

def yf_symbol(ticker):
    """yfinance uses '-' for share classes where Finnhub / RIC feeds use '.'
    (BF.B -> BF-B, MOG.A -> MOG-A). Finnhub keeps the dot; yfinance needs the dash."""
    return (ticker or "").replace(".", "-")


def match_sp_sector(industry, business_summary, sp_criteria_master, encoder):
    """
    Route the company to the S&P sector whose credit criteria best describe its actual
    business. Builds a query from the company's business summary (+ Yahoo industry) and
    semantically matches it against each S&P sector document (sector name + all its
    business/financial risk criteria). This fixes conglomerates a single Yahoo industry
    label routes poorly -- e.g. Amazon ("Internet Retail") whose criteria span retail and
    cloud, previously hard-forced to "Retail And Restaurants".

    Returns (best_sector_dict, ranked_top3) where ranked_top3 is [(sector_name, score)]
    for transparency. Falls back to legacy name word-overlap if summary/encoder missing.
    """
    sectors = sp_criteria_master["sectors"]

    query = " ".join(x for x in (business_summary or "", industry or "") if x).strip()
    if query and encoder is not None:
        try:
            sector_docs = [
                s["sector_name"] + ". " + " ".join(
                    s.get("business_risk_keywords", []) + s.get("financial_risk_keywords", []))
                for s in sectors
            ]
            q_vec  = encoder.encode(query[:2000], convert_to_tensor=True)
            d_vecs = encoder.encode(sector_docs, convert_to_tensor=True)
            sims   = util.cos_sim(q_vec, d_vecs)[0]
            ranked = sorted(
                ((sectors[i], float(sims[i])) for i in range(len(sectors))),
                key=lambda x: x[1], reverse=True,
            )
            return ranked[0][0], [(s["sector_name"], round(sc, 3)) for s, sc in ranked[:3]]
        except Exception:  # noqa: BLE001
            pass

    # Fallback: legacy name word-overlap via the override map.
    mapped      = SECTOR_OVERRIDES.get(industry, industry)
    best_score  = 0
    best_sector = sectors[0]
    for sector in sectors:
        yf_words = set(mapped.replace("&", "").replace(",", "").split())
        sp_words = set(sector["sector_name"].replace("And", "").replace(",", "").split())
        score    = len(yf_words & sp_words)
        if score > best_score:
            best_score  = score
            best_sector = sector
    return best_sector, [(best_sector["sector_name"], best_score)]


def build_sector_routing_prompt(company_name, industry, business_summary, sector_names):
    joined = "\n".join(f"- {n}" for n in sector_names)
    return (
        "You map a company to the single S&P sector whose credit-rating criteria best fit "
        "its core business.\n"
        f"COMPANY: {company_name}\n"
        f"YAHOO FINANCE INDUSTRY: {industry}\n"
        f"BUSINESS SUMMARY: {(business_summary or '')[:800]}\n\n"
        "Pick the ONE best sector from the list below. Judge by what the company actually "
        "does, not superficial word overlap (e.g. a toolmaker is Capital Goods, not Forest "
        "& Paper Products just because it sells to builders).\n"
        f"S&P SECTORS:\n{joined}\n\n"
        "Respond with ONLY compact JSON and nothing else: "
        '{"sector": "<exact sector name copied from the list>"}'
    )


def match_sp_sector_llm(company_name, industry, business_summary, sp_criteria_master):
    """
    Ask the configured judge LLM to map the company to the best-matching S&P sector.
    Returns (sector_dict, [(sector_name, 1.0)]), or None on any failure so the caller can
    fall back to the embedding/override router. Reuses the judge backend; one cheap call.
    Matches the model's answer to a real sector name (exact -> case-insensitive -> substring)
    so an off-list or slightly reworded answer never silently routes to the wrong bucket.
    """
    sectors = sp_criteria_master["sectors"]
    names   = [s["sector_name"] for s in sectors]
    prompt  = build_sector_routing_prompt(company_name, industry, business_summary, names)
    try:
        if JUDGE_BACKEND == "gemini":
            # Sector routing is a light judgment -- run it on Flash (loose quota), not the
            # quota-scarce 2.5-Pro judge model.
            verdict = _extract_json_obj(_gemini_chat(prompt, GEMINI_TRIAGE_MODEL))
        else:
            verdict = _judge_call_fn()(prompt)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(verdict, dict):
        return None
    choice = str(verdict.get("sector", "")).strip()
    if not choice:
        return None

    low = choice.lower()
    picked = next((s for n, s in zip(names, sectors) if n == choice), None)
    if picked is None:
        picked = next((s for n, s in zip(names, sectors) if n.lower() == low), None)
    if picked is None:
        picked = next((s for n, s in zip(names, sectors)
                       if low in n.lower() or n.lower() in low), None)
    if picked is None:
        return None
    return picked, [(picked["sector_name"], 1.0)]


# ==========================================
# PHASE 1 HELPER: ENTITY (IS-IT-ABOUT-US) GATE
# ==========================================

_COMPANY_NAME_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "ltd", "limited",
    "holdings", "holding", "group", "plc", "nv", "sa", "ag", "technologies",
    "technology", "com", "the", "and", "systems", "international",
}


def build_entity_matcher(company_name, ticker):
    """
    Return a predicate article -> bool that decides whether an article is actually ABOUT
    this issuer -- i.e. the issuer is the PRIMARY SUBJECT, not a passing mention. Aliases =
    significant tokens of the company name (suffixes like Inc/Corp/.com stripped, matched
    as whole words); the ticker is matched separately as a CASE-SENSITIVE whole word so
    short tickers (BA, CE, MU) don't match lowercase substrings ("ba" in "bank").

    Finnhub tags a lot of generic market commentary to mega-cap tickers (e.g. ~66% of
    Amazon's feed never names Amazon; a "CoreWeave buying opportunity" story that merely
    mentions Meta). A plain does-the-word-appear test lets that noise fill the Phase 2
    top-N and waste the judge budget -- 39/40 get rejected downstream as immaterial. So the
    gate keeps an article only when the issuer OWNS it:

        - named in the HEADLINE (headlines name their subject), OR
        - named >= 2 times across headline + summary (repeated => genuinely discussed).

    A single body-only mention (the classic passing reference) is dropped here, before the
    cross-encoder ever sees it. run_pipeline keeps the full feed as a fallback if this
    matches nothing (name-spelling mismatch), so a strict gate can never zero out coverage.
    """
    tokens  = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", company_name or "") if t]
    aliases = [t for t in tokens if len(t) >= 3 and t not in _COMPANY_NAME_SUFFIXES]
    name_re = (re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b",
                          re.IGNORECASE) if aliases else None)
    ticker_re = re.compile(r"\b" + re.escape(ticker) + r"\b") if ticker else None

    def _count(text):
        n = 0
        if name_re:
            n += len(name_re.findall(text))
        if ticker_re:
            n += len(ticker_re.findall(text))
        return n

    def is_about(article):
        headline = article.get("headline", "") or ""
        summary  = article.get("summary", "") or ""
        in_headline = bool((name_re and name_re.search(headline))
                           or (ticker_re and ticker_re.search(headline)))
        return in_headline or _count(f"{headline} {summary}") >= 2

    return is_about


# ==========================================
# PHASE 1 HELPER: SUPPLEMENTARY NEWS (GOOGLE NEWS RSS)
# ==========================================

def fetch_google_news(company_name, ticker, start_date, end_date, max_items=GOOGLE_NEWS_MAX):
    """
    Supplementary free news via Google News RSS (no API key). Returns Finnhub-shaped dicts
    {headline, summary, url, datetime, source} so results merge straight into the pipeline.
    The query is scoped to the company + date window, so it is much less noisy than Finnhub's
    ticker-tagged feed. Google links are redirects; the scraper resolves them to the real
    publisher URL like it does Finnhub's. Best-effort: returns [] on any failure.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    from urllib.parse import quote

    query = f"{company_name} after:{start_date} before:{end_date}"
    url   = ("https://news.google.com/rss/search?q=" + quote(query)
             + "&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, timeout=12, headers=_SCRAPE_HEADERS)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:  # noqa: BLE001
        return []

    out = []
    for it in root.iter("item"):
        title  = (it.findtext("title") or "").strip()
        link   = (it.findtext("link") or "").strip()
        desc   = it.findtext("description") or ""
        pub    = it.findtext("pubDate") or ""
        src_el = it.find("source")
        source = (src_el.text if src_el is not None else "") or ""

        # Google News titles are "Headline - Publisher"; split out the publisher.
        headline = title
        if source and title.endswith(f"- {source}"):
            headline = title[: -(len(source) + 2)].strip(" -")
        elif not source and " - " in title:
            headline, _, source = title.rpartition(" - ")

        summary = re.sub(r"<[^>]+>", " ", desc)
        summary = re.sub(r"\s+", " ", summary).strip()[:300]
        try:
            epoch = int(parsedate_to_datetime(pub).timestamp())
        except Exception:  # noqa: BLE001
            epoch = 0

        if not headline:
            continue
        out.append({
            "headline": headline,
            "summary":  summary or headline,
            "url":      link,
            "datetime": epoch,
            "source":   source or "Google News",
        })
        if len(out) >= max_items:
            break
    return out


def _core_company_name(company_name):
    """Company name with corporate suffixes stripped (Inc/Corp/Corporation/Holdings/...), so
    a news search matches how articles actually refer to the issuer -- 'SanDisk', not the
    exact phrase 'Sandisk Corporation' (which few articles use)."""
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", company_name or "") if t]
    core = [t for t in toks if t.lower() not in _COMPANY_NAME_SUFFIXES]
    return " ".join(core) or (company_name or "").strip()


def _sector_search_terms(sector, limit=6):
    """Short, searchable credit terms mined from a routed sector's dense S&P risk prose.
    The JSON stores long analyst sentences (not query terms), so we pull the credit-loaded
    phrases and acronyms actually present in THIS sector's business+financial risk text
    (e.g. 'working capital', 'FOCF', 'leverage') to flavour the recall search by sector."""
    text = " ".join((sector.get("business_risk_keywords", []) or [])
                    + (sector.get("financial_risk_keywords", []) or []))
    low  = text.lower()
    terms = []
    for m in re.findall(r"\b[A-Z]{2,5}\b", text):        # acronyms: FOCF, FFO, EBITDA, OEM, R&D
        if m not in terms:
            terms.append(m)
    lexicon = ["working capital", "free operating cash flow", "free cash flow",
               "operating cash flow", "capital expenditure", "leverage", "liquidity",
               "margin", "refinancing", "covenant", "supply chain", "demand", "pricing",
               "capacity", "tariff", "impairment", "hedge", "reserves", "backlog"]
    for t in lexicon:
        if t in low and t not in terms:
            terms.append(t)
    return terms[:limit]


def build_recall_query(company_name, sector, round_idx):
    """GDELT query terms for a recall-loop round: the broad credit vocabulary plus this
    sector's mined terms. round_idx widens the net (round 1 = core credit terms; round 2+
    adds the sector-specific terms). Returned as a list; fetch_gdelt_news OR-joins them."""
    terms = list(CREDIT_RISK_SEARCH_TERMS)
    if round_idx >= 2:
        terms += _sector_search_terms(sector)
    seen, out = set(), []
    for t in terms:                                      # de-dup, preserve order
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _gdelt_cache_path(ticker, start_date, end_date, tag=""):
    d = Path(__file__).resolve().parent / ".cache"
    try:
        d.mkdir(exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    suffix = f"_{tag}" if tag else ""
    return d / f"gdelt_{ticker}_{start_date}_{end_date}{suffix}.json"


def _date_chunks(start_date, end_date, days):
    """Split an inclusive [start, end] range (YYYY-MM-DD) into consecutive sub-windows of at
    most `days` days each, newest-first. Lets GDELT be paged past its 250-records/request cap
    and back beyond Finnhub's ~1-year horizon. Falls back to one window on a parse error."""
    fmt = "%Y-%m-%d"
    try:
        s = datetime.datetime.strptime(start_date, fmt).date()
        e = datetime.datetime.strptime(end_date, fmt).date()
    except ValueError:
        return [(start_date, end_date)]
    if e < s or days < 1:
        return [(start_date, end_date)]
    chunks, cur_end, one = [], e, datetime.timedelta(days=1)
    while cur_end >= s:
        cur_start = max(s, cur_end - datetime.timedelta(days=days - 1))
        chunks.append((cur_start.strftime(fmt), cur_end.strftime(fmt)))
        cur_end = cur_start - one
    return chunks


def fetch_gdelt_news(company_name, ticker, start_date, end_date, max_items=GDELT_MAX,
                     extra_terms=None):
    """
    Supplementary free news via the GDELT DOC 2.0 API (no API key). Returns Finnhub-shaped
    dicts with REAL publisher URLs (unlike Google News' base64 redirects), so the judge can
    read full article text. English sources only; date-windowed. GDELT carries no per-article
    snippet, so the title doubles as the summary.

    Searches the CORE company name (suffixes stripped) to match how articles refer to the
    issuer, and caches successful responses per ticker+window (GDELT_CACHE_TTL) so repeated
    runs don't re-hit GDELT's strict 1-request/5s limit. Best-effort: returns [] on failure.

    extra_terms (recall loop): a list of credit-risk terms OR-joined into the query to surface
    credit-relevant stories the plain-name search missed. Cached under a separate terms-hashed
    key so an expanded query never collides with (or overwrites) the base-name cache.
    """
    from urllib.parse import quote

    # Terms-aware cache tag: base search and each expanded search cache independently.
    tag = ""
    if extra_terms:
        import hashlib
        digest = hashlib.md5("|".join(sorted(extra_terms)).encode("utf-8")).hexdigest()[:8]
        tag = f"x{digest}"

    # When paginating, the merged result can far exceed the old per-request cap; total_cap is the
    # ceiling across all slices (and the slice we serve from cache), so it must gate cache-serve.
    total_cap = GDELT_TOTAL_MAX if GDELT_PAGINATE else max_items

    # Serve from cache when fresh -- avoids re-hitting the rate limit on repeated runs.
    cache = _gdelt_cache_path(ticker, start_date, end_date, tag=tag)
    try:
        if cache.exists() and (time.time() - cache.stat().st_mtime) < GDELT_CACHE_TTL:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached:
                for a in cached:                 # backfill tag for caches written before it existed
                    a.setdefault("provider", "gdelt")
                return cached[:total_cap]
    except Exception:  # noqa: BLE001
        pass

    def _stamp(d, tail):
        return d.replace("-", "") + tail

    core = _core_company_name(company_name)
    if extra_terms:
        # OR-group the credit terms; quote multi-word phrases so GDELT treats them atomically.
        ors   = " OR ".join(f'"{t}"' if " " in t else t for t in extra_terms)
        query = f'"{core}" ({ors}) sourcelang:english'
    else:
        query = f'"{core}" sourcelang:english'

    def _fetch_window(s_date, e_date, per_max):
        """One GDELT request for a single date sub-window -> Finnhub-shaped dicts (or [])."""
        url = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + quote(query)
               + "&mode=ArtList&format=json&sort=DateDesc"
               + f"&maxrecords={min(250, max(1, per_max))}"
               + f"&startdatetime={_stamp(s_date, '000000')}"
               + f"&enddatetime={_stamp(e_date, '235959')}")
        arts = []
        for attempt in (1, 2):   # GDELT ~1 req/5s; retry once after a pause on HTTP 429
            try:
                r = requests.get(url, timeout=20, headers=_SCRAPE_HEADERS)
                if r.status_code == 429:
                    if attempt == 1:
                        time.sleep(5)
                        continue
                    return []
                r.raise_for_status()
                arts = r.json().get("articles", []) or []
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return []
                time.sleep(5)
        window = []
        for a in arts:
            title = (a.get("title") or "").strip()
            link  = (a.get("url") or "").strip()
            if not title or not link:
                continue
            sd = (a.get("seendate") or "").replace("T", "").replace("Z", "")
            try:
                epoch = int(datetime.datetime.strptime(sd[:14], "%Y%m%d%H%M%S").timestamp())
            except Exception:  # noqa: BLE001
                epoch = 0
            window.append({
                "headline": title,
                "summary":  title,          # GDELT has no snippet; title doubles as summary
                "url":      link,
                "datetime": epoch,
                "source":   a.get("domain", "") or "GDELT",
                "provider": "gdelt",        # query-scoped to the company phrase -> trust in the gate
            })
        return window

    # Slice the window when it spans more than one chunk (breaks the 250/request ceiling and
    # reaches back years); otherwise a single request. Sleep 5s between slices for the rate limit.
    chunks = (_date_chunks(start_date, end_date, GDELT_CHUNK_DAYS)
              if GDELT_PAGINATE else [(start_date, end_date)])
    out = []
    for i, (cs, ce) in enumerate(chunks):
        if len(out) >= total_cap:
            break
        if i > 0:
            time.sleep(5)                        # respect GDELT's ~1 req/5s limit between slices
        got = _fetch_window(cs, ce, total_cap - len(out))
        out = merge_news(out, got)
        if i > 0 and not got:                    # older slice empty -> no more history; stop early
            break
    out = out[:total_cap]

    if out:  # cache only successful (non-empty) fetches so a throttled run retries next time
        try:
            cache.write_text(json.dumps(out), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return out


def merge_news(*sources):
    """
    Merge news lists from multiple providers, de-duplicating by normalized headline so the
    same story carried by two feeds counts once. Preserves first-seen order (pass the
    provider you trust most first).
    """
    seen, out = set(), []
    for src in sources:
        for a in (src or []):
            key = re.sub(r"\W+", "", (a.get("headline", "") or "")).lower()[:90]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(a)
    return out


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


def filter_with_crossencoder(news_data, criteria_texts, criteria_labels, top_n=CROSSENCODER_TOP_N):
    """
    Scores each article's headline + summary against every S&P criterion for the
    sector using a cross-encoder. Returns the top `top_n` articles by max criterion
    score, each annotated with matched_criterion and risk_category. Pass top_n =
    len(news_data) to annotate+sort without dropping any (judge-all mode).

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
    return results[:top_n]


# ==========================================
# PHASE 2 (DEFAULT): LLM CREDIT-MATERIALITY TRIAGE
# ==========================================

def build_triage_prompt(company, sector_name, sector_factors, batch):
    """Batch prompt: score each article 0-10 for credit-rating materiality to THIS issuer."""
    lines = []
    for i, a in enumerate(batch):
        headline = (a.get("headline", "") or "").strip()
        summary  = (a.get("summary", "") or "").strip()[:200]
        lines.append(f"{i}. {headline} -- {summary}")
    return (
        "You are an S&P credit analyst triaging news for BONDHOLDER (credit-rating) relevance, "
        "NOT equity upside.\n"
        f"TARGET COMPANY: {company}   |   S&P SECTOR: {sector_name}\n\n"
        "Score EACH article 0-10 by how likely it describes a MATERIAL CREDIT event for THIS "
        "company -- something bearing on leverage, debt, liquidity, cash flow, margins, "
        "refinancing, covenants, or a major financial/strategic shift such as a large "
        "debt-funded capex or acquisition program.\n"
        "SCORING:\n"
        "  0-2 : pure stock-price move, analyst rating/price target, hype, or the company is "
        "only a bystander/passing mention\n"
        "  3-4 : tangential or vague; company-specific credit impact unclear\n"
        "  5-7 : plausible credit-relevant event, some financial substance\n"
        "  8-10: clearly material credit event with concrete financial substance "
        "(debt, capex scale, cash-flow/margin/leverage change)\n\n"
        f"SECTOR RISK FACTORS (what matters for credit here):\n{sector_factors}\n\n"
        f"ARTICLES:\n" + "\n".join(lines) + "\n\n"
        'Respond with ONLY compact JSON, one entry per article id above, no prose:\n'
        '{"scores":[{"id":0,"s":7},{"id":1,"s":2}]}'
    )


def _triage_model_id():
    return GEMINI_TRIAGE_MODEL if JUDGE_BACKEND == "gemini" else TRIAGE_MODEL


def _triage_key(company, sector_name, article):
    """Cache key for one article's triage score (everything the score depends on)."""
    import hashlib
    raw = "|".join([company or "", sector_name or "", article.get("url", "") or "",
                    article.get("headline", "") or "", _triage_model_id(), _JUDGE_PROMPT_VERSION])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_cached_triage(company, sector_name, article):
    if not VERDICT_CACHE:
        return None
    p = HERE / ".cache" / "triage" / f"{_triage_key(company, sector_name, article)}.json"
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("s")
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_triage(company, sector_name, article, score):
    if not VERDICT_CACHE:
        return
    d = HERE / ".cache" / "triage"
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_triage_key(company, sector_name, article)}.json").write_text(
            json.dumps({"s": score}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def triage_articles(articles, company, sector_name, criteria_texts):
    """
    Cheap-LLM relevance triage (replaces the cross-encoder). Scores every on-topic article
    0-10 for credit-materiality via a cheap model, in parallel batches, and returns them
    annotated with `_triage_score`, sorted high-to-low. Each article is handed the FULL set
    of sector criteria as its judge anchor (the judge decides the final risk_category itself).

    Triage scores are cached per article (VERDICT_CACHE): the same article always gets the same
    score, so the top-N selection is reproducible run-to-run -- which, with the verdict cache,
    makes a re-run of the same issuer deterministic. Only uncached articles hit the LLM.
    """
    sector_factors = "\n".join(f"- {c}" for c in criteria_texts)
    all_criteria   = "\n".join(criteria_texts)

    # Split into cached (reuse score) vs uncached (need an LLM score).
    scores   = {}            # index in `articles` -> triage score
    to_score = []            # [(orig_index, article), ...]
    for i, a in enumerate(articles):
        s = _load_cached_triage(company, sector_name, a)
        if s is not None:
            scores[i] = s
        else:
            to_score.append((i, a))

    if to_score:
        items   = [a for _, a in to_score]
        batches = [items[j:j + TRIAGE_BATCH_SIZE] for j in range(0, len(items), TRIAGE_BATCH_SIZE)]

        def run_batch(indexed_batch):
            bi, batch = indexed_batch
            prompt = build_triage_prompt(company, sector_name, sector_factors, batch)
            try:
                if JUDGE_BACKEND == "gemini":
                    content = _gemini_chat(prompt, GEMINI_TRIAGE_MODEL)
                else:
                    content = _openrouter_chat(prompt, TRIAGE_MODEL, max_tokens=1200)
                obj   = _extract_json_obj(content) or {}
                local = {}
                for e in obj.get("scores", []):
                    try:
                        local[int(e["id"])] = max(0.0, min(10.0, float(e["s"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                return bi, local
            except Exception as exc:  # noqa: BLE001
                print(f"    [triage batch {bi} skipped] {str(exc).splitlines()[0][:100]}")
                return bi, {}

        with ThreadPoolExecutor(max_workers=_judge_workers()) as pool:
            results = list(pool.map(run_batch, list(enumerate(batches))))

        for bi, local in results:
            for local_idx, article in enumerate(batches[bi]):
                score = local.get(local_idx, 0.0)
                orig_index = to_score[bi * TRIAGE_BATCH_SIZE + local_idx][0]
                scores[orig_index] = score
                _save_triage(company, sector_name, article, score)

    if scores and to_score:
        print(f"  Triage cache: {len(scores) - len(to_score)} reused, {len(to_score)} scored fresh")

    scored = []
    for i, article in enumerate(articles):
        a = dict(article)
        a["_triage_score"]     = scores.get(i, 0.0)
        a["matched_criterion"] = all_criteria     # judge anchor (was single CE best-match)
        a["risk_category"]     = "Business Risk"   # placeholder; the judge overwrites this
        scored.append(a)

    # Deterministic order: score desc, then headline -- so equal-score ties never reshuffle the
    # top-N cutoff between runs (another run-to-run variance source, now removed).
    scored.sort(key=lambda x: (-x["_triage_score"], x.get("headline", "")))
    return scored


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


_TRAFILATURA = None


def _extract_trafilatura(html, url):
    """Extract article body with trafilatura (higher hit-rate than newspaper3k on modern
    news layouts/paywalls). Returns text or None; lazily imported, disabled if unavailable."""
    global _TRAFILATURA
    if _TRAFILATURA is None:
        try:
            import trafilatura
            _TRAFILATURA = trafilatura
        except Exception:  # noqa: BLE001
            _TRAFILATURA = False
    if not _TRAFILATURA:
        return None
    try:
        return _TRAFILATURA.extract(html, url=url, include_comments=False,
                                    include_tables=False, favor_recall=True)
    except Exception:  # noqa: BLE001
        return None


def scrape_full_text(url, fallback_summary):
    """
    Downloads raw HTML via requests (8-second hard timeout) then extracts the article body.
    Returns (text, resolved_url).

    Extraction order: trafilatura first (best hit-rate on modern layouts), then newspaper3k,
    then the Finnhub/GDELT summary fallback. resolved_url follows the provider's redirect
    (e.g. Finnhub's `api/news?id=...`) to the real publisher article, so the stored/displayed
    source link is clickable. Falls back to the original url when the request fails.
    """
    final_url = url
    try:
        response = requests.get(url, timeout=8, headers=_SCRAPE_HEADERS)
        final_url = response.url or url          # resolved publisher URL after redirects
        response.raise_for_status()
        html = response.text

        text = _extract_trafilatura(html, final_url)
        if text and len(text.split()) > 100:
            return text, final_url

        article = Article(final_url)
        article.set_html(html)
        article.parse()
        if len(article.text.split()) > 100:
            return article.text, final_url
    except Exception:
        pass
    return fallback_summary, final_url


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

_JUDGE_RUBRIC = (
    "You are a senior S&P credit analyst assessing BONDHOLDER risk, NOT equity upside. "
    "Decide whether this NEWS ARTICLE describes a MATERIAL credit-rating event for the "
    "TARGET COMPANY, judged against the S&P sector criterion provided.\n"
    "\n"
    "MATERIALITY IS DECIDED BY THE S&P CRITERIA. An event is material ONLY if it plausibly "
    "moves one of the S&P sector rating factors in the MATCHED S&P CRITERION above -- e.g. "
    "leverage, cash flow, liquidity, refinancing/debt maturities, margins, competitive/market "
    "position, or capex intensity. Name that exact factor in 'sp_factor'. If the news does not "
    "map to a listed S&P rating factor, it is NOT material.\n"
    "\n"
    "STRICT MATERIALITY RULES -- set material=false if ANY of these apply:\n"
    "  - The news maps to NO S&P rating factor from the criterion above\n"
    "  - It is a lawsuit/litigation, activist-investor push, governance/board or management "
    "dispute, executive change, or stock-price/analyst commentary -- UNLESS the article states "
    "a concrete, quantified hit to leverage, liquidity, or cash flow (a fine/settlement large "
    "enough to move the balance sheet counts; reputational/legal risk alone does not)\n"
    "  - Company is only mentioned in passing or alongside many other companies\n"
    "  - Article is generic market/macro commentary with no company-specific figures\n"
    "  - Industry-wide policy change (tariffs, regulations) with no quantified impact "
    "on this company specifically\n"
    "  - New product launch or investment announcement with no concrete financial "
    "figures (revenue, margins, debt impact) -- speculative future benefit is NOT material\n"
    "  - Analyst stock rating or price-target change (equity signal, not credit signal)\n"
    "  - Results merely MISSED or BEAT ANALYST ESTIMATES: the gap vs expectations is an "
    "equity signal, not a credit event. Judge only the REPORTED figures themselves -- "
    "material only if they show real deterioration or improvement in cash generation, "
    "liquidity, or leverage\n"
    "\n"
    "BONDHOLDER DIRECTION RULES -- judge from the creditor's view, not the shareholder's:\n"
    "  NEGATIVE: debt issuance, large capex commitments, acquisitions, production declines, "
    "margin compression, market share loss, leverage increase, liquidity pressure\n"
    "  POSITIVE: debt repayment, cost cuts with confirmed savings, capacity utilisation "
    "recovery with concrete figures, refinancing at lower rates, asset disposal reducing "
    "debt. ALSO POSITIVE: demonstrated operating improvement backed by REPORTED figures -- "
    "sustained revenue/bookings growth, positive and growing free cash flow, margin "
    "expansion, guidance raised on delivered results. Improving cash generation IS "
    "credit-positive; do not dismiss it as equity hype when the figures are reported fact\n"
    "  NEUTRAL: product launches (future benefit unproven), supplier partnerships, "
    "workforce expansions (capex outflow), regulatory relief affecting the whole industry\n"
    "\n"
    "BALANCE-SHEET CONTEXT: when a COMPANY DEBT CONTEXT line is provided, weigh Financial "
    "Risk direction against the actual debt load. For a company with little debt or net "
    "cash, a profitability wobble is rarely credit-material; for a leveraged company the "
    "same wobble can be severe. Never claim debt-service pressure without evidence of "
    "real strain on cash, liquidity, or leverage. Use this context ONLY to judge -- do "
    "NOT restate its figures in event_summary or rationale.\n"
    "\n"
    "Set 'confidence' using this rubric -- do NOT default to a round number:\n"
    "  0.90-1.00: Company is primary subject; specific dated event with concrete financial "
    "figures; credit direction unambiguous from bondholder view; criterion match is direct\n"
    "  0.75-0.89: Company directly involved; direction clear but article lacks hard figures; "
    "or criterion match is strong but not exact\n"
    "  0.60-0.74: Indirect or industry-wide event; company-specific credit impact requires "
    "significant inference; or article is opinion/forecast not reported fact\n"
    "  Below 0.60: set material=false instead\n"
    "\n"
    "'event_summary' must be ONE concise sentence (max ~30 words) stating what actually "
    "happened, readable without the original article.\n"
    "'rationale' must be ONE concise sentence (max ~35 words): why it is credit-material "
    "for a bondholder. Do NOT restate the company's overall revenue or debt totals in it.\n"
    "'key_figures': the hard numbers stated in the article that bear on credit (dollar "
    "amounts, percentages, ratios). Each is a short {\"metric\", \"value\"} pair using ONLY "
    "numbers present in the text -- never invent or round. Put NO quotation marks inside "
    "metric or value. Use an empty list [] if the article states none.\n"
)

# Per-article verdict schema (shared by the single and batched judge prompts).
_JUDGE_VERDICT_SCHEMA = (
    '{"material": true/false, "risk_category": "Business Risk|Financial Risk|Neither", '
    '"sp_factor": "<short factor label>", "direction": "positive|negative|neutral", '
    '"confidence": 0.0-1.0, '
    '"key_figures": [{"metric": "<short label>", "value": "<number as written>"}], '
    '"event_summary": "<one concise sentence>", '
    '"rationale": "<one concise sentence>"}'
)

_JUDGE_INSTRUCTIONS = (
    _JUDGE_RUBRIC
    + "\nRespond with ONLY a compact single-line JSON object -- no preamble, no markdown "
      "fences, no newlines inside the JSON:\n"
    + _JUDGE_VERDICT_SCHEMA
)


def verify_figures(key_figures, article_text):
    """
    Anti-hallucination guard: keep only figures whose numeric value provably appears
    in the source article text (comma-insensitive), and drop the rest. Deterministic
    (no model involved) -- a number the judge invented cannot survive this check, so
    every figure that reaches the report is literally present in its source article.
    """
    if not key_figures or not article_text:
        return []
    text_norm = article_text.replace(",", "")
    verified  = []
    for f in key_figures:
        if not isinstance(f, dict):
            continue
        value = str(f.get("value", "")).strip()
        nums  = re.findall(r"\d[\d,]*(?:\.\d+)?", value)
        if not nums:
            continue
        if all(n.replace(",", "") in text_norm for n in nums):
            verified.append({"metric": str(f.get("metric", "")).strip(), "value": value})
    return verified


def build_judge_prompt(company, sector_name, matched_criterion, headline, article_text,
                       debt_context=""):
    debt = f"COMPANY DEBT CONTEXT: {debt_context}\n" if debt_context else ""
    return (
        f"{_JUDGE_INSTRUCTIONS}\n\n"
        f"TARGET COMPANY: {company}\n"
        f"S&P SECTOR: {sector_name}\n"
        f"{debt}"
        f"MATCHED S&P CRITERION:\n{matched_criterion}\n\n"
        f"NEWS HEADLINE: {headline}\n"
        f"FULL ARTICLE:\n{article_text[:MAX_ARTICLE_CHARS]}"
    )


def build_batch_judge_prompt(company, sector_name, debt_context, batch):
    """Judge MANY articles in a single call -- request-efficient for the rate-limited Gemini
    backend (one call for GEMINI_JUDGE_BATCH articles instead of one each). Same rubric as the
    single-article judge; returns a JSON array of verdicts keyed by article id."""
    debt = f"COMPANY DEBT CONTEXT: {debt_context}\n" if debt_context else ""
    blocks = []
    for i, a in enumerate(batch):
        blocks.append(
            f"--- ARTICLE id={i} ---\n"
            f"MATCHED S&P CRITERION: {a.get('matched_criterion', '')}\n"
            f"HEADLINE: {a.get('headline', '')}\n"
            f"ARTICLE:\n{(a.get('full_text', '') or '')[:MAX_ARTICLE_CHARS]}"
        )
    return (
        f"{_JUDGE_RUBRIC}\n"
        f"TARGET COMPANY: {company}\n"
        f"S&P SECTOR: {sector_name}\n"
        f"{debt}\n"
        f"Judge the {len(batch)} articles below INDEPENDENTLY -- apply every rule to EACH, and do "
        f"not let one article's verdict influence another.\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with ONLY a compact single-line JSON object, exactly one entry per id "
          "above, no preamble and no markdown fences:\n"
        + '{"verdicts":[{"id":0, ' + _JUDGE_VERDICT_SCHEMA[1:-1] + '}, {"id":1, ...}]}'
    )


def _extract_json_obj(text):
    """Pull the first JSON object out of a model's text output. Tolerates code fences,
    text before the object, and trailing commentary AFTER it -- raw_decode() parses the
    first balanced object and ignores whatever follows, so a stray sentence appended by
    the model no longer drops the whole article (was: greedy regex -> 'Extra data')."""
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


def _parse_judge_json(raw):
    """Parse the Claude CLI --output-format json envelope, then extract the verdict."""
    outer  = json.loads(raw)
    result = outer.get("result", raw) if isinstance(outer, dict) else raw
    if isinstance(result, dict):
        return result
    return _extract_json_obj(result)


def _judge_via_claude_cli(prompt):
    """One judgment via the local Claude Code CLI. Raises on failure."""
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", JUDGE_MODEL,
           "--output-format", "json", "--allowedTools", ""]
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        stdin=subprocess.DEVNULL, timeout=JUDGE_TIMEOUT,
    )
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "non-zero exit").strip()[:200])
    return _parse_judge_json(res.stdout.strip())


def _find_gemini_exe():
    """Locate the Gemini CLI. npm installs it as gemini.cmd (a batch shim) on Windows -- there
    is no .exe -- so prefer the shim under %APPDATA%\\npm, then fall back to PATH."""
    import shutil
    appdata_npm = os.path.join(os.environ.get("APPDATA", ""), "npm")
    for cand in ("gemini.cmd", "gemini.exe"):
        p = os.path.join(appdata_npm, cand)
        if os.path.exists(p):
            return p
    for cand in ("gemini.cmd", "gemini.exe", "gemini"):
        p = shutil.which(cand)
        if p:
            return p
    raise RuntimeError("gemini CLI not found on PATH (install: npm i -g @google/gemini-cli)")


def _gemini_chat(prompt, model, closing="Respond now with the JSON object only."):
    """LLM call via the local Gemini CLI subprocess. Returns the raw stdout text.

    `closing` is the short `-p` instruction (JSON callers keep the default; a prose
    caller passes a plain-text instruction so the model doesn't wrap the reply in JSON).

    Inference runs on the user's OAuth Google login (cached in ~/.gemini) through the GCP
    project named in GOOGLE_CLOUD_PROJECT. The Gemini quota is RATE-limited (requests/min and
    /day), NOT billed per token, so the scarce resource is the NUMBER of calls -- the pipeline
    caps concurrency (GEMINI_MAX_WORKERS) and backs off on rate-limit errors rather than
    hammering the endpoint. See memory: gemini-cli-zero-cost-backend.

    Details that matter:
      - Prompt goes on STDIN, not argv (Windows caps the command line ~32k chars).
      - The CLI has no JSON mode; -p carries only a short closing instruction, and the caller's
        tolerant parser (_extract_json_obj) pulls the JSON object out of the reply.
      - env GEMINI_CLI_TRUST_WORKSPACE=true passes the headless trusted-workspace gate.
      - On timeout, TREE-kill (taskkill /T): the npm shim spawns a node child that survives a
        normal kill and holds the pipes open, hanging the run forever.
      - Rate-limit / quota (429, RESOURCE_EXHAUSTED) is retried with exponential backoff; auth
        failure fails fast with a clear 're-login' message (retrying can't fix a dead token).
    """
    exe = _find_gemini_exe()
    env = {**os.environ, "GEMINI_CLI_TRUST_WORKSPACE": "true"}
    cmd = [exe, "-m", model, "-p", closing]

    def _run_once():
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", env=env, cwd=str(HERE))
        try:
            out, err = proc.communicate(input=prompt, timeout=GEMINI_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:                                   # tree-kill the surviving node child
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            except OSError:
                proc.kill()
            proc.wait()
            raise RuntimeError(f"gemini CLI timed out after {GEMINI_TIMEOUT}s (tree-killed)")
        return proc.returncode, out or "", err or ""

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        rc, out, err = _run_once()
        if rc == 0:
            return out
        blob = (err or out or "").lower()
        if any(w in blob for w in ("login", "oauth", "authenticat", "ineligible", "credential")):
            raise RuntimeError("gemini CLI auth failed -- run `gemini` once to re-login and confirm "
                               f"GOOGLE_CLOUD_PROJECT is set. Detail: {(err or out)[:300]}")
        rate_limited = any(w in blob for w in ("429", "rate limit", "rate-limit", "resource_exhausted",
                                               "resource exhausted", "quota", "too many requests"))
        if rate_limited and attempt < GEMINI_MAX_RETRIES:
            # RPM-limited: wait seconds, not milliseconds. Exponential: 8s, 16s, 32s...
            time.sleep(GEMINI_BACKOFF_BASE * (2 ** attempt))
            continue
        raise RuntimeError(f"gemini CLI failed (exit {rc}): {(err or out)[:400]}")
    raise RuntimeError("gemini CLI: exhausted retries")  # unreachable, keeps intent explicit


def _judge_via_gemini_cli(prompt):
    """One judgment via the local Gemini CLI ($0 OAuth-quota backend). Raises on failure."""
    return _extract_json_obj(_gemini_chat(prompt, JUDGE_MODEL))


def _judge_call_fn():
    """Return the judge-call function for the active JUDGE_BACKEND. Each takes a prompt and
    returns a parsed verdict dict (or None if the model output had no JSON object)."""
    if JUDGE_BACKEND == "gemini":
        return _judge_via_gemini_cli
    if JUDGE_BACKEND == "openrouter":
        return _judge_via_openrouter
    return _judge_via_claude_cli


def _judge_workers():
    """Effective judge/triage thread count. The gemini CLI spawns a subprocess per call and is
    quota/rate-limited, so cap its concurrency; HTTP/CLI-Claude paths keep JUDGE_WORKERS."""
    if JUDGE_BACKEND == "gemini":
        return min(GEMINI_MAX_WORKERS, JUDGE_WORKERS)
    return JUDGE_WORKERS


def _headline_key(article):
    """Normalized headline key -- same rule merge_news dedups on, so the recall loop can tell a
    genuinely new article from one already in the judged feed."""
    return re.sub(r"\W+", "", (article.get("headline", "") or "")).lower()[:90]


def build_refute_prompt(company, sector_name, signal):
    """Skeptic prompt: challenge whether a flagged signal is really CREDIT-material. Biased to
    KEEP (only refute when clearly not credit-material) so the precision pass never quietly
    undoes the recall loop -- see memory: self-critique that defaulted to 'reject' biased scores
    down and was dropped."""
    return (
        "You are a skeptical S&P credit analyst double-checking a colleague's work. They flagged "
        "the news below as a MATERIAL credit-rating event. Challenge it: is it genuinely material "
        "to the CREDIT rating (leverage, liquidity, cash flow, refinancing, covenants, a major "
        "debt/capex/strategic shift), or is it equity-only noise, routine, or too vague to size?\n"
        f"COMPANY: {company}   |   S&P SECTOR: {sector_name}\n"
        f"CLAIMED EVENT : {signal.get('event_summary', '')}\n"
        f"DIRECTION     : {signal.get('direction', '')}   CATEGORY: {signal.get('risk_category', '')}\n"
        f"HEADLINE      : {signal.get('headline', '')}\n"
        f"ARTICLE:\n{(signal.get('full_text', '') or '')[:MAX_ARTICLE_CHARS]}\n\n"
        "Only answer material=false if you are CONFIDENT this is not a credit-material event; if "
        "there is a plausible credit angle, keep it (material=true).\n"
        'Respond with ONLY compact JSON: {"material": true, "reason": "<one line>"}'
    )


def build_refute_batch_prompt(company, sector_name, batch):
    """Challenge MANY signals in one call -- request-efficient for the rate-limited Gemini
    backend. Same conservative stance as the single refute prompt (keep unless clearly not
    credit-material); returns a JSON array of keep/drop verdicts keyed by id."""
    blocks = []
    for i, s in enumerate(batch):
        blocks.append(
            f"--- SIGNAL id={i} ---\n"
            f"CLAIMED EVENT: {s.get('event_summary', '')}\n"
            f"DIRECTION: {s.get('direction', '')}   CATEGORY: {s.get('risk_category', '')}\n"
            f"HEADLINE: {s.get('headline', '')}"
        )
    return (
        "You are a skeptical S&P credit analyst double-checking a colleague's work. For EACH "
        "signal below, decide whether it is genuinely MATERIAL to the CREDIT rating (leverage, "
        "liquidity, cash flow, refinancing, covenants, a major debt/capex/strategic shift), or "
        "whether it is equity-only noise, routine, or too vague to size.\n"
        f"COMPANY: {company}   |   S&P SECTOR: {sector_name}\n\n"
        + "\n\n".join(blocks)
        + "\n\nOnly answer material=false when you are CONFIDENT a signal is not credit-material; "
          "if there is a plausible credit angle, keep it (material=true).\n"
        'Respond with ONLY compact JSON, one entry per id above:\n'
        '{"verdicts":[{"id":0,"material":true},{"id":1,"material":false}]}'
    )


def adversarial_verify(signals, company, sector_name):
    """Precision guard for the recall loop: challenge each material signal to refute its
    credit-materiality. Returns (kept, demoted). Conservative -- a signal is demoted ONLY when
    the skeptic is confident it is not material; LLM/parse failures (or a missing id in the
    batched reply) keep the signal (never drop a real one over a hiccup). Demoted signals become
    near-misses, not discarded. On Gemini, signals are challenged GEMINI_VERIFY_BATCH per call."""
    if not signals:
        return signals, []

    if JUDGE_BACKEND == "gemini" and GEMINI_VERIFY_BATCH > 1:
        chunks = [(i, signals[i:i + GEMINI_VERIFY_BATCH])
                  for i in range(0, len(signals), GEMINI_VERIFY_BATCH)]

        def verify_chunk(chunk):
            offset, batch = chunk
            try:
                # Adversarial verify is a light yes/no -- run it on Flash, not the 2.5-Pro judge.
                obj = _extract_json_obj(_gemini_chat(
                    build_refute_batch_prompt(company, sector_name, batch),
                    GEMINI_TRIAGE_MODEL)) or {}
                keep = {}
                for v in obj.get("verdicts", []):
                    if isinstance(v, dict) and "id" in v:
                        try:
                            keep[offset + int(v["id"])] = bool(v.get("material", True))
                        except (TypeError, ValueError):
                            continue
                return keep
            except Exception:  # noqa: BLE001
                return {}                        # keep every signal in this chunk on failure

        keep_flags = {}
        with ThreadPoolExecutor(max_workers=_judge_workers()) as pool:
            for kmap in pool.map(verify_chunk, chunks):
                keep_flags.update(kmap)
        kept    = [s for i, s in enumerate(signals) if keep_flags.get(i, True)]
        demoted = [s for i, s in enumerate(signals) if not keep_flags.get(i, True)]
        return kept, demoted

    # Per-signal path (non-Gemini backends).
    call = _judge_call_fn()

    def challenge(sig):
        try:
            v = call(build_refute_prompt(company, sector_name, sig))
        except Exception:  # noqa: BLE001
            return sig, True                     # on failure, keep (don't drop a real signal)
        if not isinstance(v, dict):
            return sig, True
        return sig, bool(v.get("material", True))

    with ThreadPoolExecutor(max_workers=_judge_workers()) as pool:
        results = list(pool.map(challenge, signals))
    kept    = [s for s, ok in results if ok]
    demoted = [s for s, ok in results if not ok]
    return kept, demoted


def _openrouter_chat(prompt, model, max_tokens=500):
    """Raw OpenRouter chat completion. Returns the message content string. Raises on failure.
    Retries transient failures (network errors, 429, 5xx) up to 3 attempts with exponential
    backoff so one slow/rate-limited call doesn't silently drop a signal."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Joywin-International-Limited/credit-risk-news-scraper",
                    "X-Title": "Joywin Credit Risk Pipeline",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=JUDGE_TIMEOUT,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                raise RuntimeError(f"no choices in response: {str(data)[:200]}")
            return choices[0].get("message", {}).get("content", "")
        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            transient = isinstance(exc, requests.RequestException) or \
                        any(c in str(exc) for c in (" 429", " 500", " 502", " 503", " 504"))
            if attempt < 2 and transient:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_exc  # unreachable, but keeps intent explicit


def _judge_via_openrouter(prompt):
    """One judgment via the OpenRouter HTTP API. Raises on failure."""
    return _extract_json_obj(_openrouter_chat(prompt, JUDGE_MODEL, max_tokens=500))


def judge_article(company, sector_name, matched_criterion, headline, article_text,
                  debt_context=""):
    """
    Run one judgment on the full article text via the selected backend. Returns the
    verdict dict, or None on failure (skipped rather than crashing the run). One retry.
    """
    prompt = build_judge_prompt(company, sector_name, matched_criterion, headline,
                                article_text, debt_context)
    call   = _judge_call_fn()

    for attempt in (1, 2):
        try:
            verdict = call(prompt)
            if verdict is None:
                raise ValueError("no JSON object in judge output")
            return verdict
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"    [judge skipped] {str(exc).splitlines()[0][:120]}")
                return None
    return None


def _annotate_verdict(article, verdict):
    """Attach a parsed judge verdict to its article dict (shared by the per-article and batched
    judge paths). key_figures are verified against the source text (anti-hallucination)."""
    try:
        confidence = float(verdict.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    out = dict(article)
    out.update({
        "material":      bool(verdict.get("material")),
        "risk_category": verdict.get("risk_category", "Neither"),
        "sp_factor":     verdict.get("sp_factor", ""),
        "direction":     verdict.get("direction", "neutral"),
        "confidence":    round(confidence, 3),
        "key_figures":   verify_figures(verdict.get("key_figures", []),
                                        article.get("full_text", "")),
        "event_summary": verdict.get("event_summary", ""),
        "rationale":     verdict.get("rationale", ""),
    })
    return out


def _split_materials(judged):
    """Partition judged articles into (materials, near_misses). Material = judged material AND
    confidence >= MIN_JUDGE_CONFIDENCE; the rest are near-misses, sorted by confidence."""
    materials, near = [], []
    for r in judged:
        if r["material"] and r["confidence"] >= MIN_JUDGE_CONFIDENCE:
            materials.append(r)
        else:
            near.append(r)
    near.sort(key=lambda r: r["confidence"], reverse=True)
    return materials, near


def _verdict_key(company, article):
    """Stable cache key for one article's judge verdict. Includes everything the verdict depends
    on, so a change in company/article/criterion/model/prompt invalidates it automatically."""
    import hashlib
    raw = "|".join([company or "", article.get("url", "") or "",
                    article.get("headline", "") or "", article.get("matched_criterion", "") or "",
                    JUDGE_MODEL or "", _JUDGE_PROMPT_VERSION])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_cached_verdict(company, article):
    """Return a previously cached raw verdict dict for this article, or None."""
    if not VERDICT_CACHE:
        return None
    p = HERE / ".cache" / "verdicts" / f"{_verdict_key(company, article)}.json"
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_verdict(company, article, verdict):
    """Persist a raw verdict dict so future runs of the same issuer reuse it (reproducible + free)."""
    if not VERDICT_CACHE or not verdict:
        return
    d = HERE / ".cache" / "verdicts"
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_verdict_key(company, article)}.json").write_text(
            json.dumps(verdict), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _judge_batched_raw(articles, company, sector_name, debt_context):
    """Judge GEMINI_JUDGE_BATCH articles per Gemini call. Returns [(article, verdict|None)].
    A batch whose JSON can't be parsed is retried once, then its articles come back None."""
    batches = [articles[i:i + GEMINI_JUDGE_BATCH]
               for i in range(0, len(articles), GEMINI_JUDGE_BATCH)]

    def judge_one_batch(batch):
        prompt = build_batch_judge_prompt(company, sector_name, debt_context, batch)
        for attempt in (1, 2):
            try:
                obj = _extract_json_obj(_gemini_chat(prompt, JUDGE_MODEL)) or {}
                verdicts = obj.get("verdicts")
                if not isinstance(verdicts, list) or not verdicts:
                    raise ValueError("no verdicts array in judge output")
                by_id = {}
                for v in verdicts:
                    if isinstance(v, dict) and "id" in v:
                        try:
                            by_id[int(v["id"])] = v
                        except (TypeError, ValueError):
                            continue
                return [(a, by_id.get(i)) for i, a in enumerate(batch)]
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"    [judge batch skipped] {str(exc).splitlines()[0][:100]}")
                    return [(a, None) for a in batch]
        return [(a, None) for a in batch]

    with ThreadPoolExecutor(max_workers=_judge_workers()) as pool:
        return [pair for res in pool.map(judge_one_batch, batches) for pair in res]


def _judge_per_article_raw(articles, company, sector_name, debt_context):
    """Judge one article per call (non-Gemini backends). Returns [(article, verdict|None)]."""
    def work(article):
        v = judge_article(company, sector_name, article["matched_criterion"],
                          article.get("headline", ""), article.get("full_text", ""), debt_context)
        return (article, v)

    with ThreadPoolExecutor(max_workers=_judge_workers()) as pool:
        return list(pool.map(work, articles))


def run_judge_articles(articles, company, sector_name, debt_context=""):
    """
    Judge all filtered articles. Returns (materials, near_misses):
      - materials  : articles the judge marked material AND confident (>= MIN_JUDGE_CONFIDENCE)
      - near_misses: everything else it judged (immaterial or below the bar), sorted by confidence.
    Verdicts are cached per article (VERDICT_CACHE): a re-run of the same issuer reuses prior
    judgments verbatim -- identical output and zero repeat LLM calls. Only uncached articles are
    sent to the model, batched (GEMINI_JUDGE_BATCH) on Gemini, one-per-call elsewhere.
    """
    if not articles:
        return [], []

    judged, to_judge = [], []
    for a in articles:
        cached = _load_cached_verdict(company, a)
        if cached is not None:
            judged.append(_annotate_verdict(a, cached))
        else:
            to_judge.append(a)

    if to_judge:
        if JUDGE_BACKEND == "gemini" and GEMINI_JUDGE_BATCH > 1:
            pairs = _judge_batched_raw(to_judge, company, sector_name, debt_context)
        else:
            pairs = _judge_per_article_raw(to_judge, company, sector_name, debt_context)
        for a, v in pairs:
            if v:
                _save_verdict(company, a, v)
                judged.append(_annotate_verdict(a, v))

    reused = len(articles) - len(to_judge)
    if reused:
        print(f"  Verdict cache: {reused} reused, {len(to_judge)} judged fresh")
    return _split_materials(judged)


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


def find_contradictions(signals):
    """
    Flag pairs of signals dated the same day with opposite directions -- usually one
    event (e.g. an earnings release) read two ways by different outlets. These inflate
    apparent coverage while cancelling in the score; the analyst should reconcile them.
    Returns a list of (ref_a, ref_b) display numbers.
    """
    pairs = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            a, b = signals[i], signals[j]
            if (a.get("date") == b.get("date")
                    and {a.get("direction"), b.get("direction")} == {"positive", "negative"}):
                pairs.append((a.get("ref", i + 1), b.get("ref", j + 1)))
    return pairs


def cluster_events(signals, encoder,
                   sim_threshold=EVENT_CLUSTER_SIM, day_window=EVENT_CLUSTER_DAYS):
    """
    Group signals that describe the SAME underlying event, so it votes once rather than
    once per outlet/article. Two signals join the same event when their event_summaries
    are similar (cosine >= sim_threshold) AND they are dated within day_window days (or a
    date is missing). Union-find over those links -> connected components = events. This
    catches both reworded duplicates from many outlets and the same event split across
    Business and Financial Risk. Returns a list of clusters (each a list of signals).
    """
    n = len(signals)
    if n <= 1:
        return [[s] for s in signals]

    texts = [s.get("event_summary") or s.get("headline", "") for s in signals]
    vecs  = encoder.encode(texts, convert_to_tensor=True)

    dts = []
    for s in signals:
        try:
            dts.append(datetime.datetime.strptime(s.get("date", ""), "%Y-%m-%d"))
        except (ValueError, TypeError):
            dts.append(None)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if util.cos_sim(vecs[i], vecs[j]).item() < sim_threshold:
                continue
            close = (dts[i] is None or dts[j] is None
                     or abs((dts[i] - dts[j]).days) <= day_window)
            if close:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(signals[i])
    return list(groups.values())


def reconcile_events(clusters):
    """
    Collapse each event cluster to ONE reconciled signal:
      - direction  : sign of the confidence-weighted net of member directions -- a
                     positive+negative split on the SAME event cancels instead of being
                     double-counted (the RBLX / UPS one-event-two-reads problem).
      - confidence : |net signed confidence| / members. Unanimity is preserved (a 2x +0.8
                     cluster stays 0.8); internal disagreement shrinks it toward 0.
      - coverage   : number of members -> feeds a mild coverage weight in scoring.
      - key_figures: union of all members' verified figures (no evidence lost on collapse).
      - conflicted : True if members disagreed on direction (surfaced in the report).
    Representative display fields (event_summary, headline, url, date, sp_factor,
    risk_category) come from the highest-confidence member on the winning side. Every
    member is retained under "members" for the JSON audit trail.
    """
    sign_of = {"negative": -1, "positive": 1, "neutral": 0}
    events  = []
    for members in clusters:
        net = conf_sum = 0.0
        dirs = set()
        for s in members:
            sgn = sign_of.get(s.get("direction", "neutral"), 0)
            c   = float(s.get("confidence", 0) or 0)
            net += sgn * c
            conf_sum += c
            if sgn != 0:
                dirs.add(sgn)

        if net > 1e-9:
            direction = "positive"
        elif net < -1e-9:
            direction = "negative"
        else:
            direction = "neutral"

        m        = len(members)
        rep_conf = round(abs(net) / m, 3) if m else 0.0

        want = sign_of.get(direction, 0)
        pool = [s for s in members if sign_of.get(s.get("direction"), 0) == want] or members
        rep  = max(pool, key=lambda s: float(s.get("confidence", 0) or 0))

        merged_figs, seen = [], set()
        for s in members:
            for f in (s.get("key_figures") or []):
                k = f"{f.get('metric', '')}:{f.get('value', '')}".lower()
                if k not in seen:
                    seen.add(k)
                    merged_figs.append(f)

        ev = dict(rep)
        ev["direction"]   = direction
        ev["confidence"]  = rep_conf
        ev["coverage"]    = m
        ev["conflicted"]  = len(dirs) > 1
        ev["key_figures"] = merged_figs
        ev["members"]     = [
            {"date": s.get("date"), "direction": s.get("direction"),
             "confidence": s.get("confidence"), "risk_category": s.get("risk_category"),
             "event_summary": s.get("event_summary"), "headline": s.get("headline"),
             "url": s.get("url")}
            for s in members
        ]
        events.append(ev)
    return events


# ==========================================
# CONSOLE REPORT
# ==========================================

def _wrap_field(label, text, width=78, indent=14):
    """Format a labelled field with word-wrapping so long text is never truncated."""
    body = textwrap.fill(
        text or "", width=width,
        initial_indent="", subsequent_indent=" " * indent,
    )
    return f"  {label:<10}: {body}"


# Equity/market chatter vs business/credit-fundamental news. Deterministic keyword
# scoring -- transparent, tunable, and free (no LLM). Used only to organise the digest;
# it never touches the credit score.
_EQUITY_MARKERS = (
    "stock", "shares", "share price", "price target", "price prediction", "still a buy",
    "is a buy", "is it a buy", "better buy", "should you buy", "buy right now",
    "buy or sell", "time to sell", "sell-off", "selloff", "overweight", "underweight",
    "outperform", "buy rating", "hold rating", "top pick", "top-ranked", "momentum",
    "zacks", "valuation", "premium", "stock split", "fomo", "bull case", "bear case",
    "upside", "downside", "fair value", "forecast", "prediction:", "which stock", "vs.",
    "rally", "rallies", "plunge", "plunges", "slump", "slides", "sinks", "surges", "soar",
    "rebound", "rebounds", "falling", "drops", "swoon", "tumble", "dow jones", "s&p 500",
    "nasdaq", "futures", "jobs report", "stocks that", "most-searched", "top-searched",
    "crash", "52-week", "market cap gain",
)
_BUSINESS_MARKERS = (
    "supply agreement", "supply deal", "chip supply", "take-or-pay", "contract", "signs",
    "signed", "agreement with", "partnership", "acquire", "acquisition", "merger", "capex",
    "capital expenditure", "billion bet", "million bet", "spending", "to spend", "invests",
    "investment", "facility", "fab", "factory", "plant", "production", "manufacturing",
    "debt", "leverage", "refinanc", "bond", "credit rating", "downgrade", "upgrade",
    "outlook", "default", "bankruptcy", "dividend", "lawsuit", "settlement", "ftc",
    "antitrust", "regulator", "probe", "recall", "revenue", "margin", "guidance",
    "earnings", "layoff", "job cuts", "restructur", "emissions", "milestone",
)


def classify_news_topic(text):
    """Return 'equity' or 'business'. 'equity' = stock-price / valuation / analyst-rating /
    market-roundup chatter (secondary); 'business' = credit-relevant fundamentals
    (contracts, capex, M&A, debt, legal, results, ops). Mixed/tie headlines lean 'equity'
    so the primary digest stays clean -- equity items are still kept, just bucketed."""
    t  = (text or "").lower()
    eq = sum(1 for m in _EQUITY_MARKERS if m in t)
    bz = sum(1 for m in _BUSINESS_MARKERS if m in t)
    return "equity" if (eq > 0 and eq >= bz) else "business"


# Within the equity bucket, grade a story by SUBSTANCE. We want the BUSINESS events that
# actually move the stock for a real reason -- launches, deals, capex, strategic shifts,
# results -- and we deliberately DEMOTE analyst price-target/rating calls and valuation
# opinions ("21% undervalued"), which are commentary, not events. Deterministic, no LLM.
_EQ_CATALYST = (          # a real business event driving the move (weighted UP hardest)
    "launch", "launches", "launched", "unveil", "unveils", "announces", "announced",
    "to sell", "new business", "enters", "entering", "expands into", "rolls out",
    "rolling out", "builds", "building", "plans to", "deal", "agreement", "contract",
    "partnership", "partners with", "acquire", "acquires", "acquisition", "merger",
    "invest", "investment", "capex", "capital expenditure", "to spend", "spend",
    "spends", "spending", "billion on",
    "chip", "data center", "cloud", "factory", "plant", "facility", "fab", "product",
    "wins", "secures", "signs", "signed", "supply", "hires", "appoints", "names",
    "lawsuit", "settlement", "antitrust", "regulator", "probe", "recall", "ban",
    "results", "earnings", "revenue", "guidance", "layoff", "job cuts", "restructur",
)
_EQ_GROWTH = (            # demonstrated demand / share / scale (weighted UP)
    "growth", "expansion", "expand", "expanding", "new market", "capacity", "backlog",
    "demand", "tam", "addressable market", "market share", "record revenue",
    "record sales", "ramp", "roadmap", "orders", "adoption", "scaling", "penetration",
)
_EQ_ANALYST = (           # analyst prediction / rating -- NOT a business event (DEMOTED)
    "price target", "raises target", "cuts target", "target to", "pt to", "sees upside",
    "upside for", "% upside", "buy rating", "hold rating", "sell rating", "outperform",
    "overweight", "underweight", "reiterat", "initiates", "initiated", "analyst says",
    "analysts say", "analyst", "price prediction", "will hit", "stock forecast",
)
_EQ_VALUATION = (         # valuation opinion -- commentary, not an event (DEMOTED)
    "undervalued", "overvalued", "fair value", "intrinsic", "p/e", "pe ratio", "multiple",
    "trading at", "cheap", "expensive", "discount", "worth", "dcf", "ev/ebitda",
    "should you buy", "is it a buy", "better buy", "still a buy", "time to sell",
)
_EQ_HYPE = (              # pure price action / roundups (DEMOTED)
    "all-time high", "52-week", "52 week", "stock split", "most-searched", "top-searched",
    "momentum", "futures", "meme", "retail traders", "surge", "surges", "plunge",
    "plunges", "rally", "rallies", "soar", "soars", "sink", "sinks", "tumble", "jumps",
    "% since", "skyrocket", "dow jones", "nasdaq", "crash", "fomo", "why is",
)


def classify_equity_subtopic(text):
    """Grade an equity story by substance. Returns (label, score): label in
    {Catalyst, Growth, Analyst, Valuation, Price}. Business catalysts and demonstrated
    growth are weighted UP; analyst price-target/rating calls, valuation opinions, and
    pure price action are weighted DOWN -- so the digest surfaces news that moves the
    stock for a real business reason, not commentary."""
    t = (text or "").lower()
    c = sum(1 for m in _EQ_CATALYST if m in t)
    g = sum(1 for m in _EQ_GROWTH if m in t)
    a = sum(1 for m in _EQ_ANALYST if m in t)
    v = sum(1 for m in _EQ_VALUATION if m in t)
    h = sum(1 for m in _EQ_HYPE if m in t)
    score = 2 * c + 2 * g - 2 * a - 2 * v - h
    if c and c >= g:
        label = "Catalyst"
    elif g:
        label = "Growth"
    elif a:
        label = "Analyst"
    elif v:
        label = "Valuation"
    else:
        label = "Price"
    return label, score


def _issuer_lead_penalty(headline, name_tokens):
    """Penalty (>=0) for stories where the issuer is NOT the lead subject of the headline --
    e.g. 'CoreWeave Takes Another Hit From Meta' names Meta only at the end. If the issuer
    first appears in the back half of the headline (or not at all), it is probably a
    bystander mention, not the subject; such items are demoted out of the digest."""
    t = (headline or "").lower()
    earliest = None
    for tok in name_tokens:
        idx = t.find(tok)
        if idx >= 0:
            earliest = idx if earliest is None else min(earliest, idx)
    if earliest is None:            # issuer not in the headline at all -> heavy penalty
        return 5
    return 5 if (earliest / max(1, len(t))) > 0.5 else 0


def build_news_digest(articles):
    """Lightweight 'what's happening' list from the on-topic articles (no LLM cost).
    Every article the issuer was named in, newest first, whether or not it is
    credit-material -- so a run always shows company news even at 0 scored signals.
    Each item is tagged topic = business|equity (see classify_news_topic)."""
    digest = []
    for a in articles:
        try:
            d = datetime.datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            d = ""
        headline = a.get("headline", "")
        summary  = (a.get("summary", "") or "")
        digest.append({
            "date":     d,
            "headline": headline,
            "summary":  summary,
            "url":      a.get("url", ""),
            "topic":    classify_news_topic(f"{headline} {summary}"),
        })
    digest.sort(key=lambda x: x["date"], reverse=True)
    return digest


def _clean_text(s):
    """Drop zero-width / control chars that break non-UTF8 consoles; keep accents."""
    return "".join(c for c in (s or "").replace("\n", " ") if c.isprintable())


def _first_sentences(text, n=2, max_chars=320):
    """First n sentences of a summary, capped -- keeps the digest scannable instead of
    dumping a whole article body (some feeds put a wall of text in the summary field)."""
    t = _clean_text(text).strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = " ".join(parts[:n]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "..."
    return out


def cluster_top_news(items, encoder, k):
    """Group same-story items (many outlets, one event) by headline similarity and return
    the k biggest clusters -- coverage = importance. Each result is (representative, count);
    representative = most recent member (items are newest-first). No LLM."""
    if not items:
        return []
    vecs     = encoder.encode([it["headline"] for it in items], convert_to_tensor=True)
    assigned = [False] * len(items)
    clusters = []
    for i in range(len(items)):
        if assigned[i]:
            continue
        assigned[i] = True
        members = [i]
        for j in range(i + 1, len(items)):
            if not assigned[j] and util.cos_sim(vecs[i], vecs[j]).item() >= DIGEST_CLUSTER_SIM:
                assigned[j] = True
                members.append(j)
        clusters.append((items[min(members)], len(members)))  # rep = newest in cluster
    clusters.sort(key=lambda c: (c[1], c[0]["date"]), reverse=True)
    return clusters[:k]


def print_news_digest(digest, encoder, company_name="", ticker="", show_equity=True):
    """SECONDARY section: the equity/market stories that move the stock for a real BUSINESS
    reason -- launches, deals, capex, strategic shifts, results -- surfaced over analyst
    price-target/rating calls and valuation opinions, which are demoted. Bystander stories
    (issuer not the headline subject) are dropped. Credit-relevant news is the report above."""
    equity = [d for d in digest if d.get("topic") == "equity"]
    if not show_equity:
        if equity:
            print(f"\n  (equity/market news hidden -- SHOW_EQUITY_NEWS=False; {len(equity)} in JSON)")
        return
    if not equity:
        return

    # Issuer name tokens (suffixes stripped) + ticker -- to detect bystander headlines.
    toks = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", company_name or "") if t]
    name_tokens = [t for t in toks if len(t) >= 3 and t not in _COMPANY_NAME_SUFFIXES]
    if ticker:
        name_tokens.append(ticker.lower())

    # Cluster same-story items, then rank by BUSINESS SUBSTANCE (catalysts/growth up,
    # analyst/valuation/price down) minus a bystander penalty; coverage is only the tiebreak.
    clusters = cluster_top_news(equity, encoder, len(equity))
    ranked   = []
    for rep, cnt in clusters:
        label, score = classify_equity_subtopic(f"{rep.get('headline', '')} {rep.get('summary', '')}")
        score -= _issuer_lead_penalty(rep.get("headline", ""), name_tokens)
        ranked.append((rep, cnt, label, score))
    # Drop analyst-prediction / valuation-opinion / bystander items (net-negative substance).
    ranked = [r for r in ranked if r[3] > 0]
    ranked.sort(key=lambda x: (x[3], x[1]), reverse=True)
    top = ranked[:DIGEST_TOP_EQUITY]
    if not top:
        return

    print("\n" + "=" * 72)
    print("  SECONDARY -- BUSINESS / MARKET-MOVING NEWS  (not credit-scored)")
    print("  business catalysts (launches, deals, capex, results)  >  analyst / price calls")
    print("=" * 72)
    for rep, cnt, label, _score in top:
        outlets = f"{cnt} outlets" if cnt > 1 else "1 outlet"
        print(f"\n  [{label:<9}]  {rep['date']}   ({outlets})")
        for line in textwrap.wrap(_clean_text(rep["headline"]), width=74,
                                  initial_indent="     ", subsequent_indent="     "):
            print(line)
        summ = _first_sentences(rep.get("summary", ""), n=2)
        if summ:
            for line in textwrap.wrap(summ, width=74, initial_indent="       ",
                                      subsequent_indent="       "):
                print(line)
        if rep.get("url"):
            print(f"       {rep['url']}")   # raw + full so it copy-pastes intact
    print("=" * 72)


def developing_news_items(near, n=DEVELOPING_NEWS_N):
    """The n credit-relevant stories the judge read but that were not (yet) a rating event --
    highest judge-confidence first. These are the forward-looking business developments
    (capex plans, strategic shifts, deals in progress). Context only -- NOT scored."""
    return [x for x in near if x.get("event_summary")][:max(n, DEVELOPING_NEWS_N)]


def print_developing_news(near, has_material, n=DEVELOPING_NEWS_N):
    """Shown on EVERY run so a report always surfaces credit-relevant context, even at zero
    material signals. These are reviewed-but-not-yet-material stories; they do NOT enter the
    credit score (the score is material judge signals + the fundamentals signal only)."""
    cand = developing_news_items(near, n)
    if not cand:
        return
    print("\n" + "=" * 65)
    print("  DEVELOPING / REVIEWED -- credit-relevant, not yet a rating event")
    note = ("context for the score above" if has_material
            else f"no article cleared the {MIN_JUDGE_CONFIDENCE:.2f} materiality bar")
    print(f"  ({note}; not scored)")
    print("=" * 65)
    for i, s in enumerate(cand, 1):
        arrow = {"positive": "[+]", "negative": "[-]"}.get(s.get("direction"), "[=]")
        print(f"\n  [{i}]  {s.get('date', '')}  |  judge confidence {s.get('confidence', 0)}"
              f"  |  {arrow} {s.get('direction', 'neutral')}")
        print(_wrap_field("Event", s.get("event_summary", "")))
        if s.get("headline"):
            print(_wrap_field("Headline", s.get("headline", "")))
        if s.get("url"):
            print(f"  Source    : {s['url']}")
    print("=" * 65)


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
                cov     = sig.get("coverage", 1) or 1
                cov_str = f"  |  {cov} outlets" if cov > 1 else ""
                confl   = "  ! reconciled from mixed reads" if sig.get("conflicted") else ""
                print(f"\n  [{sig.get('ref', i)}]  {sig['date']}  |  Credit: {arrow}"
                      f"  |  Confidence: {sig['confidence']}{cov_str}{tone_str}{confl}")
                print(_wrap_field("Event", sig.get("event_summary", "")))
                print(_wrap_field("S&P factor", sig.get("sp_factor", "")))
                print(_wrap_field("Why", sig.get("rationale", "")))
                print(_wrap_field("Headline", sig.get("headline", "")))
            else:                   # legacy cosine mode
                print(f"\n  [{i}]  {sig['date']}  |  Score: {sig['similarity_score']}"
                      f"  |  Gap: {sig['confidence_gap']}")
                print(_wrap_field("Headline", sig.get("headline", "")))
                print(_wrap_field("Criterion", sig.get("matched_criterion", "")))
                print(_wrap_field("Extract", sig.get("extracted_chunk", "")))
            if sig.get("url"):
                # printed raw + full (never wrapped) so it copy-pastes into a browser intact
                print(f"  Source    : {sig['url']}")
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

    Final score is normalized by the theoretical maximum (all signals pointing same way),
    then shrunk by n/(n+SCORE_SHRINKAGE_K) so thin evidence cannot pin the ceiling.
    Both raw and shrunk scores are returned; the verdict is read off the shrunk score.
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

        # Coverage weight: a widely-covered event counts more, but only mildly (log-scaled
        # + capped) so it can't dominate. coverage defaults to 1 for legacy/unclustered.
        cov    = int(sig.get("coverage", 1) or 1)
        cov_wt = (min(COVERAGE_WEIGHT_MAX, 1.0 + COVERAGE_WEIGHT_K * math.log(cov))
                  if cov > 1 else 1.0)

        w = conf * r_wt * rec_wt * cov_wt
        raw_total += d_sign * w
        max_total += w

        if cat == "Financial Risk":
            fr_raw += d_sign * w;  fr_max += w
        else:
            br_raw += d_sign * w;  br_max += w

        if d_sign < 0:   neg_count += 1
        elif d_sign > 0: pos_count += 1
        else:            neu_count += 1

    raw_score = round(raw_total / max_total, 3) if max_total > 0 else 0.0
    br_score  = round(br_raw / br_max, 3)       if br_max    > 0 else None
    fr_score  = round(fr_raw / fr_max, 3)       if fr_max    > 0 else None

    # Evidence shrinkage: raw x n/(n+K). A unanimous 2-signal run no longer pins the
    # ±1.0 ceiling; the score now reflects both direction AND how much evidence backs it.
    total_signals = neg_count + pos_count + neu_count
    evidence = (total_signals / (total_signals + SCORE_SHRINKAGE_K)) if total_signals else 0.0
    score    = round(raw_score * evidence, 3)

    verdict = SCORE_BANDS[-1][2]
    description = SCORE_BANDS[-1][3]
    for lo, hi, label, desc in SCORE_BANDS:
        if lo <= score <= hi:
            verdict = label
            description = desc
            break

    if total_signals >= 8:
        conviction = "HIGH"
    elif total_signals >= 4:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    return {
        "score":                score,
        "score_raw":            raw_score,
        "evidence_factor":      round(evidence, 3),
        "verdict":              verdict,
        "description":          description,
        "conviction":           conviction,
        "business_risk_score":  br_score,
        "financial_risk_score": fr_score,
        "negative_signals":     neg_count,
        "positive_signals":     pos_count,
        "neutral_signals":      neu_count,
    }


def bootstrap_score_band(signals, end_date_str,
                         n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """
    Nonparametric bootstrap over the signal set to quantify how much the credit
    score depends on WHICH signals happened to appear -- i.e. how strongly the
    evidence agrees. This is an internal-agreement measure, NOT a probability of a
    rating action and NOT a statement about correctness (garbage in -> garbage out).

    Resamples the signals with replacement n_resamples times, recomputes the
    aggregate score on each resample (reusing compute_credit_score so the formula
    stays single-sourced), and returns:
      - median          : central estimate across resamples
      - band_90         : [5th, 95th] percentile band; tight = agreement, wide = fragile
      - verdict_stability: share of resamples landing in each SCORE_BANDS verdict

    Operates only on the signals passed in (already restricted to the user's date
    window) and dates them against end_date_str -- no external data, nothing to
    hallucinate. Returns None when there are too few signals for a meaningful band.
    """
    n = len(signals)
    if n < MIN_BOOTSTRAP_SIGNALS:
        return None

    rng    = random.Random(seed)
    scores = []
    for _ in range(n_resamples):
        sample = [signals[rng.randrange(n)] for _ in range(n)]
        scores.append(compute_credit_score(sample, end_date_str)["score"])

    scores.sort()

    def _pct(p):
        idx = min(len(scores) - 1, max(0, int(round(p * (len(scores) - 1)))))
        return round(scores[idx], 3)

    band_counts = {}
    for sc in scores:
        for lo, hi, label, _ in SCORE_BANDS:
            if lo <= sc <= hi:
                band_counts[label] = band_counts.get(label, 0) + 1
                break
    verdict_stability = {k: round(v / len(scores), 3) for k, v in band_counts.items()}

    return {
        "median":            _pct(0.5),
        "band_90":           [_pct(0.05), _pct(0.95)],
        "n_resamples":       n_resamples,
        "verdict_stability": verdict_stability,
    }


# ==========================================
# MARKET & COMPANY CONTEXT
# ==========================================

def compute_52w_range(ticker, end_date):
    """
    Trailing-12-month price range AS OF end_date (point-in-time -- only closes on/before
    end_date, no look-ahead), from yfinance. Returns {low, high, last, pct_in_range,
    descriptor} showing where the current price sits within its 52-week band, or None.
    """
    try:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return None
    start    = (end_dt - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    end_excl = (end_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        h = yf.Ticker(yf_symbol(ticker)).history(start=start, end=end_excl, auto_adjust=True)
        closes = [float(c) for c in h["Close"].tolist() if c == c]
    except Exception:  # noqa: BLE001
        return None
    if len(closes) < 2:
        return None

    low, high, last = min(closes), max(closes), closes[-1]
    pct = ((last - low) / (high - low) * 100) if high > low else None
    if pct is None:
        desc = ""
    elif pct >= 80:
        desc = "near 52-wk high"
    elif pct <= 20:
        desc = "near 52-wk low"
    else:
        desc = "mid-range"
    return {"low": round(low, 2), "high": round(high, 2), "last": round(last, 2),
            "pct_in_range": round(pct) if pct is not None else None, "descriptor": desc}


def _fmt_money(x):
    if not x:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"${x / 1e9:.1f}B" if abs(x) >= 1e9 else f"${x / 1e6:.0f}M"


def _cell(df, row, col):
    """Safely read df.loc[row, col] as a float; return None if missing/NaN."""
    try:
        v = float(df.loc[row, col])
        return None if v != v else v  # NaN check
    except Exception:  # noqa: BLE001
        return None


def _fin_summary(fin):
    rev_period = f" ({fin['revenue_period']})" if fin.get("revenue_period") else ""
    mcap = fin.get("market_cap") or fin.get("market_cap_fallback")
    return (f"total debt {_fmt_money(fin.get('total_debt'))}, "
            f"revenue {_fmt_money(fin.get('total_revenue'))}{rev_period}, "
            f"market cap {_fmt_money(mcap)}")


def fetch_company_financials(ticker, as_of_date, info=None,
                             lag_days=FINANCIALS_REPORTING_LAG_DAYS):
    """
    POINT-IN-TIME size snapshot as of as_of_date, from free yfinance quarterly
    statements. Picks the most recent quarter whose period-end is on/before
    (as_of_date - lag_days), so a backtest never uses figures that were probably not
    public yet. Balance-sheet debt is taken from that quarter; revenue is that
    quarter's figure (period-labelled); market cap is filled later from
    price(end) x shares. Falls back to the latest .info snapshot when quarterly
    history does not reach the window. Best-effort -- never raises.
    """
    fin = {"total_debt": None, "total_revenue": None, "revenue_period": None,
           "market_cap": None, "market_cap_fallback": None, "shares_outstanding": None,
           "as_of": None, "basis": "latest snapshot", "source": "Yahoo Finance"}
    tk = None
    try:
        tk = yf.Ticker(yf_symbol(ticker))
        if info is None:
            info = tk.info
    except Exception:  # noqa: BLE001
        info = info or {}
    fin["shares_outstanding"]  = info.get("sharesOutstanding")
    fin["market_cap_fallback"] = info.get("marketCap")

    try:
        cutoff = (datetime.datetime.strptime(as_of_date, "%Y-%m-%d")
                  - datetime.timedelta(days=lag_days)).date()
    except (ValueError, TypeError):
        cutoff = None

    picked = None
    if tk is not None and cutoff is not None:
        try:
            bs = tk.quarterly_balance_sheet
            eligible = sorted(c for c in bs.columns if c.to_pydatetime().date() <= cutoff)
            if eligible:
                picked = eligible[-1]
                fin["total_debt"] = _cell(bs, "Total Debt", picked)
                fin["as_of"]      = picked.strftime("%Y-%m-%d")
                fin["basis"]      = "quarterly (point-in-time)"
                fin["source"]     = "Yahoo Finance (quarterly statements)"
        except Exception:  # noqa: BLE001
            picked = None
    if picked is not None:
        try:
            isq = tk.quarterly_income_stmt
            elig = sorted(c for c in isq.columns if c.to_pydatetime().date() <= cutoff)
            if elig:
                pcol = elig[-1]
                fin["total_revenue"]  = _cell(isq, "Total Revenue", pcol)
                fin["revenue_period"] = "qtr ending " + pcol.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass

    # Fallback: latest snapshot when quarterly history doesn't reach the window.
    if fin["total_debt"] is None and fin["total_revenue"] is None:
        fin["total_debt"]    = info.get("totalDebt")
        fin["total_revenue"] = info.get("totalRevenue")
        fin["basis"]         = "latest snapshot (quarterly history unavailable)"
        fin["source"]        = "Yahoo Finance"
        mrq = info.get("mostRecentQuarter")
        if mrq:
            try:
                fin["as_of"] = datetime.datetime.fromtimestamp(mrq).strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                pass

    fin["summary"] = _fin_summary(fin)
    return fin


def compute_fundamentals_signal(ticker, as_of_date):
    """
    Deterministic credit signal from the quarterly fundamentals trend (no news, no LLM),
    built on compute_financial_panel(). Votes across revenue, EBITDA margin, free cash flow,
    and net leverage:
      - NEGATIVE fires when net leverage rises materially (a reliable warning on its own).
      - POSITIVE requires BROAD corroboration (>=3 metrics improving, none deteriorating), so
        a genuine turnaround (SNDK: revenue/margin/FCF up + deleveraging) scores, while a firm
        that merely deleveraged into trouble (PSKY: leverage down but revenue/margin weak) does
        not -- avoiding the false-positive that got positive filings signals disabled before.
    Confidence = 0.60 + 0.05 x (agreeing metrics), capped. Returns a standard signal dict
    (Financial Risk, source 'filings'), or None when the picture is mixed / data is thin.
    Fires even at zero news coverage -- the fix for fundamentally strong but quiet names.
    """
    panel = compute_financial_panel(ticker, as_of_date)
    if not panel:
        return None
    by = {x["name"]: x for x in panel["metrics"]}

    def nn(name):
        return [v for v in by.get(name, {}).get("values", []) if v is not None]

    def chg(name):
        return by.get(name, {}).get("change_pct")

    pos, neg = [], []

    rc = chg("Revenue")
    if rc is not None and rc > 10:
        pos.append(f"revenue {rc:+.0f}%")
    elif rc is not None and rc < -10:
        neg.append(f"revenue {rc:+.0f}%")

    em = nn("EBITDA margin")
    if len(em) >= 2:
        dpp = em[-1] - em[0]
        if dpp > 2:
            pos.append(f"EBITDA margin {dpp:+.0f}pp")
        elif dpp < -2:
            neg.append(f"EBITDA margin {dpp:+.0f}pp")

    fc, fchg = nn("Free cash flow"), chg("Free cash flow")
    if fc:
        if fc[-1] > 0 and (fchg is None or fchg >= 0):
            pos.append("FCF positive & rising")
        elif fc[-1] < 0:
            neg.append("FCF negative")

    nd = nn("Net debt")
    lev_up = False
    if len(nd) >= 2:
        nd_then, nd_now = nd[0], nd[-1]
        if nd_now > 0 and (nd_now - nd_then) > 0.10 * abs(nd_then or 1):
            neg.append("net leverage rising")
            lev_up = True
        elif nd_now < nd_then:
            pos.append("net cash" if nd_now <= 0 else "net debt falling")

    if lev_up:
        direction, drivers = "negative", neg
    elif len(pos) >= 3 and not neg:
        direction, drivers = "positive", pos
    else:
        return None

    conf = round(min(0.80 if direction == "positive" else 0.85, 0.60 + 0.05 * len(drivers)), 3)

    figs = []
    for nm in ("Revenue", "EBITDA margin", "Free cash flow", "Net debt"):
        v = nn(nm)
        if v:
            figs.append({"metric": nm.lower(),
                         "value": (f"{v[-1]:.1f}%" if by[nm]["unit"] == "pct"
                                   else _fmt_money(v[-1]))})

    last_q = panel["quarters"][-1] if panel.get("quarters") else as_of_date
    summary = (("Broad-based improvement" if direction == "positive" else "Deteriorating fundamentals")
               + ": " + ", ".join(drivers) + f" (through {last_q}, filings)")
    return {
        "date":          last_q,
        "direction":     direction,
        "confidence":    conf,
        "risk_category": "Financial Risk",
        "sp_factor":     "Fundamentals trend (quarterly filings)",
        "event_summary": summary,
        "rationale":     ("Broad, corroborated improvement in cash generation and leverage "
                          "strengthens the credit profile" if direction == "positive"
                          else "Rising net leverage pressures debt service and coverage"),
        "headline":      "Quarterly filings (Yahoo Finance)",
        "url":           "",
        "key_figures":   figs,
        "coverage":      1,
        "source":        "filings",
    }


def compute_financial_panel(ticker, as_of_date, lag_days=FINANCIALS_REPORTING_LAG_DAYS,
                            n_quarters=FINANCIAL_PANEL_QUARTERS):
    """
    Always-on quarterly fundamentals panel from yfinance statements (no news, no LLM):
    revenue, net income, gross/EBITDA/net margins, free cash flow, total debt and net debt
    across the last n_quarters on/before as_of_date, each with a simple trend. Purely
    descriptive context for the analyst -- the scored FR signal is computed separately.
    Best-effort; returns None if statements are unavailable.
    """
    try:
        tk  = yf.Ticker(yf_symbol(ticker))
        isq = tk.quarterly_income_stmt
        bs  = tk.quarterly_balance_sheet
        cf  = tk.quarterly_cashflow
    except Exception:  # noqa: BLE001
        return None
    try:
        cutoff = (datetime.datetime.strptime(as_of_date, "%Y-%m-%d")
                  - datetime.timedelta(days=lag_days)).date()
    except (ValueError, TypeError):
        return None

    if isq is None or getattr(isq, "empty", True):
        return None
    cols = sorted(c for c in isq.columns if c.to_pydatetime().date() <= cutoff)[-n_quarters:]
    if not cols:
        return None

    def series(df, *labels):
        vals = []
        for c in cols:
            v = None
            if df is not None and not getattr(df, "empty", True) and c in df.columns:
                for lab in labels:
                    v = _cell(df, lab, c)
                    if v is not None:
                        break
            vals.append(v)
        return vals

    revenue = series(isq, "Total Revenue")
    net_inc = series(isq, "Net Income", "Net Income Common Stockholders")
    gross   = series(isq, "Gross Profit")
    ebitda  = series(isq, "EBITDA", "Normalized EBITDA")
    ebit    = series(isq, "EBIT", "Operating Income")
    op_cf   = series(cf,  "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex   = series(cf,  "Capital Expenditure")
    fcf_dir = series(cf,  "Free Cash Flow")
    debt    = series(bs,  "Total Debt")
    cash    = series(bs,  "Cash And Cash Equivalents",
                          "Cash Cash Equivalents And Short Term Investments")

    ebitda   = [ebitda[i] if ebitda[i] is not None else ebit[i] for i in range(len(cols))]
    net_debt = [debt[i] - (cash[i] or 0) if debt[i] is not None else None
                for i in range(len(cols))]

    def fcf_at(i):
        if fcf_dir[i] is not None:
            return fcf_dir[i]
        if op_cf[i] is not None:
            return op_cf[i] - (abs(capex[i]) if capex[i] is not None else 0)
        return None
    fcf = [fcf_at(i) for i in range(len(cols))]

    def margin(num, den):
        return [(100.0 * n / d if (n is not None and d not in (None, 0)) else None)
                for n, d in zip(num, den)]

    metrics = [
        {"name": "Revenue",        "unit": "money", "values": revenue},
        {"name": "Net income",     "unit": "money", "values": net_inc},
        {"name": "Gross margin",   "unit": "pct",   "values": margin(gross, revenue)},
        {"name": "EBITDA",         "unit": "money", "values": ebitda},
        {"name": "EBITDA margin",  "unit": "pct",   "values": margin(ebitda, revenue)},
        {"name": "Net margin",     "unit": "pct",   "values": margin(net_inc, revenue)},
        {"name": "Free cash flow", "unit": "money", "values": fcf},
        {"name": "Total debt",     "unit": "money", "values": debt},
        {"name": "Net debt",       "unit": "money", "values": net_debt},
    ]
    for m in metrics:
        nn = [v for v in m["values"] if v is not None]
        if len(nn) >= 2 and nn[0] != 0:
            chg = (nn[-1] - nn[0]) / abs(nn[0])
            m["trend"] = "up" if chg > 0.02 else ("down" if chg < -0.02 else "flat")
            m["change_pct"] = round(chg * 100, 1)
        else:
            m["trend"] = ""
            m["change_pct"] = None

    return {"as_of": as_of_date, "source": "Yahoo Finance (quarterly statements)",
            "quarters": [c.strftime("%Y-%m-%d") for c in cols], "metrics": metrics}


def print_financial_panel(panel, ticker, company_name):
    """Render the quarterly fundamentals panel as a compact metric x quarter table."""
    if not panel:
        return
    qs = panel["quarters"]
    print("\n" + "=" * 65)
    print(f"  FINANCIAL HEALTH (quarterly filings)  -  {company_name} ({ticker})")
    print("=" * 65)
    print(f"  {'Metric':<15}" + "".join(f"{q:>13}" for q in qs) + f"{'trend':>12}")
    for m in panel["metrics"]:
        cells = []
        for v in m["values"]:
            if v is None:
                cells.append("n/a")
            elif m["unit"] == "money":
                cells.append(_fmt_money(v))
            else:
                cells.append(f"{v:.1f}%")
        chg = m.get("change_pct")
        tr  = m.get("trend", "")
        trend_str = f"{tr} {chg:+.0f}%" if (tr and chg is not None) else (tr or "")
        print(f"  {m['name']:<15}" + "".join(f"{c:>13}" for c in cells) + f"{trend_str:>12}")
    print(f"\n  (source: {panel.get('source')}; oldest -> newest quarter; trend = first vs last)")
    print("=" * 65)


def compute_market_context(signals, ticker, start_date, end_date, benchmark=MARKET_BENCHMARK):
    """
    Summary-level market snapshot AS OF end_date, using only real yfinance closes
    inside [start_date, end_date] -- no prices fetched outside the window, nothing
    synthesised. Returns:
      - last_close on/before end_date
      - stock vs benchmark cumulative return over the window (abnormal = stock - bench)
      - chart-ready series (dates, stock_close, benchmark rebased) + signal markers
        (date, direction) for the dashboard.
    This is an EQUITY reaction proxy, not bond repricing. Does NOT annotate individual
    signals. Returns None if price data is unavailable.
    """
    try:
        end_excl = (datetime.datetime.strptime(end_date, "%Y-%m-%d")
                    + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        end_excl = end_date
    try:
        stk = yf.Ticker(yf_symbol(ticker)).history(start=start_date, end=end_excl, auto_adjust=True)
        bmk = yf.Ticker(benchmark).history(start=start_date, end=end_excl, auto_adjust=True)
        if stk.empty or bmk.empty:
            print("    [market snapshot skipped] no price data in window")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"    [market snapshot skipped] {str(exc).splitlines()[0][:120]}")
        return None

    dates     = [d.strftime("%Y-%m-%d") for d in stk.index]
    stk_close = [float(c) for c in stk["Close"].tolist()]
    bmk_map   = {d.strftime("%Y-%m-%d"): float(c) for d, c in zip(bmk.index, bmk["Close"])}
    bmk_close = [bmk_map.get(d) for d in dates]

    last_close = stk_close[-1] if stk_close else None
    stock_ret  = ((stk_close[-1] / stk_close[0] - 1) * 100
                  if len(stk_close) >= 2 and stk_close[0] else None)
    b_first = next((c for c in bmk_close if c), None)
    b_last  = next((c for c in reversed(bmk_close) if c), None)
    bench_ret = (b_last / b_first - 1) * 100 if (b_first and b_last) else None
    abnormal  = (round(stock_ret - bench_ret, 2)
                 if (stock_ret is not None and bench_ret is not None) else None)

    base_s = next((c for c in stk_close if c), None)
    norm_b = [round(c / b_first * base_s, 4) if (c and b_first and base_s) else None
              for c in bmk_close]
    markers = [{"date": s.get("date"), "direction": s.get("direction")} for s in signals]

    return {
        "ticker":                     ticker,
        "benchmark":                  benchmark,
        "source":                     "Yahoo Finance",
        "as_of":                      end_date,
        "window_start":               dates[0] if dates else start_date,
        "window_end":                 dates[-1] if dates else end_date,
        "last_close":                 round(last_close, 2) if last_close else None,
        "stock_return_pct":           round(stock_ret, 2) if stock_ret is not None else None,
        "benchmark_return_pct":       round(bench_ret, 2) if bench_ret is not None else None,
        "abnormal_return_pct":        abnormal,
        "dates":                      dates,
        "stock_close":                stk_close,
        "benchmark_close_normalized": norm_b,
        "signal_markers":             markers,
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
    raw = result.get("score_raw")
    ev  = result.get("evidence_factor")
    if raw is not None and ev is not None and ev < 1.0:
        n = (result.get("negative_signals", 0) + result.get("positive_signals", 0)
             + result.get("neutral_signals", 0))
        print(f"                   (raw {raw:+.3f} x evidence factor {ev:.2f} at n={n} "
              f"-- thin evidence is pulled toward 0)")
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

    stability = result.get("stability")
    if stability:
        lo, hi = stability["band_90"]
        order  = [b[2] for b in SCORE_BANDS]  # most-negative -> most-positive
        vi     = order.index(verdict) if verdict in order else 0
        # "verdict-or-worse" for credit = this band + all more-negative bands
        here_or_worse = sum(
            share for lbl, share in stability["verdict_stability"].items()
            if lbl in order[:vi + 1]
        )
        print(f"\n  Uncertainty    : median {stability['median']:+.3f}   "
              f"90% band [{lo:+.3f}, {hi:+.3f}]")
        print(f"  Verdict stab.  : {here_or_worse * 100:.0f}% of {stability['n_resamples']} "
              f"resamples land {verdict}-or-worse")
        print(f"                   (signal agreement, NOT probability of a rating action)")
    else:
        print(f"\n  Uncertainty    : too few signals for a band "
              f"(need >= {MIN_BOOTSTRAP_SIGNALS})")

    print(f"\n  Score bands :")
    for lo, hi, label, desc in SCORE_BANDS:
        marker = " <--" if verdict == label else ""
        print(f"    [{lo:+.2f} to {hi:+.2f}]  {label}{marker}")
    print("=" * 65 + "\n")


def print_end_date_snapshot(end_date, financials, market_ctx, agg_figures,
                            score_result, ticker, company_name, contradictions=None):
    """
    Single 'as of end_date' panel: the bottom-line verdict, the price/return vs
    benchmark over the window, the company financials, and the hard numbers cited
    across the news -- everything aggregated here rather than repeated per signal.
    """
    neg = score_result.get("negative_signals", 0)
    pos = score_result.get("positive_signals", 0)
    neu = score_result.get("neutral_signals", 0)

    print("\n" + "=" * 65)
    print(f"  SNAPSHOT AS OF {end_date}   -   {company_name} ({ticker})")
    print("=" * 65)
    print(f"\n  Bottom line    : {neg} negative | {pos} positive | {neu} neutral  ->  "
          f"{score_result.get('verdict')}  (score {score_result.get('score', 0):+.3f})")

    for a, b in (contradictions or []):
        print(f"  ! Conflict     : signals #{a} and #{b} are same-date, opposite "
              f"directions -- likely ONE event read two ways; reconcile before acting")

    if market_ctx:
        lc = market_ctx.get("last_close")
        sr = market_ctx.get("stock_return_pct")
        br = market_ctx.get("benchmark_return_pct")
        ab = market_ctx.get("abnormal_return_pct")
        price_str = f"${lc:.2f}" if lc is not None else "n/a"
        if sr is not None and br is not None:
            print(f"  Price          : {price_str}   (window: {ticker} {sr:+.1f}% vs "
                  f"{market_ctx.get('benchmark')} {br:+.1f}%, abnormal {ab:+.1f}%)")
            net = pos - neg
            if ab is not None and net != 0:
                aligned = (ab < 0) if net < 0 else (ab > 0)
                side    = "negative" if net < 0 else "positive"
                if aligned:
                    print(f"                   -> equity already moved with the {side} news "
                          f"(partly reflected in price)")
                else:
                    print(f"                   -> equity has NOT reflected the {side} credit "
                          f"view yet (potential lead-time edge)")
            ws, we = market_ctx.get("window_start"), market_ctx.get("window_end")
            print(f"                   (source: {market_ctx.get('source', 'Yahoo Finance')}, "
                  f"{ws} -> {we}; equity reaction, NOT bond repricing)")
        else:
            print(f"  Price          : {price_str}")

        r52 = market_ctx.get("fiftytwo_week")
        if r52 and r52.get("low") is not None:
            pct  = r52.get("pct_in_range")
            desc = r52.get("descriptor", "")
            pos_str = f"{pct}% of range" if pct is not None else ""
            extra   = "  ".join(x for x in (pos_str, desc) if x)
            print(f"  52-wk range    : ${r52['low']:.2f} - ${r52['high']:.2f}   "
                  f"(current ${r52['last']:.2f}{'  ->  ' + extra if extra else ''})")
    else:
        print("  Price          : market data unavailable")

    if financials:
        asof  = financials.get("as_of")
        src   = financials.get("source", "Yahoo Finance")
        basis = financials.get("basis", "")
        asof_str = f", as of {asof}" if asof else ""
        print(f"  Financials     : {financials.get('summary', 'n/a')}")
        print(f"                   (source: {src}{asof_str}; basis: {basis})")

    if agg_figures:
        print("  Key figures    : (verified present in the cited news; #n = signal above)")
        for f in agg_figures:
            tag  = f"#{f['ref']}" if f.get("ref") else f.get("date", "")
            body = textwrap.fill(f"{f['figure']}  [{tag}]",
                                 width=60, subsequent_indent=" " * 6)
            print(f"    - {body}")
    print("=" * 65 + "\n")


# ==========================================
# MAIN PIPELINE
# ==========================================

def _ask(prompt, default):
    """Print a prompt with a default value and return the user's input (or default)."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def _choose_judge():
    """Show the numbered judge menu and set JUDGE_BACKEND + JUDGE_MODEL globally."""
    global JUDGE_BACKEND, JUDGE_MODEL
    print("\n  Select the judge model (Phase 4):")
    for i, (label, _, _) in enumerate(JUDGE_CHOICES, 1):
        print(f"    {i}. {label}")
    raw = input(f"\n  Choice [1-{len(JUDGE_CHOICES)}, default 1]: ").strip()
    try:
        idx = int(raw) - 1 if raw else 0
        if not 0 <= idx < len(JUDGE_CHOICES):
            raise ValueError
    except ValueError:
        print("  Invalid choice; using 1 (Claude Haiku 4.5 via OpenRouter).")
        idx = 0
    _, JUDGE_BACKEND, JUDGE_MODEL = JUDGE_CHOICES[idx]


USE_NEWS_SUMMARY = True   # generate a short AI news narrative (one extra LLM call/run; off = fallback)


def _claude_text(prompt: str) -> str:
    """Plain-text (non-JSON) completion via the local Claude Code CLI."""
    cmd = [CLAUDE_BIN, "-p", prompt, "--model", JUDGE_MODEL,
           "--output-format", "json", "--allowedTools", ""]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         stdin=subprocess.DEVNULL, timeout=JUDGE_TIMEOUT)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "non-zero exit").strip()[:200])
    outer = json.loads(res.stdout.strip())
    return (outer.get("result", "") if isinstance(outer, dict) else str(outer)) or ""


def _llm_text(prompt: str) -> str:
    """One free-form TEXT completion on the active judge backend (prose, not JSON)."""
    if JUDGE_BACKEND == "gemini":
        return _gemini_chat(prompt, JUDGE_MODEL,
                            closing="Respond now with the summary paragraph only, as plain prose.")
    if JUDGE_BACKEND == "openrouter":
        return _openrouter_chat(prompt, JUDGE_MODEL)
    return _claude_text(prompt)


def summarize_news(company: str, ticker: str, period: str,
                   score_result: dict, signals: list, developing: list) -> str:
    """One short AI narrative of the company's credit-relevant news — so there is ALWAYS a
    readable takeaway, whatever the state (material signals AND/OR developing context). Best-
    effort: on any failure it returns a deterministic one-liner and NEVER raises (a summary
    must not break a scored run)."""
    verdict    = score_result.get("verdict", "")
    score      = score_result.get("score", 0)
    conviction = score_result.get("conviction", "")

    def _line(s):
        body = (s.get("event_summary") or s.get("headline") or "").strip()
        return f"- [{s.get('direction','?')}/{s.get('risk_category','?')}] {s.get('sp_factor','')}: {body}"
    mat = "\n".join(_line(s) for s in (signals or [])) or "  (none cleared the materiality bar)"
    dev = "\n".join(f"- {d.get('headline','')}" for d in (developing or [])) or "  (none)"

    # Deterministic fallback so a summary always exists (LLM off / unavailable / errored).
    tops = ([(s.get("event_summary") or s.get("headline") or "") for s in (signals or [])][:3]
            or [d.get("headline", "") for d in (developing or [])][:3])
    fallback = (f"{verdict} (score {score}, {conviction} conviction)."
                + ("" if not any(tops) else " Key items: " + "; ".join(t for t in tops if t) + "."))
    if not USE_NEWS_SUMMARY:
        return fallback

    prompt = (
        f"You are a credit analyst. In 2-3 concise sentences (~60 words), summarize the "
        f"credit-relevant news for {company} ({ticker}) over {period}. Be factual and specific — "
        f"name the concrete events and figures; do not hype or speculate. State the overall "
        f"direction and its single main driver.\n\n"
        f"Pipeline read: {verdict} (score {score}, {conviction} conviction).\n\n"
        f"Material credit signals:\n{mat}\n\n"
        f"Other developing / reviewed news (context, not scored):\n{dev}\n\n"
        f"Write only the summary paragraph."
    )
    try:
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", (_llm_text(prompt) or "").strip()).strip()
        return text or fallback
    except Exception:  # noqa: BLE001 - a summary must never break the run
        return fallback


def build_news_signal(score_result: dict, signals: list,
                      summary: str = "", developing: list = None) -> dict:
    """Compact NewsSignal for the credit engine (joywin_contracts.NewsSignal): one overall
    credit direction + score + conviction + an AI `summary` + one `events` list of the news.

    Every event is flagged `credit_relevant`: True = a material, scored signal (drives the
    score); False = a neutral/context story that was reviewed but not material. This keeps a
    single news list (not two) while letting the reader tell what actually mattered — and the
    output always carries news, even when nothing cleared the materiality bar."""
    def _item(s: dict, credit_relevant: bool) -> dict:
        return {
            "credit_relevant": credit_relevant,   # True = material (scored); False = neutral context
            "date":          s.get("date", ""),
            "direction":     s.get("direction") or ("" if credit_relevant else "neutral"),
            "risk_category": s.get("risk_category", ""),
            "sp_factor":     s.get("sp_factor", ""),
            "event":         s.get("event_summary") or s.get("headline", ""),
            "headline":      s.get("headline", ""),
            "url":           s.get("url", ""),
            "confidence":    s.get("confidence"),
        }
    events = ([_item(s, True)  for s in (signals or [])]
              + [_item(d, False) for d in (developing or [])])
    return {
        "verdict":    score_result.get("verdict"),
        "score":      score_result.get("score"),
        "conviction": score_result.get("conviction"),
        "summary":    summary,
        "events":     events,
    }


def _pop_output_flag(argv: list) -> tuple:
    """Pull `--output PATH` out of argv (order-independent) so the positional
    TICKER/COMPANY/START/END parsing is unaffected. Returns (argv_without, path)."""
    argv = list(argv)
    path = None
    if "--output" in argv:
        i = argv.index("--output")
        if i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
        else:
            del argv[i]
    return argv, path


def run_pipeline():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    global TICKER, COMPANY_NAME, START_DATE, END_DATE, JUDGE_BACKEND, JUDGE_MODEL, SHOW_EQUITY_NEWS

    print("=" * 65)
    print("  CREDIT RISK EXTRACTION PIPELINE  -  Joywin International")
    print("=" * 65)

    # Non-interactive mode for batch runs / the backtest harness:
    #   python credit_risk_pipeline.py TICKER "Company Name" START END [JUDGE_CHOICE 1-8]
    cli = sys.argv[1:]
    cli, news_output_path = _pop_output_flag(cli)
    if news_output_path is not None and len(cli) < 4:
        # --output signals the engine (headless) is driving; never drop to prompts.
        print("ERROR: --output requires all inputs as arguments: "
              "TICKER \"Company\" START END [JUDGE]  (non-interactive mode cannot prompt).")
        sys.exit(2)
    if len(cli) >= 4:
        TICKER, COMPANY_NAME = cli[0].upper(), cli[1]
        START_DATE, END_DATE = cli[2], cli[3]
        try:
            idx = int(cli[4]) - 1 if len(cli) >= 5 else 0
        except ValueError:
            idx = 0
        if not 0 <= idx < len(JUDGE_CHOICES):
            idx = 0
        _, JUDGE_BACKEND, JUDGE_MODEL = JUDGE_CHOICES[idx]
        if len(cli) >= 6 and cli[5].lower() in ("noequity", "hideequity", "0", "false", "no"):
            SHOW_EQUITY_NEWS = False
        print(f"\n  Non-interactive run (CLI args)")
    else:
        print("\n  Press Enter to keep the default shown in [brackets].\n")
        TICKER       = _ask("Ticker symbol",  TICKER).upper()
        COMPANY_NAME = _ask("Company name",   COMPANY_NAME)
        START_DATE   = _ask("Start date (YYYY-MM-DD)", START_DATE)
        END_DATE     = _ask("End date   (YYYY-MM-DD)", END_DATE)
        _choose_judge()
        SHOW_EQUITY_NEWS = _ask("Include equity/market news in digest? (y/n)",
                                "y").lower().startswith("y")

    print()
    print(f"  Target : {COMPANY_NAME} ({TICKER})")
    print(f"  Period : {START_DATE}  ->  {END_DATE}")
    print(f"  Judge  : {JUDGE_MODEL}  ({JUDGE_BACKEND})\n")

    if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        print("ERROR: Set the FINNHUB_API_KEY environment variable before running.")
        return

    if JUDGE_BACKEND == "openrouter" and not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY (in .env or the environment) to use an "
              "OpenRouter model.")
        return

    if JUDGE_BACKEND == "gemini":
        try:
            _find_gemini_exe()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            print("  WARNING: GOOGLE_CLOUD_PROJECT is not set. A Workspace Google account needs it "
                  "for the free Gemini CLI tier; a personal @gmail account can ignore this.\n")

    print("Loading NLP models...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(HERE / "sector_risk_kw_new.json", "r", encoding="utf-8") as f:
        sp_criteria_master = json.load(f)

    # ── Phase 1: Routing ──────────────────────────────────────────
    print("\n[Phase 1] Data ingestion & sector routing...")

    yf_ticker = yf.Ticker(yf_symbol(TICKER))
    info      = yf_ticker.info
    industry  = info.get("industry", "Unknown")
    print(f"  Yahoo Finance industry : {industry}")

    routed = None
    if USE_LLM_SECTOR_ROUTING:
        routed = match_sp_sector_llm(
            COMPANY_NAME, industry, info.get("longBusinessSummary", ""), sp_criteria_master)
    if routed:
        target_sector, sector_candidates = routed
        print(f"  Matched S&P sector     : {target_sector['sector_name']}  (LLM routing)")
    else:
        target_sector, sector_candidates = match_sp_sector(
            industry, info.get("longBusinessSummary", ""), sp_criteria_master, encoder)
        print(f"  Matched S&P sector     : {target_sector['sector_name']}  (embedding fallback)")
        if len(sector_candidates) > 1:
            alt = ", ".join(f"{n} ({sc})" for n, sc in sector_candidates[1:])
            print(f"  (runner-up sectors     : {alt})")

    financials = fetch_company_financials(TICKER, END_DATE, info)
    print(f"  Financials ({financials['basis']}, as of {financials['as_of']}) : "
          f"{financials['summary']}")

    criteria_labels, criteria_texts = [], []
    for kw in target_sector.get("business_risk_keywords", []):
        criteria_labels.append("Business Risk")
        criteria_texts.append(kw)
    for kw in target_sector.get("financial_risk_keywords", []):
        criteria_labels.append("Financial Risk")
        criteria_texts.append(kw)

    news_data = []
    if USE_FINNHUB:
        finnhub_url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={TICKER}&from={START_DATE}&to={END_DATE}&token={FINNHUB_API_KEY}"
        )
        for attempt in range(3):
            try:
                resp = requests.get(finnhub_url, timeout=15).json()
                news_data = resp if isinstance(resp, list) else []
                break
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                news_data = []
        print(f"  Finnhub articles fetched : {len(news_data)}")

    if USE_GDELT:
        gdelt = fetch_gdelt_news(COMPANY_NAME, TICKER, START_DATE, END_DATE)
        print(f"  GDELT fetched            : {len(gdelt)}")
        before   = len(news_data)
        news_data = merge_news(news_data, gdelt)
        print(f"  Merged feed (deduped)    : {len(news_data)}  (+{len(news_data) - before} net new)")

    if USE_GOOGLE_NEWS:
        gnews = fetch_google_news(COMPANY_NAME, TICKER, START_DATE, END_DATE)
        print(f"  Google News fetched      : {len(gnews)}")
        before   = len(news_data)
        news_data = merge_news(news_data, gnews)
        print(f"  Merged feed (deduped)    : {len(news_data)}  (+{len(news_data) - before} net new)")

    # Entity gate: keep only articles that actually name the issuer. Finnhub tags generic
    # market commentary and competitor stories to big tickers; those are pure downstream
    # noise. Fall back to the full feed only if the gate removes everything (name mismatch).
    # GDELT items are already scoped by an exact company-name-phrase query, so trust them and
    # skip the headline gate (their summary is only the title, which the strict gate would drop
    # ~90% of). The gate still applies to Finnhub/Google News, whose ticker tagging is noisy.
    is_about = build_entity_matcher(COMPANY_NAME, TICKER)
    on_topic = [a for a in news_data if a.get("provider") == "gdelt" or is_about(a)]
    if on_topic:
        dropped = len(news_data) - len(on_topic)
        print(f"  On-topic (names issuer)  : {len(on_topic)}  ({dropped} off-topic dropped)")
        news_data = on_topic
    elif news_data:
        print("  [entity gate matched 0 -- keeping full feed; check company name spelling]")

    # Deterministic order (newest first, headline tiebreak) so the same feed always yields the
    # same triage/judge sequence -- one less source of run-to-run variance.
    news_data.sort(key=lambda a: (-(a.get("datetime") or 0), a.get("headline", "")))

    # "What's happening" digest: every on-topic article, whether or not it scores.
    news_digest = build_news_digest(news_data)

    if MAX_ARTICLES and len(news_data) > MAX_ARTICLES:
        news_data = sorted(news_data, key=lambda a: a.get("datetime", 0),
                           reverse=True)[:MAX_ARTICLES]
        print(f"  Capped to most-recent    : {len(news_data)}")

    near_misses = []   # judged-but-below-bar candidates, shown when no news qualifies

    if USE_LLM_JUDGE:
        sector_name = target_sector["sector_name"]
        # Point-in-time debt line so the judge can size Financial Risk correctly. Independent of
        # which articles we feed it, so compute once and reuse across every recall round.
        mcap = financials.get("market_cap") or financials.get("market_cap_fallback")
        debt_ctx = ""
        if financials.get("total_debt") is not None or mcap:
            debt_ctx = (f"total debt {_fmt_money(financials.get('total_debt'))}, "
                        f"market cap {_fmt_money(mcap)} (as of {financials.get('as_of')})")

        def judge_batch(articles, round_idx):
            """Phases 2-4 for one batch of articles: triage -> scrape survivors -> judge.
            round_idx>=2 relaxes the triage bar (LOOP_RELAX_TRIAGE) to widen the recall net on
            later rounds. Returns (materials, near_misses, scraped_count, fallback_count, kept)."""
            if not articles:
                return [], [], 0, 0, 0
            # ── Phase 2: relevance filter -- LLM triage (default) or cross-encoder (fallback) ──
            if USE_LLM_TRIAGE and (OPENROUTER_API_KEY or JUDGE_BACKEND == "gemini"):
                triage_model = GEMINI_TRIAGE_MODEL if JUDGE_BACKEND == "gemini" else TRIAGE_MODEL
                min_score = LOOP_RELAX_TRIAGE if round_idx >= 2 else TRIAGE_MIN_SCORE
                print(f"\n[Phase 2] LLM credit-materiality triage ({len(articles)} articles "
                      f"-> keep score >= {min_score}, cap {TRIAGE_MAX_KEEP}) via {triage_model}...")
                ranked = triage_articles(articles, COMPANY_NAME, sector_name, criteria_texts)
                kept   = [a for a in ranked if a["_triage_score"] >= min_score][:TRIAGE_MAX_KEEP]
                if not kept:                       # triage failed/empty -> don't starve the judge
                    kept = ranked[:TRIAGE_MAX_KEEP]
                    print("  [triage returned no scored articles -- falling back to most-recent]")
                top_score = ranked[0]["_triage_score"] if ranked else 0
                print(f"  Kept : {len(kept)} articles (materiality >= {min_score}; "
                      f"top score {top_score:.0f}/10)")
            else:
                print(f"\n[Phase 2] Cross-encoder relevance filter ({len(articles)} -> top {CROSSENCODER_TOP_N})...")
                print("  Loading cross-encoder model...")
                kept = filter_with_crossencoder(articles, criteria_texts, criteria_labels)
                print(f"  Kept : {len(kept)} articles (best match to S&P criteria)")

            # ── Phase 3: Scrape only the filtered articles ────────────
            print(f"\n[Phase 3] Scraping {len(kept)} selected articles ({SCRAPE_WORKERS} threads)...")

            def fetch(article):
                # Fully defensive: a single malformed article (bad URL, odd timestamp) must never
                # crash the scrape pool -- pool.map re-raises worker exceptions and would abort the
                # whole issuer. On any error, fall back to the summary.
                article = dict(article)
                summary = article.get("summary", "")
                try:
                    url = article.get("url", "")
                    if url:
                        text, resolved = scrape_full_text(url, summary)
                    else:
                        text, resolved = summary, url
                    article["full_text"] = text
                    if resolved:                     # store the clickable publisher URL
                        article["url"] = resolved
                    scraped = text != summary
                except Exception:  # noqa: BLE001
                    article["full_text"] = summary
                    scraped = False
                try:
                    ts = article.get("datetime") or 0
                    article["date"] = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                except (OSError, OverflowError, ValueError, TypeError):
                    article["date"] = ""             # bad epoch -> leave date blank, never crash
                return article, scraped

            with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as pool:
                fetched_pairs = list(pool.map(fetch, kept))
            sc = sum(1 for _, s in fetched_pairs if s)
            fc = len(fetched_pairs) - sc
            fetched_articles = [a for a, _ in fetched_pairs]
            print(f"  Full-text scraped : {sc} | Summary fallback : {fc}")

            # ── Phase 4: judge (one call per article) ────
            print(f"\n[Phase 4] Judge: {JUDGE_MODEL} via {JUDGE_BACKEND} ({_judge_workers()} workers)...")
            mats, nears = run_judge_articles(fetched_articles, COMPANY_NAME, sector_name, debt_ctx)
            print(f"  Judged material : {len(mats)}  ({len(nears)} reviewed but below the bar)")
            return mats, nears, sc, fc, len(kept)

        # ── First pass on the full merged feed ──
        all_signals, near_misses, scraped_count, fallback_count, filtered_count = \
            judge_batch(news_data, 1)
        judged_keys = {_headline_key(a) for a in news_data}

        # ── Recall-recovery loop: re-search when material signals are thin ──
        # Only material-signal count gates the loop (that was the backtest's blind spot). Each
        # round re-queries GDELT with expanded credit terms, judges only genuinely NEW on-topic
        # articles, and accumulates. Stops on: enough signals, round cap, or a dry round.
        if USE_RECALL_LOOP and USE_GDELT:
            rounds = 0
            while len(all_signals) < LOOP_MIN_MATERIAL and rounds < LOOP_MAX_ROUNDS:
                rounds += 1
                terms = build_recall_query(COMPANY_NAME, target_sector, rounds + 1)
                print(f"\n[Recall loop {rounds}/{LOOP_MAX_ROUNDS}] only {len(all_signals)} material "
                      f"(< {LOOP_MIN_MATERIAL}) -- re-searching GDELT with {len(terms)} credit terms...")
                extra = fetch_gdelt_news(COMPANY_NAME, TICKER, START_DATE, END_DATE,
                                         extra_terms=terms)
                fresh = merge_news([a for a in extra
                                    if _headline_key(a) not in judged_keys and is_about(a)])
                if not fresh:
                    print("  Round added no new on-topic articles -- stopping loop.")
                    break
                for a in fresh:
                    judged_keys.add(_headline_key(a))
                print(f"  {len(fresh)} new on-topic articles -> judging...")
                m, n, sc, fc, kc = judge_batch(fresh, rounds + 1)
                all_signals    += m
                near_misses    += n
                scraped_count  += sc
                fallback_count += fc
                filtered_count += kc
                print(f"  Round {rounds}: +{len(m)} material (running total {len(all_signals)})")

        if USE_FINBERT and all_signals:
            print("\n[Phase 4b] FinBERT tone (second opinion)...")
            all_signals = add_finbert_tone(all_signals)

        # ── Phase 4c: adversarial verification -- precision guard for the wider recall net ──
        if USE_ADVERSARIAL_VERIFY and all_signals:
            print(f"\n[Phase 4c] Adversarial verify: challenging {len(all_signals)} material signals...")
            all_signals, demoted = adversarial_verify(all_signals, COMPANY_NAME, sector_name)
            near_misses += demoted
            print(f"  Confirmed {len(all_signals)} | {len(demoted)} demoted to near-miss")

        # Clean internal scratch fields.
        for s in all_signals + near_misses:
            s.pop("_ce_score", None)
            s.pop("_triage_score", None)
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
            if url:
                text, resolved = scrape_full_text(url, summary)
                if resolved:
                    meta = {**meta, "url": resolved}
            else:
                text = summary
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

    if USE_LLM_JUDGE:
        # Event-level dedup: one event -> one reconciled vote, weighted by coverage.
        clusters       = cluster_events(all_signals, encoder)
        unique_signals = reconcile_events(clusters)
        unique_signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top_signals    = unique_signals[:TOP_N_OUTPUT]
        merged         = len(all_signals) - len(unique_signals)
        print(f"  Articles judged : {len(all_signals)}  ->  Events : {len(unique_signals)}"
              f"  ({merged} same-event votes merged)")
    else:
        unique_signals = deduplicate(all_signals, encoder)
        unique_signals.sort(key=lambda x: x.get(rank_key, 0), reverse=True)
        top_signals    = unique_signals[:TOP_N_OUTPUT]
        print(f"  Unique signals : {len(unique_signals)} | Reporting : {len(top_signals)}")

    # ── Phase 5c: deterministic Financial-Risk signal from quarterly filings ──
    # No news, no LLM -- fires even at zero news coverage; catches quiet leverage moves.
    if USE_LLM_JUDGE and USE_FILINGS_FR:
        print("\n[Phase 5c] Fundamentals signal (quarterly revenue/margin/FCF/leverage trend)...")
        fr_sig = compute_fundamentals_signal(TICKER, END_DATE)
        if fr_sig:
            top_signals.append(fr_sig)
            print(f"  [fundamentals] {fr_sig['direction']}: {fr_sig['event_summary']}")
        else:
            print("  [fundamentals] mixed/immaterial trend (or insufficient quarterly data)")

    # ── Market snapshot as of end date (summary-level, not per signal) ──
    market_ctx = None
    if USE_MARKET_SNAPSHOT and USE_LLM_JUDGE:
        print("\n[Phase 5b] Market snapshot as of end date...")
        market_ctx = compute_market_context(top_signals, TICKER, START_DATE, END_DATE)
        if market_ctx:
            market_ctx["fiftytwo_week"] = compute_52w_range(TICKER, END_DATE)

    # Point-in-time market cap = end-date price x shares outstanding (free, as-of-window).
    if market_ctx and market_ctx.get("last_close") and financials.get("shares_outstanding"):
        financials["market_cap"] = market_ctx["last_close"] * financials["shares_outstanding"]
        financials["market_cap_basis"] = f"price on {market_ctx.get('window_end')} x shares"
        financials["summary"] = _fin_summary(financials)

    # Stable display number per signal (matches the report [n]), so key figures can
    # cite their source compactly as "#n" instead of repeating the headline.
    _r = 0
    for category in ("Business Risk", "Financial Risk"):
        for s in top_signals:
            if s.get("risk_category") == category and "ref" not in s:
                _r += 1
                s["ref"] = _r
    for s in top_signals:
        if "ref" not in s:
            _r += 1
            s["ref"] = _r

    # Aggregate the hard numbers cited across all reported signals (dedup, for summary).
    # Each carries the source signal number (#ref) + headline + url it was verified against.
    agg_figures, _seen = [], set()
    for s in top_signals:
        for f in (s.get("key_figures") or []):
            item = f"{f.get('metric', '')}: {f.get('value', '')}".strip(": ").strip()
            if item and item.lower() not in _seen:
                _seen.add(item.lower())
                agg_figures.append({
                    "figure":  item,
                    "ref":     s.get("ref"),
                    "date":    s.get("date", ""),
                    "source":  (s.get("headline") or "")[:70],
                    "url":     s.get("url", ""),
                })

    # ── Output ────────────────────────────────────────────────────
    print_report(top_signals, TICKER, COMPANY_NAME, target_sector["sector_name"])

    score_result = compute_credit_score(top_signals, END_DATE)
    score_result["stability"] = bootstrap_score_band(top_signals, END_DATE)
    contradictions = find_contradictions(top_signals)
    print_score_summary(score_result, TICKER, COMPANY_NAME)
    print_end_date_snapshot(END_DATE, financials if USE_LLM_JUDGE else None,
                            market_ctx, agg_figures, score_result, TICKER, COMPANY_NAME,
                            contradictions)

    financial_panel = None
    if USE_LLM_JUDGE and USE_FINANCIAL_PANEL:
        financial_panel = compute_financial_panel(TICKER, END_DATE)
        print_financial_panel(financial_panel, TICKER, COMPANY_NAME)

    # Always surface the developing / reviewed credit-relevant stories (context, not scored),
    # so every run shows news even when nothing cleared the materiality bar.
    if USE_LLM_JUDGE:
        print_developing_news(near_misses, has_material=bool(all_signals))

    print_news_digest(news_digest, encoder, company_name=COMPANY_NAME, ticker=TICKER,
                      show_equity=SHOW_EQUITY_NEWS)

    # AI news summary + top developing stories — ALWAYS populated (deterministic fallback if the
    # LLM is off/unavailable), so there is a readable news takeaway whatever the scored state.
    developing = developing_news_items(near_misses, 3) if USE_LLM_JUDGE else []
    news_summary = summarize_news(COMPANY_NAME, TICKER, f"{START_DATE} to {END_DATE}",
                                  score_result, top_signals, developing)
    print("\n" + "-" * 65)
    print("  NEWS SUMMARY")
    print("-" * 65)
    print(news_summary + "\n")

    output = {
        "ticker":            TICKER,
        "company_name":      COMPANY_NAME,
        "applied_sp_sector": target_sector["sector_name"],
        "period":            {"from": START_DATE, "to": END_DATE},
        "run_timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode":              "llm_judge" if USE_LLM_JUDGE else "cosine_only",
        "configuration": {
            "use_llm_judge":        USE_LLM_JUDGE,
            "judge_backend":        JUDGE_BACKEND if USE_LLM_JUDGE else None,
            "judge_model":          JUDGE_MODEL if USE_LLM_JUDGE else None,
            "min_judge_confidence": MIN_JUDGE_CONFIDENCE if USE_LLM_JUDGE else None,
            "crossencoder_model":   CROSSENCODER_MODEL if USE_LLM_JUDGE else None,
            "crossencoder_top_n":   CROSSENCODER_TOP_N if USE_LLM_JUDGE else None,
            "show_equity_news":     SHOW_EQUITY_NEWS,
            "use_finbert":          USE_FINBERT if USE_LLM_JUDGE else False,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "dedup_threshold":      DEDUP_THRESHOLD,
        },
        "stats": {
            "articles_fetched":    len(news_data),
            "after_ce_filter":     filtered_count if USE_LLM_JUDGE else None,
            "full_text_scraped":   scraped_count,
            "summary_fallback":    fallback_count,
            "judged_material":     len(all_signals) if USE_LLM_JUDGE else None,
            "unique_signals":      len(unique_signals),
            "reported_signals":    len(top_signals),
        },
        "company_financials": financials if USE_LLM_JUDGE else None,
        "financial_panel":    financial_panel,
        "developing_news":    developing_news_items(near_misses, 5) if USE_LLM_JUDGE else [],
        "market_context":     market_ctx,
        "key_figures_summary": agg_figures,
        "contradictions":     contradictions,
        "news_summary":       news_summary,
        "credit_score": score_result,
        "signals": top_signals,
        "news_digest": news_digest,
    }

    results_dir = HERE / "results"
    os.makedirs(results_dir, exist_ok=True)
    output_filename = os.path.join(results_dir, f"credit_signals_{TICKER}_{END_DATE}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Full output written to: {output_filename}")

    # Compact NewsSignal for the credit engine, written to the caller-given path
    # (the full results/ JSON above is unaffected — this is the side-input contract).
    if news_output_path:
        news_signal = build_news_signal(score_result, top_signals,
                                        summary=news_summary, developing=developing)
        news_signal["ticker"] = TICKER          # traceability; contract ignores extras
        os.makedirs(os.path.dirname(os.path.abspath(news_output_path)) or ".", exist_ok=True)
        with open(news_output_path, "w", encoding="utf-8") as f:
            json.dump(news_signal, f, indent=2, ensure_ascii=False)
        print(f"NewsSignal written to: {news_output_path}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
