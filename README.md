# Credit Risk Extraction Pipeline

> Automated NLP pipeline that scans daily financial news and surfaces the events most material to a company's S&P credit rating — replacing the manual analyst workflow of reading hundreds of articles per day.


---

## How It Works

The pipeline runs in five sequential phases designed to maximise signal precision and eliminate stock-market noise:

| Phase | Name | What It Does |
|:---:|---|---|
| 1 | **Data Ingestion & Routing** | Fetches news from Finnhub API; maps Yahoo Finance industry to the correct S&P sector via a curated 100-entry override dictionary |
| 2 | **Parallel Full-Text Scraping** | Downloads complete articles across 15 concurrent threads via `newspaper3k`; falls back to Finnhub summary on paywalls or failures |
| 3 | **Entity Filtering** | spaCy NLP bouncer — discards paragraphs where the target company is not a primary actor (not just quoted or mentioned in passing) |
| 4 | **Batch Semantic Matching** | Encodes all candidate paragraphs in one batched call; computes a full cosine-similarity matrix against S&P risk criteria; a confidence-gap filter rejects ambiguous matches |
| 5 | **Deduplication & Ranking** | Collapses near-duplicate signals from multiple outlets; ranks by relevance score |

---

## Setup

### Prerequisites
- Python 3.9+
- A free [Finnhub API key](https://finnhub.io/)

### Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Set your API key

```bash
# Windows CMD
set FINNHUB_API_KEY=your_key_here

# PowerShell
$env:FINNHUB_API_KEY="your_key_here"

# macOS / Linux
export FINNHUB_API_KEY=your_key_here
```

---

## Usage

Edit the `CONFIGURATION` block at the top of `credit_risk_pipeline.py`:

```python
TICKER       = "AAPL"          # NYSE / NASDAQ ticker symbol
COMPANY_NAME = "Apple"         # Full company name used for entity matching
START_DATE   = "2026-06-01"    # YYYY-MM-DD
END_DATE     = "2026-06-30"    # YYYY-MM-DD
```

Then run:

```bash
python credit_risk_pipeline.py
```

Results are printed to the console and saved as `credit_signals_{TICKER}_{END_DATE}.json`.

---

## Tuning Parameters

Adjust these in the `CONFIGURATION` block to balance **precision vs. recall**:

| Parameter | Default | Effect |
|---|:---:|---|
| `SIMILARITY_THRESHOLD` | `0.42` | Raise to reduce false positives; lower to catch more signals |
| `MIN_CONFIDENCE_GAP` | `0.04` | Raise to reject more ambiguous matches |
| `MAX_SIGNALS_PER_ARTICLE` | `2` | Prevents one article from dominating the output |
| `SCRAPE_WORKERS` | `15` | Parallel HTTP threads for article scraping |
| `DEDUP_THRESHOLD` | `0.88` | Cosine similarity above which two signals are treated as duplicates |
| `TOP_N_OUTPUT` | `25` | Number of signals in the final ranked report |

---

## Output Format

Console output is grouped by risk category and sorted by score. A full JSON report is also saved:

```json
{
  "ticker": "BAC",
  "company_name": "Bank of America",
  "applied_sp_sector": "Financial Services Finance Companies",
  "period": { "from": "2026-05-01", "to": "2026-06-30" },
  "run_timestamp": "2026-06-30 14:22",
  "stats": {
    "articles_fetched": 289,
    "full_text_scraped": 41,
    "summary_fallback": 248,
    "raw_signals": 18,
    "unique_signals": 12,
    "reported_signals": 12
  },
  "signals": [
    {
      "date": "2026-06-15",
      "headline": "Bank of America raises dividend after Fed stress test...",
      "risk_category": "Financial Risk",
      "matched_criterion": "...",
      "similarity_score": 0.5312,
      "confidence_gap": 0.0821,
      "extracted_chunk": "...",
      "url": "https://..."
    }
  ]
}
```

---

## S&P Sector Coverage

The pipeline maps Yahoo Finance industry tags to **37 S&P credit sectors**, including:

Banking · Technology · Pharmaceuticals · Energy · Retail · Utilities · Mining · Insurance · Real Estate · Transportation · and more.

---

## Tech Stack

| Library | Role |
|---|---|
| `spacy` | Named entity recognition & dependency parsing |
| `sentence-transformers` | Semantic vector encoding (`all-MiniLM-L6-v2`) |
| `newspaper3k` | Full-text article scraping |
| `yfinance` | Industry classification lookup |
| `finnhub` (REST API) | Financial news feed |
| `torch` | Tensor operations for batch cosine similarity |

---

## Design Decisions

**Why paragraph-level chunking?** Sentence-level scanning breaks pronoun context (a journalist's second sentence often uses "it" or "they"). Paragraphs preserve enough context for the NLP model to understand what the text is actually about.

**Why JSON scenario descriptions instead of short keywords?** Short 2–3 word keywords cause "vector dilution" — the embedding collapses to a generic financial concept with no discriminative power. Full 2–3 sentence S&P event scenarios give the model a precise target to aim at.

**Why a confidence gap filter?** When a paragraph is generic financial language, all S&P criteria score similarly (e.g. 0.38, 0.37, 0.36). A gap filter rejects these — only paragraphs with one clearly dominant match are flagged.

---

*Joywin International Limited — Quantitative Credit Research*
