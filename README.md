# Credit Risk News Intelligence Pipeline

> Automated NLP pipeline that scans financial news and surfaces events material to a company's S&P credit rating — ahead of rating agency action. Designed to replace the manual analyst workflow of reading hundreds of articles per day.

**Joywin International Limited — Quantitative Credit Research**

---

## Overview

Rating agencies lag real-world events by weeks to months. This pipeline creates a structured, dated, directional credit signal stream per issuer from daily news, giving analysts lead time to review or hedge bond positions before an agency acts.

The pipeline runs in five sequential phases:

| Phase | Name | What It Does |
|:---:|---|---|
| 1 | **Data Ingestion & Sector Routing** | Fetches up to 300 articles from Finnhub; maps Yahoo Finance industry to the correct S&P sector via a 100-entry override dictionary |
| 2 | **Cross-Encoder Relevance Filter** | Scores each article headline + summary jointly against every S&P criterion using a cross-encoder model; forwards only the top 25 to scraping |
| 3 | **Selective Full-Text Scraping** | Downloads full article text for the 25 filtered articles only; 8-second hard timeout; falls back to Finnhub summary on paywall or failure |
| 4 | **Claude Judge** | Claude reads the full article text alongside the matched S&P criterion and returns a structured credit signal: direction, confidence, S&P factor, event summary, and rationale |
| 4b | **FinBERT Tone** | Runs FinBERT on the judge's event summary as a secondary signal; flags `tone_alignment = divergent` where surface sentiment disagrees with credit direction. **Claude's judgment is authoritative** — FinBERT has no credit context and cannot reason about bondholder implications |
| 5 | **Dedup, Scoring & Report** | Collapses near-duplicate signals; computes a normalized credit score in [-1, +1]; applies a 5-band verdict with conviction label |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | |
| [Finnhub API key](https://finnhub.io/) | Free tier sufficient |
| [Claude Code CLI](https://claude.ai/code) | Required for Phase 4 — the judge calls `claude -p` using your existing Claude subscription. No separate API key needed. |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `spaCy` (`en_core_web_sm`) is only needed if running the legacy cosine path (`USE_LLM_JUDGE = False`):
> ```bash
> python -m spacy download en_core_web_sm
> ```

### 2. Set your Finnhub API key

```bash
# Windows PowerShell
$env:FINNHUB_API_KEY="your_key_here"

# Windows CMD
set FINNHUB_API_KEY=your_key_here

# macOS / Linux
export FINNHUB_API_KEY=your_key_here
```

Or copy `.env.example` to `.env` and fill in your key (the pipeline reads it via `os.environ`).

### 3. Install Claude Code CLI

```powershell
# Windows (PowerShell)
irm https://claude.ai/install.ps1 | iex
```

Then run `claude` once to log in. The pipeline calls `claude -p` using your existing subscription — no additional API key is required.

---

## Usage

Run the pipeline interactively:

```bash
python credit_risk_pipeline.py
```

On startup, the pipeline prompts for:

```
Ticker symbol   [default: BA]      : STLA
Company name    [default: Boeing]  : Stellantis
Start date      [YYYY-MM-DD]       : 2025-11-12
End date        [YYYY-MM-DD]       : 2026-02-05
Judge model     [haiku/sonnet]     : haiku
```

Results are printed to the console and saved to `results/credit_signals_{TICKER}_{END_DATE}.json`.

---

## Configuration

Key parameters in the `CONFIGURATION` block of `credit_risk_pipeline.py`:

| Parameter | Default | Description |
|---|:---:|---|
| `MAX_ARTICLES` | `300` | Articles fetched from Finnhub per run |
| `CROSSENCODER_TOP_N` | `25` | Articles forwarded to full-text scraping after CE filter |
| `MAX_ARTICLE_CHARS` | `4000` | Character limit on article text sent to the judge |
| `JUDGE_MODEL` | `"haiku"` | `"haiku"` (fast, token-efficient) or `"sonnet"` (higher accuracy) |
| `JUDGE_WORKERS` | `4` | Parallel Claude judge threads |
| `MIN_JUDGE_CONFIDENCE` | `0.6` | Minimum confidence for a signal to be included |
| `DEDUP_THRESHOLD` | `0.88` | Cosine similarity above which two signals are treated as duplicates |
| `TOP_N_OUTPUT` | `25` | Maximum signals in the final ranked report |
| `USE_FINBERT` | `True` | Enable FinBERT tone as secondary signal |
| `USE_LLM_JUDGE` | `True` | Set `False` to run legacy cosine path for A/B comparison |

---

## Credit Scoring

Each judged signal contributes to a normalized score in **[-1.0, +1.0]**:

```
score = Σ( confidence × direction_sign × risk_weight × recency_weight ) / max_possible
```

| Parameter | Values |
|---|---|
| `direction_sign` | `negative = -1`, `positive = +1`, `neutral = 0` |
| `risk_weight` | `Financial Risk = 1.5`, `Business Risk = 1.0` |
| `recency_weight` | `0–30 days = 1.0`, `31–60 days = 0.8`, `61–90 days = 0.6` |

**5-band verdict system:**

| Range | Verdict | Interpretation |
|---|---|---|
| [-1.00, -0.50] | **DOWNGRADE WATCH** | Likely downgrade or outlook negative |
| [-0.50, -0.20] | **CAUTION** | Deteriorating — monitor closely |
| [-0.20, +0.20] | **STABLE** | No material credit action expected |
| [+0.20, +0.50] | **IMPROVING** | Possible outlook positive or stable revision |
| [+0.50, +1.00] | **UPGRADE WATCH** | Likely upgrade or outlook positive |

**Conviction label** (signal count): `LOW` (1–3) · `MEDIUM` (4–7) · `HIGH` (8+)

> Note: A score of -1.0 with LOW conviction (few unanimous signals) is qualitatively weaker than -1.0 with HIGH conviction. Always read score and conviction together.

---

## Output Format

### Console

```
[Phase 4] Judging 18 filtered articles with Claude (haiku)...
  Judged material : 5

[Credit Score]
=================================================================
  STLA (Stellantis)
=================================================================

  Overall score  : -0.743   [CAUTION]
  Interpretation : Deteriorating — monitor closely
  [-1.0 NEGATIVE |################-----------| POSITIVE +1.0]

  Business Risk  : -0.743
  Financial Risk : N/A

  Signals        : 5 negative  |  0 positive  |  0 neutral
  Conviction     : MEDIUM  (4-7 signals)
=================================================================
```

### JSON (`results/credit_signals_{TICKER}_{END_DATE}.json`)

```json
{
  "ticker": "STLA",
  "company_name": "Stellantis",
  "applied_sp_sector": "Auto And Commercial Vehicle Manufacturing",
  "period": { "from": "2025-11-12", "to": "2026-02-05" },
  "run_timestamp": "2026-06-30 16:39",
  "mode": "llm_judge",
  "configuration": {
    "use_llm_judge": true,
    "judge_model": "haiku",
    "min_judge_confidence": 0.6,
    "crossencoder_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "crossencoder_top_n": 25,
    "use_finbert": true,
    "dedup_threshold": 0.88
  },
  "stats": {
    "articles_fetched": 150,
    "after_ce_filter": 25,
    "full_text_scraped": 18,
    "summary_fallback": 7,
    "judged_material": 5,
    "unique_signals": 5,
    "reported_signals": 5
  },
  "credit_score": {
    "score": -0.743,
    "verdict": "CAUTION",
    "description": "Deteriorating — monitor closely",
    "conviction": "MEDIUM",
    "business_risk_score": -0.743,
    "financial_risk_score": null,
    "negative_signals": 5,
    "positive_signals": 0,
    "neutral_signals": 0
  },
  "signals": [
    {
      "date": "2026-01-07",
      "headline": "Stellantis Italy Output Drops 20% to 379,706 Units as Fiat 500 Hybrid Ramps",
      "source": "Yahoo",
      "url": "https://...",
      "matched_criterion": "Disruptive technological shifts toward electrification...",
      "risk_category": "Business Risk",
      "sp_factor": "Electrification transition challenges; severe capacity underutilization",
      "direction": "negative",
      "confidence": 0.87,
      "event_summary": "Stellantis' Italian production fell 20% to 379,706 units in 2025...",
      "rationale": "The collapse in EV demand directly manifests the electrification risk...",
      "finbert_tone": "negative",
      "finbert_score": 0.976,
      "tone_alignment": "aligned"
    }
  ]
}
```

---

## S&P Sector Coverage

The pipeline maps Yahoo Finance industry tags to **37 S&P credit sectors** defined in `sector_risk_kw_new.json`, including:

Banking · Technology · Pharmaceuticals · Auto Manufacturing · Aerospace & Defense · Energy · Retail · Utilities · Mining · Insurance · Real Estate · Transportation · Chemicals · Media · Telecommunications · and more.

---

## Tech Stack

| Library | Role |
|---|---|
| `sentence-transformers` | Cross-encoder relevance filter (Phase 2) and deduplication cosine similarity (Phase 5) |
| `transformers` / `ProsusAI/finbert` | FinBERT tone scoring on judge-written event summaries (Phase 4b) |
| Claude Code CLI (`claude -p`) | Full-article credit judgment — direction, confidence, S&P factor, rationale (Phase 4) |
| `newspaper3k` | Full-text article scraping |
| `yfinance` | Industry classification lookup for sector routing |
| Finnhub REST API | Financial news feed |
| `torch` | Tensor operations for cosine deduplication |
| `spaCy` | Named entity recognition (legacy cosine path only) |

---

## Design Decisions

**Why a cross-encoder instead of cosine similarity for relevance filtering?**
Cosine similarity compares independent embeddings — the article embedding and the criterion embedding are produced separately, so the model never sees them together. A cross-encoder reads both texts jointly and scores relevance directly. For our task (does this article relate to *this specific* S&P criterion?), joint reasoning is strictly more informative than independent encoding.

**Why judge the full article rather than paragraph fragments?**
Paragraph-level chunking breaks pronoun context and splits multi-paragraph arguments. A journalist's second sentence often uses "it" or "the company" — isolated from the first sentence, those paragraphs appear generic. The judge receives the full text (up to 4,000 characters) to preserve argumentative structure.

**Why Claude rather than a fine-tuned model for the judgment step?**
Credit materiality requires understanding of bondholder economics, which differs from equity sentiment. A debt issuance that is operationally routine may be mildly negative for bondholders (dilutive to coverage ratios). FinBERT and similar models trained on equity news cannot make this distinction. Claude reasons from the criterion text provided, giving the judgment credit-specific context rather than generic financial sentiment.

**Why 2–3 sentence natural language criteria in the JSON rather than keywords?**
Short keywords cause embedding collapse — "liquidity risk" maps to a broad financial vector with no discriminative power. Full-sentence S&P scenario descriptions give the cross-encoder a precise, structured target, dramatically improving signal-to-noise.

**Why separate conviction from score?**
The [-1, +1] score can hit its ceiling when all signals point the same direction, regardless of how many signals there are. A score of -1.0 from 2 signals is qualitatively different from -1.0 from 12. The conviction label (LOW/MEDIUM/HIGH) surfaces this without changing the score formula.

---

## Known Limitations

- **Score ceiling:** The score saturates at ±1.0 when all signals agree in direction. Use conviction to assess signal depth.
- **Downgrade detection is stronger than upgrade detection:** Bad events are specific and dateable; recovery is diffuse (multiple gradual improvements). This is a structural characteristic of financial news, not a model artifact.
- **Finnhub coverage varies by issuer size:** Large-cap issuers may generate 200+ articles per window; small-cap issuers may generate fewer than 70, reducing statistical reliability.
- **Paywalled sources:** Full-text scraping falls back to the Finnhub API summary (~150 words) for paywalled articles. Summary-based judgments are less accurate than full-text judgments.
