# Credit Risk News Intelligence Pipeline

> Automated NLP pipeline that scans financial news and surfaces events material to a company's S&P credit rating — ahead of rating agency action. Designed to replace the manual analyst workflow of reading hundreds of articles per day.

**Joywin International Limited — Quantitative Credit Research**

---

## Overview

Rating agencies lag real-world events by weeks to months. This pipeline creates a structured, dated, directional credit signal stream per issuer from daily news, giving analysts lead time to review or hedge bond positions before an agency acts.

| Phase | Name | What it does |
|:---:|---|---|
| 1 | **Ingestion, routing & entity gate** | Fetches news from **Finnhub + GDELT** (merged, de-duplicated); routes the issuer to the correct S&P sector with an **LLM sector router** (Yahoo industry + business summary → S&P sector; embedding/override fallback); drops articles that don't actually name the issuer (**entity gate**) |
| 2 | **LLM credit-materiality triage** | A cheap LLM scores every on-topic article 0–10 for credit materiality and forwards the top ~40 to the judge. Replaces the old cross-encoder, which rewarded generic money-language and buried real stories. Cross-encoder is retained as a fallback |
| 3 | **Selective full-text scraping** | Downloads full text for the filtered articles only (**trafilatura → newspaper3k → summary** fallback); resolves Finnhub redirect URLs to the real publisher page |
| 4 | **The judge** | An LLM reads the full article against the S&P sector criteria and returns a structured credit signal: direction, confidence, S&P factor, event summary, rationale, and verbatim `key_figures` (guarded by `verify_figures`, which drops any number not present in the source text). Backend chosen at startup: **OpenRouter** (default, e.g. Haiku 4.5) or local **Claude Code CLI** |
| 4b | **FinBERT tone** | Secondary sentiment check on the judge's event summary; flags `tone_alignment = divergent`. **The judge's credit direction is authoritative** — FinBERT has no bondholder context |
| 5 | **Event dedup, scoring & report** | Clusters same-event signals and reconciles them to **one vote per event** (conflicting reads cancel); weights by outlet coverage; computes a normalized score with evidence shrinkage; 5-band verdict + conviction; **bootstrap uncertainty band**; market snapshot / priced-in check; deterministic fundamentals signal; always-on financial panel + developing-news + equity digest |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | |
| [Finnhub API key](https://finnhub.io/) | Free tier sufficient. GDELT needs no key. |
| [OpenRouter API key](https://openrouter.ai/) | Default path: powers Phase 2 triage and the Phase 4 judge. |
| [Claude Code CLI](https://claude.ai/code) | Optional alternative judge backend (`claude -p`, uses your Claude subscription). Choose it at startup instead of OpenRouter. |

> Running with **no OpenRouter key**: set `USE_LLM_TRIAGE = False` (falls back to the cross-encoder) and pick the Claude CLI judge at startup.

---

## Setup

```bash
pip install -r requirements.txt
```

Set keys via environment or a `.env` file (read with `os.environ`):

```bash
# macOS / Linux
export FINNHUB_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
```

```powershell
# Windows PowerShell
$env:FINNHUB_API_KEY="your_key"
$env:OPENROUTER_API_KEY="your_key"
```

Optional Claude CLI backend:

```powershell
irm https://claude.ai/install.ps1 | iex   # then run `claude` once to log in
```

`spaCy` (`en_core_web_sm`) is only needed for the legacy cosine path (`USE_LLM_JUDGE = False`).

---

## Usage

```bash
python credit_risk_pipeline.py
```

Interactive startup prompts for ticker, company, date window, and judge backend/model. Non-interactive (batch) mode:

```bash
python credit_risk_pipeline.py TICKER "Company Name" START_DATE END_DATE [JUDGE_CHOICE]
```

Results print to the console and save to `results/credit_signals_{TICKER}_{END_DATE}.json`.

**Validation harness** — batch-runs the labelled cases in `backtest_cases.csv` and prints a scoreboard / confusion matrix:

```bash
python backtest_harness.py            # run all cases
python recap.py                       # free re-score from saved results/ JSONs (no API)
```

---

## Configuration

Key parameters in the `CONFIGURATION` block of `credit_risk_pipeline.py`:

| Parameter | Default | Description |
|---|:---:|---|
| `MAX_ARTICLES` | `300` | Finnhub articles fetched per run |
| `USE_FINNHUB` / `USE_GDELT` | `True` / `True` | News sources (merged + de-duplicated); GDELT is query-scoped and cached 12h |
| `USE_LLM_SECTOR_ROUTING` | `True` | LLM maps Yahoo industry → S&P sector (embedding/override fallback) |
| `USE_LLM_TRIAGE` | `True` | Phase 2 = LLM triage; `False` → cross-encoder fallback |
| `TRIAGE_MODEL` | `google/gemini-2.5-flash-lite` | Cheap OpenRouter model for triage |
| `TRIAGE_MIN_SCORE` | `4` | Keep articles scoring ≥ this (0–10) |
| `TRIAGE_MAX_KEEP` | `40` | Hard cap of articles forwarded to the judge |
| `CROSSENCODER_TOP_N` | `40` | Top-N when the cross-encoder fallback is used |
| `JUDGE_BACKEND` / `JUDGE_MODEL` | chosen at startup | `openrouter` (e.g. Haiku 4.5) or `claude` CLI |
| `MIN_JUDGE_CONFIDENCE` | `0.65` | Minimum confidence for a signal to count |
| `JUDGE_WORKERS` | `6` | Parallel judge threads |
| `EVENT_CLUSTER_SIM` / `EVENT_CLUSTER_DAYS` | `0.50` / `3` | Same-event clustering (cosine within N days) |
| `COVERAGE_WEIGHT_K` / `_MAX` | `0.25` / `1.75` | Outlet-coverage weight `w = 1 + K·ln(outlets)`, capped |
| `SCORE_SHRINKAGE_K` | `2` | Evidence shrinkage `n/(n+K)` — tempers thin-evidence scores |
| `BOOTSTRAP_RESAMPLES` / `_SEED` | `1000` / `42` | Uncertainty band (needs ≥4 signals); reproducible |
| `USE_MARKET_SNAPSHOT` | `True` | Price vs. benchmark + abnormal-return "priced-in" check |
| `USE_FILINGS_FR` / `USE_FINANCIAL_PANEL` | `True` / `True` | Deterministic fundamentals signal + always-on quarterly panel |
| `SHOW_EQUITY_NEWS` / `DIGEST_TOP_EQUITY` | `True` / `5` | Secondary equity digest, ranked by business substance |
| `DEVELOPING_NEWS_N` | `3` | Min neutral-but-relevant stories always shown (context, not scored) |
| `USE_FINBERT` | `True` | FinBERT tone as secondary signal |
| `USE_LLM_JUDGE` | `True` | `False` = legacy cosine path for A/B comparison |

---

## Credit scoring

Each event (after dedup) contributes to a normalized score in **[-1.0, +1.0]**:

```
raw   = Σ( confidence × direction_sign × risk_weight × recency_weight × coverage_weight ) / max_possible
score = raw × n / (n + SCORE_SHRINKAGE_K)      # evidence shrinkage tempers thin evidence
```

| Parameter | Values |
|---|---|
| `direction_sign` | `negative = -1`, `positive = +1`, `neutral = 0` |
| `risk_weight` | `Financial Risk = 1.5`, `Business Risk = 1.0` |
| `recency_weight` | `0–30 d = 1.0`, `31–60 d = 0.8`, `61–90 d = 0.6` |
| `coverage_weight` | `1 + 0.25·ln(outlets)`, capped at `1.75` |

**5-band verdict:** DOWNGRADE WATCH `[-1.00,-0.50]` · CAUTION `[-0.50,-0.20]` · STABLE `[-0.20,+0.20]` · IMPROVING `[+0.20,+0.50]` · UPGRADE WATCH `[+0.50,+1.00]`.

**Conviction** (signal count): `LOW` 1–3 · `MEDIUM` 4–7 · `HIGH` 8+. A **bootstrap band** (median + 90% range + verdict-stability %) reports how firm the score is; it abstains below 4 signals. Read score, conviction, and band together.

---

## Beyond the score

- **Event-level dedup** — one real event = one vote. Reworded duplicates and cross-category splits are clustered; conflicting reads net out; outlet count feeds the coverage weight.
- **Priced-in check** — stock return vs. benchmark around the window; a negative signal the equity hasn't reflected is where the lead-time edge lives.
- **Fundamentals signal** — deterministic read of quarterly filings (revenue / EBITDA margin / FCF / net leverage); negative on rising leverage, positive only on broad corroborated improvement.
- **Financial panel** — always-on quarterly table (revenue, margins, FCF, debt, net debt) with trends.
- **Developing news + equity digest** — context stories and stock-moving equity news (ranked by business substance), shown separately from the scored credit report.

---

## S&P sector coverage

Routes issuers to the S&P credit sectors defined in `sector_risk_kw_new.json` (Banking · Technology · Pharmaceuticals · Auto Manufacturing · Aerospace & Defense · Energy · Retail · Utilities · Mining · Insurance · Real Estate · Transportation · Chemicals · Media · Telecommunications · and more).

---

## Tech stack

| Library / service | Role |
|---|---|
| OpenRouter | Triage (Phase 2) + default judge (Phase 4) |
| Claude Code CLI (`claude -p`) | Alternative judge backend |
| Finnhub REST API · GDELT DOC 2.0 | News feeds (GDELT keyless, query-scoped, cached) |
| `sentence-transformers` | Cross-encoder fallback filter + cosine clustering/dedup |
| `transformers` / `ProsusAI/finbert` | Secondary tone scoring |
| `trafilatura` / `newspaper3k` | Full-text article scraping |
| `yfinance` | Sector routing, point-in-time financials, market snapshot, fundamentals |
| `torch` | Tensor ops for embeddings/cosine |
| `spaCy` | NER (legacy cosine path only) |

---

## Design decisions

**Why LLM triage replaced the cross-encoder (Phase 2).** The cross-encoder scored short headlines against dense criteria prose and rewarded generic money-language — it once ranked a bystander's "$30B debt" story #1 and buried the issuer's real $135B-capex story at rank 74. A cheap LLM reads each article for *credit materiality* and reasons a step ahead (capex → leverage → credit-negative). The cross-encoder remains as a no-key fallback.

**Why judge the full article, not fragments.** Paragraph chunking breaks pronoun and multi-paragraph context. The judge gets the full text so it can follow the argument, then extracts `key_figures` verbatim (verified against the source to guarantee zero fabricated numbers).

**Why an LLM judge rather than a fine-tuned sentiment model.** Credit materiality is bondholder economics, not equity sentiment — a routine debt issuance can be mildly *negative* for creditors. Models trained on equity news can't make that call; the judge reasons from the S&P criteria provided.

**Why event-level dedup.** One event covered by five outlets, or split across Business/Financial risk, would otherwise vote five times and inflate conviction. Collapsing to one reconciled vote (with coverage as a mild, capped weight) fixed the largest source of score error.

**Why separate conviction and a bootstrap band from the score.** The [-1,+1] score saturates when signals agree; conviction and the band expose how much evidence and agreement actually back a number, without distorting the score itself.

---

## Known limitations

- **Score ceiling** — saturates at ±1.0 when signals agree; use conviction + band for depth.
- **Downgrade detection > upgrade detection** — bad events are specific and dateable; recovery is diffuse. Structural to financial news; quiet deleveraging often lives in filings, not headlines (partly addressed by the fundamentals signal).
- **Coverage varies by issuer size** — small-caps may yield few articles, reducing reliability; very old windows can exceed Finnhub's ~1yr free history.
- **Run-to-run variance** — judge nondeterminism + a shifting news feed mean single runs aren't perfectly reproducible; the harness is the gate.
- **Paywalls** — scraping falls back to a short summary; summary-based judgments are weaker than full-text.
