"""
Backtest Harness
================
Runs the credit pipeline non-interactively over every labelled case in
backtest_cases.csv and scores predictions against actual S&P rating actions.
This is the measurement layer: every judge-prompt or scoring change should be
evaluated here, not by eyeballing a single issuer.

Usage
-----
    python backtest_harness.py                 # run all cases with judge choice 1
    python backtest_harness.py --judge 5       # use another judge (menu number)
    python backtest_harness.py --only STLA,CE  # subset of tickers
    python backtest_harness.py --dry           # score existing results/ JSONs only,
                                               # no pipeline runs, no API cost

Label file (backtest_cases.csv)
-------------------------------
    ticker,company,start,end,expected,action_date,action
    expected is one of: negative | stable | positive
    (negative = downgrade/CreditWatch-negative/outlook-negative, positive = upgrade,
     stable = affirmation. Windows should END BEFORE action_date to test lead time.)

Prediction mapping (mirrors SCORE_BANDS boundaries)
---------------------------------------------------
    score <= -0.20  -> negative
    score >= +0.20  -> positive
    otherwise       -> stable

NOTE: each non-dry run calls the LLM judge (~$0.05/issuer at Haiku 4.5 prices via
OpenRouter) and takes ~1-2 min per case. Results land in results/ as usual.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
PIPELINE = HERE / "credit_risk_pipeline.py"
CASES    = HERE / "backtest_cases.csv"
RESULTS  = HERE / "results"

# Gemini account pool for the $0 backend (--accounts). Rotating issuers across two logged-in
# Google accounts roughly doubles the daily 2.5-Pro quota. Account A = the default login
# (~/.gemini + its GOOGLE_CLOUD_PROJECT); account B lives in its own USERPROFILE dir with its
# own GCP project. Each case is a separate subprocess run fully before the next, so there is no
# concurrent use of the two logins. Edit the paths/projects here if the setup changes.
GEMINI_ACCOUNTS = [
    {"name": "A", "userprofile": None,           "project": None},                # default login
    {"name": "B", "userprofile": r"C:\gemini_b", "project": "gemini-cli-502606"},
]

NEG_THRESHOLD = -0.20
POS_THRESHOLD = +0.20


def classify(score):
    if score <= NEG_THRESHOLD:
        return "negative"
    if score >= POS_THRESHOLD:
        return "positive"
    return "stable"


def load_cases(only=None):
    with open(CASES, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("ticker")]
    if only:
        keep = {t.strip().upper() for t in only.split(",")}
        rows = [r for r in rows if r["ticker"].upper() in keep]
    return rows


def result_path(case):
    return RESULTS / f"credit_signals_{case['ticker'].upper()}_{case['end']}.json"


def run_case(case, judge_choice, attempts=2, account=None):
    """Run the pipeline non-interactively for one labelled case. Retries once on a non-zero exit
    (e.g. a native segfault in concurrent HTML parsing) -- the triage/verdict caches make the
    retry fast and it usually clears the transient crash. Each case is its own subprocess, so a
    crash never takes down the rest of the batch.

    account (optional): {userprofile, project} selecting which Gemini login this case runs on.
    Passed to the subprocess via env -- USERPROFILE redirects the CLI to that account's ~/.gemini,
    GOOGLE_CLOUD_PROJECT selects its GCP project. None = the default login."""
    cmd = [sys.executable, str(PIPELINE), case["ticker"], case["company"],
           case["start"], case["end"], str(judge_choice)]
    env = os.environ.copy()
    env["USE_NEWS_SUMMARY"] = "0"   # backtests score only — skip the per-case AI summary LLM call
    label = ""
    if account:
        if account.get("userprofile"):
            env["USERPROFILE"] = account["userprofile"]
        if account.get("project"):
            env["GOOGLE_CLOUD_PROJECT"] = account["project"]
        label = f"   [Gemini account {account['name']}]"
    print(f"\n>>> {case['ticker']}  {case['start']} -> {case['end']}  "
          f"(expect {case['expected']}: {case['action']}){label}")
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", env=env)
        if proc.returncode == 0:
            return True
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        note = "RUN FAILED" if attempt == attempts else f"run failed (attempt {attempt}) -- retrying"
        print(f"    {note} (exit {proc.returncode}): " + " | ".join(tail))
    return False


def score_case(case):
    """Read the run's JSON output and compare prediction vs expected label."""
    path = result_path(case)
    if not path.exists():
        return {**case, "score": None, "predicted": "MISSING",
                "signals": 0, "hit": False}
    data  = json.loads(path.read_text(encoding="utf-8"))
    cs    = data.get("credit_score", {})
    score = cs.get("score", 0.0)
    return {
        **case,
        "score":     score,
        "score_raw": cs.get("score_raw"),
        "verdict":   cs.get("verdict", ""),
        "signals":   len(data.get("signals", [])),
        "predicted": classify(score),
        "hit":       classify(score) == case["expected"],
    }


def print_scoreboard(rows):
    print("\n" + "=" * 78)
    print("  BACKTEST SCOREBOARD")
    print("=" * 78)
    print(f"  {'ticker':<7}{'expected':<10}{'predicted':<11}{'score':>7}  "
          f"{'raw':>7}  {'n':>3}  {'hit':<5} action")
    print("  " + "-" * 74)
    for r in rows:
        s   = f"{r['score']:+.3f}" if r["score"] is not None else "  --  "
        raw = f"{r['score_raw']:+.3f}" if r.get("score_raw") is not None else "  --  "
        hit = "PASS" if r["hit"] else "FAIL"
        print(f"  {r['ticker']:<7}{r['expected']:<10}{r['predicted']:<11}{s:>7}  "
              f"{raw:>7}  {r['signals']:>3}  {hit:<5} {r['action'][:34]}")

    # Confusion matrix (rows = expected, cols = predicted).
    labels = ["negative", "stable", "positive"]
    matrix = {e: {p: 0 for p in labels + ["MISSING"]} for e in labels}
    for r in rows:
        if r["expected"] in matrix:
            matrix[r["expected"]][r.get("predicted", "MISSING")] += 1
    print("\n  Confusion matrix (rows=expected, cols=predicted):")
    print(f"  {'':<10}" + "".join(f"{p:<10}" for p in labels))
    for e in labels:
        print(f"  {e:<10}" + "".join(f"{matrix[e][p]:<10}" for p in labels))

    scored = [r for r in rows if r["score"] is not None]
    hits   = sum(1 for r in scored if r["hit"])
    downs  = [r for r in scored if r["expected"] == "negative"]
    ups    = [r for r in scored if r["expected"] == "positive"]
    print(f"\n  Overall hit rate    : {hits}/{len(scored)}"
          f"{'' if len(scored) == len(rows) else f'  ({len(rows)-len(scored)} missing)'}")
    if downs:
        print(f"  Downgrade-side hits : {sum(r['hit'] for r in downs)}/{len(downs)}")
    if ups:
        print(f"  Upgrade-side hits   : {sum(r['hit'] for r in ups)}/{len(ups)}")
    print("\n  NOTE: n=" + str(len(rows)) + " labelled cases. Grow backtest_cases.csv "
          "toward 20-30 before trusting any hit-rate.")
    print("=" * 78 + "\n")


def main():
    args   = sys.argv[1:]
    dry    = "--dry" in args
    rotate = "--accounts" in args      # rotate issuers across the GEMINI_ACCOUNTS pool
    only   = None
    judge  = 1
    pin    = None                      # --account NAME: pin every case to one account
    if "--only" in args:
        only = args[args.index("--only") + 1]
    if "--judge" in args:
        judge = int(args[args.index("--judge") + 1])
    if "--account" in args:
        name = args[args.index("--account") + 1]
        pin  = next((a for a in GEMINI_ACCOUNTS if a["name"] == name), None)
        if pin is None:
            print(f"Unknown account '{name}'; valid: {[a['name'] for a in GEMINI_ACCOUNTS]}")
            return

    cases = load_cases(only)
    if not cases:
        print("No cases found in backtest_cases.csv")
        return

    if not dry:
        est = len(cases)
        if pin is not None:
            print(f"Running {est} case(s) with judge choice {judge}, pinned to Gemini account {pin['name']}...")
        elif rotate:
            print(f"Running {est} case(s) with judge choice {judge}, rotating across "
                  f"{len(GEMINI_ACCOUNTS)} Gemini accounts ({', '.join(a['name'] for a in GEMINI_ACCOUNTS)})...")
        else:
            print(f"Running {est} case(s) with judge choice {judge} (~1-2 min per case)...")
        for i, case in enumerate(cases):
            if pin is not None:
                account = pin
            elif rotate:
                account = GEMINI_ACCOUNTS[i % len(GEMINI_ACCOUNTS)]
            else:
                account = None
            run_case(case, judge, account=account)

    print_scoreboard([score_case(c) for c in cases])


if __name__ == "__main__":
    main()
