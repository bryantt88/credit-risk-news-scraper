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
import subprocess
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
PIPELINE = HERE / "credit_risk_pipeline.py"
CASES    = HERE / "backtest_cases.csv"
RESULTS  = HERE / "results"

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


def run_case(case, judge_choice):
    """Run the pipeline non-interactively for one labelled case."""
    cmd = [sys.executable, str(PIPELINE), case["ticker"], case["company"],
           case["start"], case["end"], str(judge_choice)]
    print(f"\n>>> {case['ticker']}  {case['start']} -> {case['end']}  "
          f"(expect {case['expected']}: {case['action']})")
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print("    RUN FAILED: " + " | ".join(tail))
        return False
    return True


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
    args  = sys.argv[1:]
    dry   = "--dry" in args
    only  = None
    judge = 1
    if "--only" in args:
        only = args[args.index("--only") + 1]
    if "--judge" in args:
        judge = int(args[args.index("--judge") + 1])

    cases = load_cases(only)
    if not cases:
        print("No cases found in backtest_cases.csv")
        return

    if not dry:
        est = len(cases)
        print(f"Running {est} case(s) with judge choice {judge} "
              f"(~$0.05 and ~1-2 min per case)...")
        for case in cases:
            run_case(case, judge)

    print_scoreboard([score_case(c) for c in cases])


if __name__ == "__main__":
    main()
