"""Standalone backtest recap. Reads results/*.json and scores each labelled case in
backtest_cases.csv. Free -- no API, no pipeline runs. Works even if `--dry` doesn't.

Usage:  python recap.py
"""
import csv, glob, json, os
from collections import Counter

NEG_THRESHOLD, POS_THRESHOLD = -0.20, 0.20


def classify(score):
    if score <= NEG_THRESHOLD:
        return "negative"
    if score >= POS_THRESHOLD:
        return "positive"
    return "stable"


def find_result(ticker, end):
    """Prefer the exact end-date file; else fall back to the most recent run for the ticker."""
    exact = f"results/credit_signals_{ticker}_{end}.json"
    if os.path.exists(exact):
        return exact
    cands = sorted(glob.glob(f"results/credit_signals_{ticker}_*.json"), key=os.path.getmtime)
    return cands[-1] if cands else None


def main():
    cases = [r for r in csv.DictReader(open("backtest_cases.csv", encoding="utf-8"))
             if r.get("ticker")]
    rows = []
    for c in cases:
        t, end = c["ticker"].upper(), c["end"]
        path = find_result(t, end)
        if not path:
            rows.append((t, c["expected"], "MISSING", None, 0, False))
            continue
        d      = json.load(open(path, encoding="utf-8"))
        cs     = d.get("credit_score", {})
        score  = cs.get("score", 0.0)
        n      = len(d.get("signals", []))
        pred   = classify(score)
        rows.append((t, c["expected"], pred, score, n, pred == c["expected"]))

    print(f"\n{'ticker':<7}{'expected':<10}{'predicted':<11}{'score':>8}{'n':>4}  hit")
    print("-" * 48)
    for t, exp, pred, score, n, hit in rows:
        s = f"{score:+.3f}" if score is not None else "   --"
        print(f"{t:<7}{exp:<10}{pred:<11}{s:>8}{n:>4}  {'PASS' if hit else 'FAIL'}")

    scored = [r for r in rows if r[2] != "MISSING"]
    hits   = sum(1 for r in scored if r[5])
    missing = [r[0] for r in rows if r[2] == "MISSING"]
    print("-" * 48)
    print(f"Overall: {hits}/{len(scored)} scored"
          + (f"  ({len(missing)} missing: {', '.join(missing)})" if missing else ""))
    for lab in ("negative", "stable", "positive"):
        sub = [r for r in scored if r[1] == lab]
        if sub:
            print(f"  {lab:<9}: {sum(1 for r in sub if r[5])}/{len(sub)}")


if __name__ == "__main__":
    main()
