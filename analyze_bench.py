#!/usr/bin/env python3
"""
analyze_bench.py — summarize stt_hk benchmark feedback.

Usage:
    .venv/bin/python analyze_bench.py [--feedback .output/feedback.jsonl]
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict

SPOT          = Path(__file__).parent
FEEDBACK_FILE = SPOT / ".output" / "feedback.jsonl"
BENCH_STATE   = SPOT / ".output" / "bench_state.json"


def load(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def normalize_rating(raw) -> float:
    """Return a 0–1 score regardless of whether rating is int (1–5) or legacy str."""
    if isinstance(raw, int):
        return (raw - 1) / 4
    return {"perfect": 1.0, "good": 0.5, "bad": 0.0}.get(raw, 0.5)


def analyze(entries: list[dict]) -> dict:
    models = defaultdict(lambda: {"scores": [], "rtf": [], "audio_secs": [], "ratings": []})
    for e in entries:
        m = models[e["model"]]
        m["scores"].append(normalize_rating(e["rating"]))
        m["ratings"].append(e["rating"])
        if e.get("rtf"):
            m["rtf"].append(e["rtf"])
        if e.get("audio_secs"):
            m["audio_secs"].append(e["audio_secs"])
    return dict(models)


def avg_score(scores: list) -> float:
    return 100 * sum(scores) / len(scores) if scores else 0.0


def fmt_bar(value: float, width: int = 20) -> str:
    filled = round(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


def avg(lst: list, fmt=".2f") -> str:
    return f"{sum(lst)/len(lst):{fmt}}" if lst else "n/a"


def print_report(entries: list[dict], stats: dict):
    models = sorted(stats.items(), key=lambda kv: avg_score(kv[1]["scores"]), reverse=True)

    print(f"\nstt_hk benchmark — {len(entries)} rated transcriptions\n")
    print(f"{'Model':<18}  {'n':>4}  {'Score':>6}  {'Quality':22}  {'avg rating':>10}  {'RTF':>6}  {'avg s':>6}")
    print("─" * 82)

    for model, data in models:
        n = len(data["scores"])
        if n == 0:
            continue
        s = avg_score(data["scores"])
        # convert scores (0–1) back to 1–5 scale for display
        avg_rating = avg([s * 4 + 1 for s in data["scores"]], fmt=".1f")
        print(
            f"{model:<18}  {n:>4}  {s:>5.1f}%  {fmt_bar(s)}  "
            f"{avg_rating:>10}  "
            f"{avg(data['rtf']):>6}  {avg(data['audio_secs']):>6}"
        )

    print()

    # breakdown by clip length
    print("Quality by clip length (all models combined):")
    buckets: dict[str, list] = defaultdict(list)
    for e in entries:
        secs = e.get("audio_secs", 0)
        lo = int(secs // 10) * 10
        bucket = f"{lo:>2}–{lo+10}s"
        buckets[bucket].append(normalize_rating(e["rating"]))

    if buckets:
        print(f"  {'Clip length':<10}  {'n':>4}  {'Score':>6}  Quality")
        print("  " + "─" * 46)
        for bucket in sorted(buckets):
            scores = buckets[bucket]
            s = avg_score(scores)
            print(f"  {bucket:<10}  {len(scores):>4}  {s:>5.1f}%  {fmt_bar(s, 16)}")
    print()

    best = models[0][0] if models else "n/a"
    print(f"Best model so far: {best}\n")


def clear_reminder():
    try:
        state = json.loads(BENCH_STATE.read_text())
        if state.pop("needs_review", None):
            BENCH_STATE.write_text(json.dumps(state))
            print("(Benchmark reminder cleared.)\n")
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=Path, default=FEEDBACK_FILE)
    args = parser.parse_args()

    if not args.feedback.exists():
        print(f"No feedback file found at {args.feedback}")
        print("Run stt_hk a few times and rate some transcriptions first.")
        sys.exit(1)

    entries = load(args.feedback)
    if not entries:
        print("Feedback file is empty.")
        sys.exit(1)

    stats = analyze(entries)
    print_report(entries, stats)
    clear_reminder()


if __name__ == "__main__":
    main()
