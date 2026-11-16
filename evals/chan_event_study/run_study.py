#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the point-in-time Chan buy/sell-point event study on Yahoo Finance history.

Needs no API key and no LLM. History is split-/dividend-adjusted
(``auto_adjust=True``) so corporate actions do not create fake strokes.

Usage:
    python evals/chan_event_study/run_study.py --symbols AAPL,0700.HK --years 8
    python evals/chan_event_study/run_study.py --symbols AAPL --json-out chan_study.json

Exit codes: 0 = ran, 1 = no data for every symbol, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.chan_event_study.study import (  # noqa: E402
    HORIZONS,
    drift_baseline,
    events_to_dicts,
    replay_points,
    summarise,
)

_INTERVALS = {"1d": "1d", "1w": "1wk"}


def _download(symbol: str, timeframe: str, years: int):
    import yfinance as yf

    df = yf.download(symbol, period=f"{years}y", interval=_INTERVALS[timeframe],
                     auto_adjust=True, progress=False, multi_level_index=False)
    if df is None or df.empty:
        return None
    df = df.reset_index().rename(columns=str.lower).rename(columns={"datetime": "date"})
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def _format_rows(rows) -> str:
    cols = ["side", "kind", "state", "n"]
    for h in HORIZONS:
        cols += [f"mean_{h}", f"hit_{h}", f"invalidated_{h}", f"excess_{h}"]
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append("\t".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", required=True, help="comma-separated Yahoo symbols, e.g. AAPL,0700.HK")
    parser.add_argument("--timeframe", default="1d", choices=sorted(_INTERVALS))
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=120, help="bars before the first replay step")
    parser.add_argument("--json-out", help="write events + summary JSON here")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("no symbols")

    all_events, baselines, skipped = [], {}, []
    for symbol in symbols:
        frame = _download(symbol, args.timeframe, args.years)
        if frame is None or len(frame) <= args.warmup + max(HORIZONS):
            skipped.append(symbol)
            print(f"[skip] {symbol}: not enough history", file=sys.stderr)
            continue
        events = replay_points(frame, args.timeframe, symbol, warmup=args.warmup)
        baselines[symbol] = drift_baseline(frame, args.warmup)
        all_events.extend(events)
        print(f"[ok] {symbol}: {len(frame)} bars, {len(events)} point sightings", file=sys.stderr)

    if not baselines:
        return 1
    rows = summarise(all_events, baselines)
    print(_format_rows(rows))
    print("\ndrift baseline (mean long forward return per bar):")
    for symbol, base in baselines.items():
        print(symbol, {h: round(v, 4) if v is not None else None for h, v in base.items()})
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "symbols": symbols, "skipped": skipped, "timeframe": args.timeframe, "years": args.years,
            "baselines": baselines, "summary": rows, "events": events_to_dicts(all_events),
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
