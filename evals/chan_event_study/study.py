# -*- coding: utf-8 -*-
"""Point-in-time event study of the Chan buy/sell points produced by ``src.chan.engine``.

The engine is re-run on ``frame[: t + 1]`` for every bar ``t``, so each point is
only seen with the data that existed at that bar. A point is recorded the first
time it appears in a given ``state`` (``candidate`` and ``confirmed``) while still
valid; the entry is that bar's close. Outcomes are measured on later bars only:

* forward return after ``h`` bars (sign-adjusted so a good sell is positive),
* whether price crossed the point's ``invalidation_price`` within ``h`` bars,
* excess return over the mean forward return of every bar in the same sample
  (the drift a random entry would have earned).

This is a reporter for judging whether the structure carries signal; it is not a
trading backtest (no costs, no position sizing, no overlap handling).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.chan.engine import build_level_snapshot

HORIZONS = (5, 10, 20)
TRACKED_STATES = ("candidate", "confirmed")


@dataclass
class PointEvent:
    symbol: str
    timeframe: str
    evidence_id: str
    kind: str
    side: str
    basis: str
    state: str
    point_dt: str
    seen_bar: int
    seen_dt: str
    entry: float
    invalidation_price: float
    bars_after_point: int
    forward_return: Dict[int, Optional[float]] = field(default_factory=dict)
    invalidated_within: Dict[int, Optional[bool]] = field(default_factory=dict)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if "volume" not in out:
        out["volume"] = 0.0
    return out


def replay_points(
    frame: pd.DataFrame,
    timeframe: str,
    symbol: str,
    *,
    warmup: int = 120,
    states: Sequence[str] = TRACKED_STATES,
) -> List[PointEvent]:
    """Walk the frame bar by bar and record the first sighting of each point per state."""
    data = _prepare(frame)
    events: List[PointEvent] = []
    seen: set = set()
    for t in range(max(warmup, 30) - 1, len(data)):
        level = build_level_snapshot(data.iloc[: t + 1], timeframe, symbol)
        for point in level.get("buy_sell_points") or []:
            state = point.get("state")
            if state not in states or not point.get("still_valid"):
                continue
            key = (point["evidence_id"], state)
            if key in seen:
                continue
            seen.add(key)
            events.append(PointEvent(
                symbol=symbol,
                timeframe=timeframe,
                evidence_id=point["evidence_id"],
                kind=point["type"],
                side=point["side"],
                basis=point.get("basis", ""),
                state=state,
                point_dt=point["dt"],
                seen_bar=t,
                seen_dt=str(data["date"].iloc[t]),
                entry=float(data["close"].iloc[t]),
                invalidation_price=float(point["invalidation_price"]),
                bars_after_point=int(point.get("bars_since", 0)),
            ))
    _score(events, data)
    return events


def _score(events: Iterable[PointEvent], data: pd.DataFrame) -> None:
    closes, highs, lows = data["close"].to_numpy(), data["high"].to_numpy(), data["low"].to_numpy()
    last = len(data) - 1
    for ev in events:
        sign = 1.0 if ev.side == "buy" else -1.0
        for h in HORIZONS:
            end = ev.seen_bar + h
            if end > last:
                ev.forward_return[h] = None
                ev.invalidated_within[h] = None
                continue
            ev.forward_return[h] = sign * (closes[end] - ev.entry) / ev.entry
            window = slice(ev.seen_bar + 1, end + 1)
            if ev.side == "buy":
                ev.invalidated_within[h] = bool(lows[window].min() < ev.invalidation_price)
            else:
                ev.invalidated_within[h] = bool(highs[window].max() > ev.invalidation_price)


def drift_baseline(frame: pd.DataFrame, warmup: int = 120) -> Dict[int, Optional[float]]:
    """Mean h-bar forward return over every bar after warm-up (a random-entry long)."""
    closes = _prepare(frame)["close"].to_numpy()
    out: Dict[int, Optional[float]] = {}
    for h in HORIZONS:
        rets = [(closes[t + h] - closes[t]) / closes[t] for t in range(warmup - 1, len(closes) - h)]
        out[h] = mean(rets) if rets else None
    return out


def summarise(events: Sequence[PointEvent], baselines: Dict[str, Dict[int, Optional[float]]]) -> List[Dict]:
    """Aggregate per (side, kind, state): count, mean/median return, hit rate, invalidation rate, excess."""
    groups: Dict[tuple, List[PointEvent]] = {}
    for ev in events:
        groups.setdefault((ev.side, ev.kind, ev.state), []).append(ev)
        groups.setdefault((ev.side, "ALL", ev.state), []).append(ev)
    rows = []
    for (side, kind, state), evs in sorted(groups.items()):
        row = {"side": side, "kind": kind, "state": state, "n": len(evs)}
        for h in HORIZONS:
            scored = [ev for ev in evs if ev.forward_return.get(h) is not None]
            rets = [ev.forward_return[h] for ev in scored]
            if not rets:
                row[f"n_{h}"] = 0
                continue
            sign = 1.0 if side == "buy" else -1.0
            drift = [sign * baselines[ev.symbol][h] for ev in scored if baselines[ev.symbol].get(h) is not None]
            row[f"n_{h}"] = len(rets)
            row[f"mean_{h}"] = round(mean(rets), 4)
            row[f"median_{h}"] = round(median(rets), 4)
            row[f"hit_{h}"] = round(sum(r > 0 for r in rets) / len(rets), 3)
            row[f"invalidated_{h}"] = round(sum(bool(ev.invalidated_within[h]) for ev in scored) / len(scored), 3)
            if drift:
                row[f"excess_{h}"] = round(mean(rets) - mean(drift), 4)
        rows.append(row)
    return rows


def events_to_dicts(events: Sequence[PointEvent]) -> List[Dict]:
    return [asdict(ev) for ev in events]
