# -*- coding: utf-8 -*-
"""Point-in-time Chan event study (evals/chan_event_study)."""

import numpy as np
import pandas as pd
import pytest

from evals.chan_event_study.study import PointEvent, _score, drift_baseline, replay_points, summarise


def _frame(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(closes), freq="B"),
        "open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes, "volume": 1.0,
    })


def _event(side, entry_bar, entry, invalidation, state="confirmed"):
    return PointEvent(symbol="T", timeframe="1d", evidence_id="D.3B.x", kind="3B" if side == "buy" else "3S",
                      side=side, basis="pivot_exit", state=state, point_dt="x", seen_bar=entry_bar,
                      seen_dt="x", entry=entry, invalidation_price=invalidation, bars_after_point=1)


def test_scoring_is_sign_adjusted_and_uses_only_later_bars():
    data = _frame([100.0] * 5 + [110.0] * 30)
    buy, sell = _event("buy", 4, 100.0, 99.0), _event("sell", 4, 100.0, 105.0)
    _score([buy, sell], data)
    assert buy.forward_return[5] == pytest.approx(0.10)
    assert sell.forward_return[5] == pytest.approx(-0.10)
    assert buy.invalidated_within[5] is False  # lows stay at 109.5
    assert sell.invalidated_within[5] is True  # highs reach 110.5 > 105
    late = _event("buy", 30, 110.0, 100.0)
    _score([late], data)
    assert late.forward_return[5] is None  # not enough future bars


def test_summary_reports_excess_over_drift():
    data = _frame(np.linspace(100, 200, 200))
    ev = _event("buy", 150, float(data["close"].iloc[150]), 1.0)
    _score([ev], data)
    rows = summarise([ev], {"T": drift_baseline(data, warmup=120)})
    row = next(r for r in rows if r["kind"] == "3B")
    assert row["n"] == 1 and row["hit_5"] == 1.0
    assert "excess_5" in row


def test_replay_never_sees_a_point_before_its_own_bar():
    pytest.importorskip("czsc")
    turns = [100, 112, 104, 114, 106, 130, 122, 134, 126, 150, 142, 154, 146, 170, 164, 176, 168]
    closes = [float(turns[0])]
    for a, b in zip(turns, turns[1:]):
        closes.extend(np.linspace(a, b, 13)[1:])
    frame = _frame(closes)
    events = replay_points(frame, "1d", "T", warmup=60)
    assert events, "synthetic uptrend should produce at least one 3B"
    for ev in events:
        assert pd.Timestamp(ev.point_dt) <= pd.Timestamp(ev.seen_dt)
        assert ev.state in ("candidate", "confirmed")
    states = {(ev.evidence_id, ev.state): ev.seen_bar for ev in events}
    for (eid, state), bar in states.items():
        if state == "confirmed" and (eid, "candidate") in states:
            assert states[(eid, "candidate")] <= bar
