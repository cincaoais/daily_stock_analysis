# -*- coding: utf-8 -*-
"""Tests for the Chan structure engine (src/chan)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.chan import engine
from src.chan.engine import (
    Pivot,
    Stroke,
    build_chan_snapshot,
    build_level_snapshot,
    classify_trend,
    detect_trade_points,
    stroke_divergence,
    stroke_state,
)


def _ts(day: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)


def _strokes(points, areas=None, confirmed_last=True):
    """Build alternating strokes from turning points [(day, price), ...]."""
    out = []
    for i in range(len(points) - 1):
        (d0, p0), (d1, p1) = points[i], points[i + 1]
        out.append(Stroke(
            direction="up" if p1 > p0 else "down",
            start_dt=_ts(d0),
            end_dt=_ts(d1),
            high=max(p0, p1),
            low=min(p0, p1),
            macd_area=(areas[i] if areas else abs(p1 - p0)),
            bar_count=d1 - d0 + 1,
            confirmed=confirmed_last or i < len(points) - 2,
        ))
    return out


def _pivot(zg, zd, start_day, formed_day, end_day, last_stroke_index, gg=None, dd=None):
    return Pivot(
        zg=zg, zd=zd, gg=gg if gg is not None else zg, dd=dd if dd is not None else zd,
        start_dt=_ts(start_day), end_dt=_ts(end_day), formed_dt=_ts(formed_day),
        stroke_count=3, last_stroke_index=last_stroke_index,
    )


# -- divergence ---------------------------------------------------------------

def test_divergence_requires_new_extreme_and_smaller_area():
    # down(100->80, area 10), up, down(90->75, area 6): new low + weaker => bottom divergence
    strokes = _strokes([(0, 100), (10, 80), (20, 90), (30, 75)], areas=[10, 5, 6])
    found = stroke_divergence(strokes, 2)
    assert found == {
        "type": "bottom", "strength_ratio": 0.6, "stroke_index": 2,
        "compared_stroke_index": 0, "confirmed": True,
    }


def test_no_divergence_without_new_extreme_or_with_stronger_move():
    higher_low = _strokes([(0, 100), (10, 80), (20, 90), (30, 82)], areas=[10, 5, 6])
    assert stroke_divergence(higher_low, 2) is None
    stronger = _strokes([(0, 100), (10, 80), (20, 90), (30, 75)], areas=[10, 5, 12])
    assert stroke_divergence(stronger, 2) is None
    assert stroke_divergence(stronger, 1) is None  # no earlier same-direction stroke


def test_short_forming_stroke_is_not_judged():
    strokes = _strokes([(0, 100), (10, 80), (20, 90), (22, 75)], areas=[10, 5, 1], confirmed_last=False)
    assert strokes[-1].bar_count < engine.MIN_FORMING_BARS
    assert stroke_divergence(strokes, 2) is None


def test_zero_area_never_counts_as_divergence():
    strokes = _strokes([(0, 100), (10, 80), (20, 90), (30, 75)], areas=[10, 5, 0])
    assert stroke_divergence(strokes, 2) is None


# -- trade points -------------------------------------------------------------

def test_first_and_second_buy_after_downtrend_divergence():
    # Two descending pivots, then a weaker new low (1B), bounce, higher low (2B).
    points = [(0, 100), (10, 90), (20, 96), (30, 88),      # pivot A strokes 0-2
              (40, 70), (50, 76), (60, 68),                 # pivot B strokes 4-6 (approx.)
              (70, 74), (80, 60), (90, 66), (100, 63)]
    areas = [10, 6, 8, 18, 6, 8, 6, 5, 3, 4]
    strokes = _strokes(points, areas=areas)
    pivots = [
        _pivot(zg=96, zd=90, start_day=0, formed_day=30, end_day=30, last_stroke_index=None),
        _pivot(zg=76, zd=70, start_day=40, formed_day=70, end_day=70, last_stroke_index=None),
    ]
    found = {(p.kind, p.stroke_index): p for p in detect_trade_points(strokes, pivots)}

    first = found[("1B", 7)]
    assert first.price == 60 and first.invalidation_price == 60
    assert first.basis == "trend_divergence"
    second = found[("2B", 9)]
    assert second.price == 63 and second.invalidation_price == 60
    assert second.basis == "after_1B"


def test_first_buy_inside_single_pivot_is_range_divergence():
    points = [(0, 100), (10, 90), (20, 96), (30, 88), (40, 94), (50, 84)]
    strokes = _strokes(points, areas=[10, 6, 8, 6, 5])
    pivots = [_pivot(zg=96, zd=90, start_day=0, formed_day=30, end_day=40, last_stroke_index=None)]
    kinds = [(p.kind, p.basis) for p in detect_trade_points(strokes, pivots)]
    assert ("1B", "range_divergence") in kinds


def test_no_first_buy_when_low_stays_inside_pivot():
    points = [(0, 100), (10, 92), (20, 96), (30, 91)]
    strokes = _strokes(points, areas=[10, 6, 4])
    pivots = [_pivot(zg=96, zd=90, start_day=0, formed_day=20, end_day=30, last_stroke_index=None)]
    assert detect_trade_points(strokes, pivots) == []


def test_third_buy_and_third_sell_on_pivot_exit():
    up_exit = _strokes([(0, 90), (10, 96), (20, 91), (30, 110), (40, 100)])
    pivot = _pivot(zg=96, zd=91, start_day=0, formed_day=20, end_day=30, last_stroke_index=2)
    (point,) = detect_trade_points(up_exit, [pivot])
    assert (point.kind, point.stroke_index, point.price, point.invalidation_price) == ("3B", 3, 100, 96)

    back_inside = _strokes([(0, 90), (10, 96), (20, 91), (30, 110), (40, 95)])
    assert detect_trade_points(back_inside, [pivot]) == []

    down_exit = _strokes([(0, 96), (10, 90), (20, 95), (30, 78), (40, 86)])
    pivot_down = _pivot(zg=95, zd=90, start_day=0, formed_day=20, end_day=30, last_stroke_index=2)
    (sell,) = detect_trade_points(down_exit, [pivot_down])
    assert (sell.kind, sell.price, sell.invalidation_price) == ("3S", 86, 90)


def test_unconfirmed_stroke_gives_unconfirmed_point():
    strokes = _strokes([(0, 90), (10, 96), (20, 91), (30, 110), (40, 100)], confirmed_last=False)
    pivot = _pivot(zg=96, zd=91, start_day=0, formed_day=20, end_day=30, last_stroke_index=2)
    (point,) = detect_trade_points(strokes, [pivot])
    assert point.kind == "3B" and point.confirmed is False


def test_stroke_state_marks_newest_finished_stroke_as_candidate():
    finished = _strokes([(0, 90), (10, 96), (20, 91), (30, 110)])
    assert [stroke_state(finished, i) for i in range(3)] == ["confirmed", "confirmed", "candidate"]
    forming = _strokes([(0, 90), (10, 96), (20, 91), (30, 110)], confirmed_last=False)
    assert [stroke_state(forming, i) for i in range(3)] == ["confirmed", "candidate", "forming"]


def test_level_roles_pick_daily_as_operating_level():
    ok = {"status": "available"}
    assert engine._level_roles({"W": ok, "D": ok, "60M": ok, "30M": ok}) == {
        "direction": "W", "operating": "D", "confirmation": "60M"}
    assert engine._level_roles({"W": ok, "D": {"status": "error"}, "60M": ok}) == {
        "direction": None, "operating": "W", "confirmation": "60M"}
    assert engine._level_roles({"D": {"status": "missing"}}) == {
        "direction": None, "operating": None, "confirmation": None}


def test_classify_trend():
    low = _pivot(zg=96, zd=90, start_day=0, formed_day=10, end_day=20, last_stroke_index=None)
    high = _pivot(zg=120, zd=110, start_day=30, formed_day=40, end_day=50, last_stroke_index=None)
    overlap = _pivot(zg=100, zd=94, start_day=30, formed_day=40, end_day=50, last_stroke_index=None)
    assert classify_trend([]) == "undetermined"
    assert classify_trend([low]) == "consolidation"
    assert classify_trend([low, high]) == "up"
    assert classify_trend([high, low]) == "down"
    assert classify_trend([low, overlap]) == "consolidation"


# -- snapshot assembly --------------------------------------------------------

def _zigzag_frame(turns, leg_bars=12, start="2025-01-01", freq="B"):
    """OHLCV frame walking linearly through ``turns`` with ``leg_bars`` bars per leg."""
    closes = [float(turns[0])]
    for a, b in zip(turns, turns[1:]):
        closes.extend(np.linspace(a, b, leg_bars + 1)[1:])
    closes = np.array(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(closes), freq=freq),
        "open": opens,
        "high": np.maximum(opens, closes) + 0.2,
        "low": np.minimum(opens, closes) - 0.2,
        "close": closes,
        "volume": 1000.0,
    })


UPTREND_TURNS = [100, 112, 104, 114, 106, 130, 122, 134, 126, 150, 142, 154, 146, 170, 164]


def test_level_snapshot_on_synthetic_uptrend():
    pytest.importorskip("czsc")
    frame = _zigzag_frame(UPTREND_TURNS)
    level = build_level_snapshot(frame, "1d", "TEST")

    assert level["status"] == "available"
    assert level["bar_count"] == len(frame)
    assert level["segment"] == {"status": "not_supported"}
    assert level["trend_type"] == "up"

    bis = level["recent_bis"]
    assert len(bis) >= 4
    assert all(a["direction"] != b["direction"] for a, b in zip(bis, bis[1:]))
    assert all(b["state"] in ("confirmed", "candidate") for b in bis[:-1])
    assert [b["state"] for b in bis].count("candidate") == 1
    assert level["last_confirmed_bi"]["state"] == "candidate"
    for point in level["buy_sell_points"]:
        assert point["state"] in {"forming", "candidate", "confirmed", "invalidated"}
        assert (point["state"] == "invalidated") == (not point["still_valid"])
        assert "confirmed" not in point
    for pivot in level["zhongshu"]:
        assert pivot["gg"] >= pivot["zg"] > pivot["zd"] >= pivot["dd"]
    for point in level["buy_sell_points"]:
        assert point["type"] in {"1B", "2B", "3B", "1S", "2S", "3S"}
        assert point["evidence_id"].startswith("D.%s." % point["type"])
    assert any(p["type"] == "3B" for p in level["buy_sell_points"])
    assert level["key_levels"]["support"] == sorted(level["key_levels"]["support"], reverse=True)
    json.dumps(level)  # must be JSON-serialisable


def test_snapshot_is_deterministic_and_accepts_unsorted_duplicated_input():
    pytest.importorskip("czsc")
    frame = _zigzag_frame(UPTREND_TURNS)
    shuffled = pd.concat([frame, frame.tail(5)]).sample(frac=1.0, random_state=7)
    assert build_level_snapshot(shuffled, "1d", "TEST") == build_level_snapshot(frame, "1d", "TEST")


def test_multi_level_snapshot_shape():
    pytest.importorskip("czsc")
    daily = _zigzag_frame(UPTREND_TURNS)
    intraday = _zigzag_frame(UPTREND_TURNS, start="2026-01-05 09:30", freq="30min")
    snapshot = build_chan_snapshot(
        {"1d": daily, "30m": intraday, "1w": daily.head(10), "5m": daily},
        "TEST", market="US", sources={"1d": "FakeFetcher"},
    )
    assert snapshot["schema_version"] == "chan_snapshot/1.1"
    assert list(snapshot["levels"]) == ["W", "D", "30M"]  # coarse to fine; unknown key ignored
    # W is missing, so D operates with no direction level and 30M confirms.
    assert snapshot["level_roles"] == {"direction": None, "operating": "D", "confirmation": "30M"}
    assert snapshot["as_of"] == max(lv["last_bar"] for lv in snapshot["levels"].values() if lv.get("last_bar"))
    assert snapshot["levels"]["D"]["source"] == "FakeFetcher"
    assert snapshot["levels"]["W"]["status"] == "missing"
    assert snapshot["data_quality"]["status"] == "partial"
    assert snapshot["data_quality"]["unusable_levels"] == ["W"]
    assert snapshot["data_quality"]["engine"] == "czsc"
    assert ":" in snapshot["levels"]["30M"]["last_bar"]  # intraday keeps the bar time
    assert ":" not in snapshot["levels"]["D"]["last_bar"]
    assert snapshot["cross_level"]["bi_directions"].keys() == {"D", "30M"}
    json.dumps(snapshot)


def test_bad_input_is_reported_not_raised(monkeypatch):
    assert build_level_snapshot(None, "1d")["status"] == "missing"
    assert build_level_snapshot(pd.DataFrame({"date": [], "close": []}), "1d")["status"] == "missing"
    no_ohlc = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40), "close": 1.0})
    assert build_level_snapshot(no_ohlc, "1d")["status"] == "error"
    short = _zigzag_frame([100, 110], leg_bars=10)
    assert build_level_snapshot(short, "1d")["status"] == "missing"
    with pytest.raises(ValueError):
        build_level_snapshot(short, "5m")

    def _missing():
        raise ImportError("No module named 'czsc'")

    monkeypatch.setattr(engine, "_load_czsc", _missing)
    level = build_level_snapshot(_zigzag_frame(UPTREND_TURNS), "1d")
    assert level["status"] == "engine_unavailable"
    assert "czsc is not installed" in level["warnings"][0]
    assert build_chan_snapshot({"1d": _zigzag_frame(UPTREND_TURNS)}, "X")["data_quality"]["status"] == "missing"
