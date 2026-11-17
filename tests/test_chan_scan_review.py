# -*- coding: utf-8 -*-
"""Scoring of earlier --scan picks (src/chan/scan_review.py). No network."""

import json

import numpy as np
import pandas as pd
import pytest

from src.chan.scan_review import (
    load_scan_payloads,
    render_review_markdown,
    review_past_scans,
    score_event,
    unique_events,
)


def _frame(closes, start="2026-11-02"):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"date": pd.bdate_range(start, periods=len(closes)), "open": closes,
                         "high": closes + 0.5, "low": closes - 0.5, "close": closes, "volume": 1.0})


def _point(code, kind, state="confirmed", invalidation=95.0, evidence=None):
    side = "buy" if kind.endswith("B") else "sell"
    return {"code": code, "evidence_id": evidence or f"D.{kind}.2026-10-30", "type": kind, "side": side,
            "state": state, "invalidation_price": invalidation}


def _write(tmp_path, stamp, as_of, points, selected):
    path = tmp_path / f"chan_scan_{stamp}.json"
    path.write_text(json.dumps({"as_of": as_of, "points": points, "selected": selected}), encoding="utf-8")
    return path


def test_unique_events_keep_the_first_scan_and_mark_llm_picks(tmp_path):
    _write(tmp_path, "20261103", "2026-11-02", [_point("AAA", "1B"), _point("BBB", "3S", invalidation=110)], ["AAA"])
    _write(tmp_path, "20261104", "2026-11-03", [_point("AAA", "1B")], ["AAA"])  # same point again
    (tmp_path / "chan_scan_20261105.json").write_text("not json", encoding="utf-8")
    events = unique_events(load_scan_payloads(tmp_path))
    assert [(e["code"], e["scan_date"], e["sent_to_llm"]) for e in events] == [
        ("AAA", "2026-11-02", True), ("BBB", "2026-11-02", False)]


def test_score_event_is_sign_adjusted_with_benchmark_excess_and_invalidation():
    stock = _frame([100] + [110] * 25)
    spy = _frame([100] + [102] * 25)
    buy = {**_point("AAA", "1B", invalidation=99.0), "scan_date": "2026-11-02"}
    score = score_event(buy, stock, spy)
    assert score[5]["return"] == pytest.approx(0.10)
    assert score[5]["excess"] == pytest.approx(0.08)
    assert score[5]["invalidated"] is False
    sell = {**_point("AAA", "1S", invalidation=105.0), "scan_date": "2026-11-02"}
    score = score_event(sell, stock, spy)
    assert score[5]["return"] == pytest.approx(-0.10) and score[5]["invalidated"] is True
    late = {**buy, "scan_date": "2026-12-03"}
    assert score_event(late, stock, spy)[5] is None  # scan date has fewer than 5 bars after it


def test_review_groups_and_renders(tmp_path):
    today = _write(tmp_path, "20261117", "2026-11-16", [_point("ZZZ", "1B")], ["ZZZ"])
    _write(tmp_path, "20261103", "2026-11-02",
           [_point("AAA", "1B"), _point("CCC", "2B", state="forming"), _point("BBB", "3S", invalidation=120)], ["AAA"])
    frames = {"AAA": _frame([100] + [110] * 25), "BBB": _frame([100] + [90] * 25),
              "CCC": _frame([100] * 26), "SPY": _frame([100] * 26)}
    review = review_past_scans(tmp_path, frames, exclude=[today])
    assert review["events"] == 3 and review["scans"] == 1
    summary = review["summary"]
    assert summary["buy_sent_to_llm"][5]["mean_excess_pct"] == pytest.approx(10.0)
    assert summary["sell"][5]["mean_return_pct"] == pytest.approx(10.0)
    assert summary["buy_forming"][5]["n"] == 1 and summary["buy_not_sent"][5]["n"] == 0
    text = render_review_markdown(review, "en")
    assert "## Past scan picks" in text and "| Buy, sent to LLM | 5 | 1 | 10.0 | 10.0 | 100.0 | 0.0 |" in text
    assert "历史扫描结果回看" in render_review_markdown(review, "zh")
