# -*- coding: utf-8 -*-
"""Tests for the Chan snapshot service and its prompt wiring (src/chan/service.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from data_provider.base import DataFetchError
from src.chan import service
from src.chan.service import (
    chan_snapshot_json,
    compact_chan_snapshot,
    format_chan_prompt_section,
    load_chan_snapshot,
    parse_timeframes,
)

TURNS = [100, 112, 104, 114, 106, 130, 122, 134, 126, 150, 142, 154, 146, 170, 164]


def _frame(freq="B", start="2025-01-01", leg_bars=12):
    closes = [float(TURNS[0])]
    for a, b in zip(TURNS, TURNS[1:]):
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


class _FakeManager:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def get_kline_data(self, stock_code, timeframe="1d", start_date=None, end_date=None, days=365):
        self.calls.append((stock_code, timeframe, days))
        if timeframe in self.failing:
            raise DataFetchError("所有数据源获取 %s %s K线失败:\n[X] boom" % (stock_code, timeframe))
        if timeframe in ("60m", "30m"):
            return _frame(freq="30min", start="2026-01-05 09:30"), "FakeIntraday"
        return _frame(), "FakeDaily"


def test_parse_timeframes_keeps_supported_in_level_order():
    assert parse_timeframes(None) == ("1w", "1d", "60m", "30m")
    assert parse_timeframes("30m, 1D ,5m") == ("1d", "30m")
    assert parse_timeframes(["60m"]) == ("60m",)
    assert parse_timeframes("bogus") == ("1w", "1d", "60m", "30m")


def test_load_chan_snapshot_fetches_each_timeframe_and_records_sources():
    pytest.importorskip("czsc")
    manager = _FakeManager()
    snapshot = load_chan_snapshot(manager, "AAPL", market="us", name="Apple", timeframes="1d,30m")

    assert [call[1] for call in manager.calls] == ["1d", "30m"]
    assert all(call[0] == "AAPL" and call[2] > 0 for call in manager.calls)
    assert list(snapshot["levels"]) == ["D", "30M"]
    assert snapshot["levels"]["D"]["source"] == "FakeDaily"
    assert snapshot["levels"]["30M"]["source"] == "FakeIntraday"
    assert snapshot["market"] == "us" and snapshot["name"] == "Apple"
    assert snapshot["data_quality"]["status"] == "available"


def test_failed_timeframe_becomes_missing_level_with_reason():
    pytest.importorskip("czsc")
    snapshot = load_chan_snapshot(_FakeManager(failing={"1w"}), "AAPL", timeframes="1w,1d")

    weekly = snapshot["levels"]["W"]
    assert weekly["status"] == "missing"
    assert weekly["warnings"][0].startswith("fetch_failed: 所有数据源获取 AAPL 1w K线失败")
    assert "\n" not in weekly["warnings"][0]
    assert snapshot["levels"]["D"]["status"] == "available"
    assert snapshot["data_quality"]["status"] == "partial"
    assert snapshot["data_quality"]["unusable_levels"] == ["W"]


def test_all_timeframes_failing_yields_missing_snapshot_and_empty_section():
    snapshot = load_chan_snapshot(_FakeManager(failing={"1w", "1d", "60m", "30m"}), "AAPL")
    assert snapshot["data_quality"]["status"] == "missing"
    assert format_chan_prompt_section(None) == ""
    assert format_chan_prompt_section({"levels": {}}) == ""


def _point(dt, still_valid):
    return {"evidence_id": "D.3B.%s" % dt, "type": "3B", "dt": dt, "still_valid": still_valid}


def test_compact_snapshot_trims_strokes_and_stale_points_without_mutating():
    level = {
        "status": "available",
        "recent_bis": [{"id": i} for i in range(7)],
        "buy_sell_points": [
            _point("2026-01-01", False), _point("2026-02-01", False), _point("2026-03-01", False),
            _point("2026-04-01", True), _point("2026-05-01", False),
        ],
    }
    snapshot = {"levels": {"D": level}}
    compact = compact_chan_snapshot(snapshot)

    assert [b["id"] for b in compact["levels"]["D"]["recent_bis"]] == [2, 3, 4, 5, 6]
    assert [p["dt"] for p in compact["levels"]["D"]["buy_sell_points"]] == [
        "2026-03-01", "2026-04-01", "2026-05-01",
    ]
    assert len(level["recent_bis"]) == 7 and len(level["buy_sell_points"]) == 5


def test_prompt_json_sheds_detail_to_fit_budget(monkeypatch):
    stroke = {"id": "D.bi.x", "note": "x" * 400}
    snapshot = {"levels": {key: {"recent_bis": [stroke] * 5, "last_confirmed_bi": stroke, "trend_type": "up"}
                           for key in ("W", "D", "60M", "30M")}}
    monkeypatch.setattr(service, "_PROMPT_MAX_CHARS", 3000)
    payload = json.loads(chan_snapshot_json(snapshot))
    assert all("recent_bis" not in level for level in payload["levels"].values())
    assert all(level["trend_type"] == "up" for level in payload["levels"].values())


def test_prompt_section_language_and_valid_json():
    pytest.importorskip("czsc")
    snapshot = load_chan_snapshot(_FakeManager(), "AAPL", timeframes="1d")
    zh = format_chan_prompt_section(snapshot, "zh")
    en = format_chan_prompt_section(snapshot, "en")

    assert "缠论结构快照" in zh and "失效价" in zh
    assert "Chan Structure Snapshot" in en and "invalidation_price" in en
    assert format_chan_prompt_section(snapshot, "ko").startswith(en[:40])
    body = en.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    assert json.loads(body)["schema_version"] == "chan_snapshot/1.0"
