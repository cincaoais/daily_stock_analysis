# -*- coding: utf-8 -*-
"""Frozen Chan scenarios for the prompt A/B harness.

Every case starts from a real AAPL snapshot (``fixtures/aapl_snapshot.json``,
built from Yahoo data on 2026-11-13) and edits only what the scenario needs, so
the model always sees realistic structure. ``expect`` holds what a correct
answer must satisfy; see ``metrics.score_result``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aapl_snapshot.json"

NEUTRAL_NEWS = "2026-11-12 Apple announces date of next product event (no financial details)."
BULLISH_NEWS = ("2026-11-12 Apple beats quarterly revenue estimates and raises guidance; several brokers "
                "lift price targets.")


def _base() -> Dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _clear_valid_points(snapshot: Dict[str, Any]) -> None:
    for level in snapshot["levels"].values():
        level["buy_sell_points"] = [p for p in level.get("buy_sell_points") or [] if not p.get("still_valid")]


def _daily_point(snapshot, kind, state, *, basis="pivot_exit", offset=-0.03) -> Dict[str, Any]:
    daily = snapshot["levels"]["D"]
    close = float(daily["last_close"])
    side = "buy" if kind.endswith("B") else "sell"
    price = round(close * (1 + offset), 2)
    invalidation = round(price * (0.98 if side == "buy" else 1.02), 2)
    dt = daily["last_bar"]
    return {
        "evidence_id": f"D.{kind}.{dt}", "type": kind, "side": side, "level": "D", "dt": dt,
        "price": price, "invalidation_price": invalidation, "basis": basis, "state": state,
        "still_valid": True, "bars_since": 3,
    }


def _context(snapshot: Dict[str, Any], news: str) -> Dict[str, Any]:
    daily = snapshot["levels"]["D"]
    return {
        "code": "AAPL",
        "stock_name": "Apple",
        "date": str(daily["last_bar"])[:10],
        "today": {"close": daily["last_close"]},
        "chan_snapshot": snapshot,
        "_news": news,
    }


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    real = _base()
    cases.append({"id": "aapl_real", "context": _context(real, NEUTRAL_NEWS),
                  "expect": {"must_mention": ["divergence|背驰"]}})

    none = _base()
    _clear_valid_points(none)
    cases.append({"id": "no_valid_points", "context": _context(none, NEUTRAL_NEWS),
                  "expect": {"allowed_decisions": ["hold", "sell"]}})

    forming = _base()
    _clear_valid_points(forming)
    forming["levels"]["D"]["buy_sell_points"].append(_daily_point(forming, "3B", "forming"))
    cases.append({"id": "forming_only", "context": _context(forming, NEUTRAL_NEWS),
                  "expect": {"allowed_decisions": ["hold"]}})

    aligned = _base()
    _clear_valid_points(aligned)
    aligned["levels"]["W"]["trend_type"] = "up"
    aligned["levels"]["W"]["divergence"] = {"present": False, "method": "macd_area"}
    aligned["levels"]["D"]["buy_sell_points"].append(_daily_point(aligned, "3B", "confirmed"))
    cases.append({"id": "confirmed_3b_aligned", "context": _context(aligned, NEUTRAL_NEWS),
                  "expect": {"allowed_decisions": ["buy", "hold"]}})

    range_div = _base()
    _clear_valid_points(range_div)
    range_div["levels"]["D"]["trend_type"] = "consolidation"
    range_div["levels"]["D"]["buy_sell_points"].append(
        _daily_point(range_div, "1B", "confirmed", basis="range_divergence"))
    cases.append({"id": "range_divergence_1b", "context": _context(range_div, NEUTRAL_NEWS),
                  "expect": {"confidence_not_high": True, "must_mention": ["pivot|中枢|range|盘整"]}})

    partial = _base()
    partial["levels"]["D"]["status"] = "partial"
    partial["levels"]["D"]["warnings"] = ["only 90 bars; structure may be incomplete below 120"]
    partial["data_quality"]["status"] = "partial"
    cases.append({"id": "partial_daily_data", "context": _context(partial, NEUTRAL_NEWS),
                  "expect": {"confidence_not_high": True}})

    bearish = _base()
    _clear_valid_points(bearish)
    bearish["levels"]["D"]["buy_sell_points"].append(_daily_point(bearish, "3S", "confirmed", offset=0.02))
    cases.append({"id": "bullish_news_bearish_structure", "context": _context(bearish, BULLISH_NEWS),
                  "expect": {"allowed_decisions": ["hold", "sell"]}})

    return [copy.deepcopy(c) for c in cases]
