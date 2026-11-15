# -*- coding: utf-8 -*-
"""Fetch multi-timeframe K-lines and turn them into a prompt-ready Chan snapshot."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.chan.engine import LEVEL_KEYS, SUPPORTED_TIMEFRAMES, build_chan_snapshot

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES: Tuple[str, ...] = SUPPORTED_TIMEFRAMES

# ``days`` passed to DataFetcherManager.get_kline_data per timeframe: enough bars
# for several 中枢 on each level without paging far into history.
_LOOKBACK_DAYS = {"1w": 900, "1d": 365, "60m": 120, "30m": 45}

_PROMPT_RECENT_BIS = 5
_PROMPT_STALE_POINTS = 2  # invalidated buy/sell points kept per level, newest first
_PROMPT_MAX_CHARS = 7500  # stays under the multi-agent 8000-char pre-fetched cap


def parse_timeframes(value: Any) -> Tuple[str, ...]:
    """Parse a comma-separated timeframe list, keeping only supported ones in level order."""
    if value is None:
        return DEFAULT_TIMEFRAMES
    items: Iterable[str] = value.split(",") if isinstance(value, str) else value
    wanted = {str(item).strip().lower() for item in items if str(item).strip()}
    selected = tuple(tf for tf in SUPPORTED_TIMEFRAMES if tf in wanted)
    return selected or DEFAULT_TIMEFRAMES


def load_chan_snapshot(
    fetcher_manager: Any,
    stock_code: str,
    *,
    market: Optional[str] = None,
    name: Optional[str] = None,
    timeframes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Fetch each timeframe through ``fetcher_manager.get_kline_data`` and build the snapshot.

    A timeframe that cannot be fetched is reported as a ``missing`` level with the
    fetch error in its warnings; this function does not raise for data problems.
    """
    frames: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    fetch_errors: Dict[str, str] = {}
    for timeframe in parse_timeframes(timeframes):
        try:
            frame, source = fetcher_manager.get_kline_data(
                stock_code, timeframe=timeframe, days=_LOOKBACK_DAYS[timeframe]
            )
            frames[timeframe] = frame
            sources[timeframe] = source
        except Exception as exc:  # noqa: BLE001 - one failed level must not drop the others
            logger.warning("[chan] %s %s K线获取失败: %s", stock_code, timeframe, exc)
            frames[timeframe] = None
            fetch_errors[timeframe] = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__

    snapshot = build_chan_snapshot(frames, stock_code, market=market, name=name, sources=sources)
    for timeframe, error in fetch_errors.items():
        level_key = LEVEL_KEYS[timeframe]
        warning = "fetch_failed: %s" % error
        snapshot["levels"][level_key]["warnings"].insert(0, warning)
        snapshot["data_quality"]["warnings"].insert(0, "%s: %s" % (level_key, warning))
    return snapshot


def compact_chan_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a smaller copy for prompts: fewer strokes and only relevant buy/sell points."""
    compact = copy.deepcopy(snapshot)
    for level in compact.get("levels", {}).values():
        if "recent_bis" in level:
            level["recent_bis"] = level["recent_bis"][-_PROMPT_RECENT_BIS:]
        points: List[Dict[str, Any]] = level.get("buy_sell_points") or []
        if points:
            valid = [p for p in points if p.get("still_valid")]
            stale = [p for p in points if not p.get("still_valid")][-_PROMPT_STALE_POINTS:]
            level["buy_sell_points"] = sorted(stale + valid, key=lambda p: p["dt"])
    return compact


def chan_snapshot_json(snapshot: Dict[str, Any]) -> str:
    """Serialise the compact snapshot, shedding detail until it fits the prompt budget."""
    compact = compact_chan_snapshot(snapshot)
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    for drop_key in ("recent_bis", "last_confirmed_bi"):
        if len(text) <= _PROMPT_MAX_CHARS:
            break
        for level in compact.get("levels", {}).values():
            level.pop(drop_key, None)
        text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return text


_SECTION_TEXT = {
    "zh": (
        "## 🧭 缠论结构快照（代码计算，唯一结构事实来源）",
        "- 下列 JSON 由程序根据 K 线计算，级别：W=周线，D=日线，60M/30M=分钟线。笔、中枢、背驰、买卖点一律以它为准，"
        "不要自行重算或凭记忆补充；`segment` 为 `not_supported` 表示线段未计算，不得描述线段。\n"
        "- `confirmed=false` 表示尚在形成中，只能表述为“形成中/待确认”；`still_valid=false` 的买卖点已被击穿，只能作为历史参考。\n"
        "- `invalidation_price` 是该买卖点的失效价，据此设置止损；`status` 不是 `available` 的级别必须说明数据局限并降低置信度。\n"
        "- 引用结构事实时请带上 `evidence_id` 或级别（例如“日线三买 D.3B.2026-10-20”）。",
    ),
    "en": (
        "## 🧭 Chan Structure Snapshot (computed in code; the only source of structural facts)",
        "- The JSON below is computed from K-lines. Levels: W=weekly, D=daily, 60M/30M=intraday. Take strokes, "
        "pivots, divergence and buy/sell points from it only; do not recompute or recall them. "
        "`segment` = `not_supported` means segments are not computed, so do not describe them.\n"
        "- `confirmed=false` means still forming: describe it as forming/unconfirmed. Points with `still_valid=false` "
        "have been broken and are history only.\n"
        "- `invalidation_price` is where a buy/sell point fails; anchor stop-losses to it. For any level whose "
        "`status` is not `available`, state the data limitation and lower confidence.\n"
        "- Cite the `evidence_id` or level when using a structural fact (e.g. \"daily 3B D.3B.2026-10-20\").",
    ),
}


def format_chan_prompt_section(snapshot: Optional[Dict[str, Any]], language: str = "zh") -> str:
    """Render the snapshot as a prompt section; empty string when there is nothing to show."""
    if not snapshot or not snapshot.get("levels"):
        return ""
    title, rules = _SECTION_TEXT["zh" if str(language or "zh").lower().startswith("zh") else "en"]
    return "\n%s\n%s\n```json\n%s\n```\n" % (title, rules, chan_snapshot_json(snapshot))
