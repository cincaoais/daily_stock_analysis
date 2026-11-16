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
_PROMPT_MIN_RECENT_BIS = 3  # first shedding step keeps this many strokes per level
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


def _trim_recent_bis(level: Dict[str, Any]) -> None:
    if "recent_bis" in level:
        level["recent_bis"] = level["recent_bis"][-_PROMPT_MIN_RECENT_BIS:]


def _drop_invalidated_points(level: Dict[str, Any]) -> None:
    if level.get("buy_sell_points"):
        level["buy_sell_points"] = [p for p in level["buy_sell_points"] if p.get("still_valid")]


# Applied in order until the JSON fits; current structure and live points go last.
_SHED_STEPS = (
    _trim_recent_bis,
    lambda level: level.pop("recent_bis", None),
    lambda level: level.pop("last_confirmed_bi", None),
    _drop_invalidated_points,
)


def chan_snapshot_json(snapshot: Dict[str, Any]) -> str:
    """Serialise the compact snapshot, shedding detail until it fits the prompt budget."""
    compact = compact_chan_snapshot(snapshot)
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    for step in _SHED_STEPS:
        if len(text) <= _PROMPT_MAX_CHARS:
            break
        for level in compact.get("levels", {}).values():
            step(level)
        text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return text


_SECTION_TEXT = {
    "zh": (
        "## 🧭 缠论结构快照（代码计算，唯一结构事实来源）",
        "- 下列 JSON 由程序根据 K 线计算，级别：W=周线，D=日线，60M/30M=分钟线。笔、中枢、背驰、买卖点一律以它为准，"
        "不要自行重算或凭记忆补充；`segment` 为 `not_supported` 表示线段未计算，不得描述线段。`as_of` 是最新 K 线时间。\n"
        "- `level_roles`：`direction` 级别定方向，`operating` 级别给买卖点，`confirmation` 级别定时机；背驰力度只在同一级别内比较。\n"
        "- `state`：`forming`=笔未完成，只能观察；`candidate`=位于最新完成笔上，仍可能被重画，最多小仓试探并写明待确认；"
        "`confirmed`=其后已有新笔，结构固定；`invalidated`=已击穿失效价，仅作历史参考。不得在文字中升级状态。\n"
        "- `invalidation_price` 是失效价，止损锚定于此，不用固定百分比；`basis` 为 `range_divergence` 的一买/一卖是盘整背驰，不给方向。"
        "背驰只是预警，单独不能构成买卖。\n"
        "- `status` 不是 `available` 的级别必须说明数据局限并降低置信度。引用结构事实时带上 `evidence_id`（例如“日线三买 D.3B.2026-10-20”）。",
    ),
    "en": (
        "## 🧭 Chan Structure Snapshot (computed in code; the only source of structural facts)",
        "- The JSON below is computed from K-lines. Levels: W=weekly, D=daily, 60M/30M=intraday. Take strokes, "
        "pivots, divergence and buy/sell points from it only; do not recompute or recall them. "
        "`segment` = `not_supported` means segments are not computed, so do not describe them. `as_of` is the latest bar.\n"
        "- `level_roles`: the `direction` level sets the bias, the `operating` level supplies buy/sell points, the "
        "`confirmation` level times entries. Compare divergence strength only within one level.\n"
        "- `state`: `forming` = unfinished stroke, watch only; `candidate` = on the newest finished stroke and can "
        "still be redrawn, small trial at most and say it awaits confirmation; `confirmed` = a newer stroke exists, "
        "structure fixed; `invalidated` = price broke the invalidation price, history only. Never upgrade a state in prose.\n"
        "- `invalidation_price` is where a point fails; anchor stops to it, never to a fixed percentage. A 1B/1S with "
        "`basis` = `range_divergence` is a consolidation divergence and gives no direction. Divergence is a warning, "
        "never a trade signal on its own.\n"
        "- For any level whose `status` is not `available`, state the limitation and lower confidence. Cite the "
        "`evidence_id` for structural facts (e.g. \"daily 3B D.3B.2026-10-20\").",
    ),
}


def format_chan_prompt_section(snapshot: Optional[Dict[str, Any]], language: str = "zh") -> str:
    """Render the snapshot as a prompt section; empty string when there is nothing to show."""
    if not snapshot or not snapshot.get("levels"):
        return ""
    title, rules = _SECTION_TEXT["zh" if str(language or "zh").lower().startswith("zh") else "en"]
    return "\n%s\n%s\n```json\n%s\n```\n" % (title, rules, chan_snapshot_json(snapshot))
