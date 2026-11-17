# -*- coding: utf-8 -*-
"""How did earlier ``--scan`` picks do? Scores saved scan points against later daily bars.

Each point is scored once, from the first scan that reported it (consecutive scans
repeat a point while it stays recent). Entry is the close on that scan's data date,
because that is the first price a user could act on. Returns are sign-adjusted
(sell points gain when price falls) and compared with SPY over the same window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS: Tuple[int, ...] = (5, 10, 20)
BENCHMARK = "SPY"
MIN_SAMPLE = 30  # below this the summary says the sample is too small to judge


def load_scan_payloads(report_dir: Path, exclude: Iterable[Path] = ()) -> List[Dict[str, Any]]:
    """Saved scan JSON files, oldest first; unreadable files are skipped."""
    skip = {Path(p).resolve() for p in exclude}
    payloads = []
    for path in sorted(report_dir.glob("chan_scan_*.json")):
        if path.resolve() in skip:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[chan-scan] 跳过无法读取的扫描记录 %s: %s", path.name, exc)
            continue
        if payload.get("as_of") and isinstance(payload.get("points"), list):
            payloads.append(payload)
    payloads.sort(key=lambda p: str(p["as_of"]))
    return payloads


def unique_events(payloads: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One event per (code, evidence_id), taken from the earliest scan that saw it."""
    seen = set()
    events = []
    for payload in payloads:
        selected = set(payload.get("selected") or [])
        scan_date = str(payload["as_of"])[:10]
        for point in payload["points"]:
            key = (point.get("code"), point.get("evidence_id"))
            if None in key or key in seen:
                continue
            seen.add(key)
            events.append({**point, "scan_date": scan_date, "sent_to_llm": point["code"] in selected
                           and point.get("side") == "buy"})
    return events


def _bar_index(frame: pd.DataFrame, date: str) -> Optional[int]:
    dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
    return dates.index(date) if date in dates else None


def score_event(event: Mapping[str, Any], frame: Optional[pd.DataFrame],
                benchmark: Optional[pd.DataFrame], horizons: Sequence[int] = HORIZONS) -> Dict[str, Any]:
    """Forward return, excess over the benchmark and invalidation flag per horizon (None = not enough bars)."""
    out: Dict[str, Any] = {h: None for h in horizons}
    if frame is None or frame.empty:
        return out
    start = _bar_index(frame, event["scan_date"])
    bench_start = _bar_index(benchmark, event["scan_date"]) if benchmark is not None else None
    if start is None:
        return out
    entry = float(frame["close"].iloc[start])
    sign = 1.0 if event.get("side") == "buy" else -1.0
    invalidation = event.get("invalidation_price")
    for h in horizons:
        end = start + h
        if end >= len(frame) or entry <= 0:
            continue
        ret = sign * (float(frame["close"].iloc[end]) / entry - 1.0)
        bench = None
        if bench_start is not None and bench_start + h < len(benchmark):
            b0 = float(benchmark["close"].iloc[bench_start])
            bench = sign * (float(benchmark["close"].iloc[bench_start + h]) / b0 - 1.0) if b0 > 0 else None
        window = frame.iloc[start + 1:end + 1]
        if invalidation is None:
            hit = None
        elif sign > 0:
            hit = bool(window["low"].min() < float(invalidation))
        else:
            hit = bool(window["high"].max() > float(invalidation))
        out[h] = {"return": ret, "excess": None if bench is None else ret - bench, "invalidated": hit}
    return out


def _group(event: Mapping[str, Any]) -> str:
    if event.get("side") == "sell":
        return "sell"
    if event.get("state") == "forming":
        return "buy_forming"
    return "buy_sent_to_llm" if event.get("sent_to_llm") else "buy_not_sent"


GROUP_ORDER = ("buy_sent_to_llm", "buy_not_sent", "buy_forming", "sell")


def summarise(events: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[int, Any]],
              horizons: Sequence[int] = HORIZONS) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for group in GROUP_ORDER:
        rows = [s for e, s in zip(events, scores) if _group(e) == group]
        entry: Dict[str, Any] = {"events": len(rows)}
        for h in horizons:
            scored = [r[h] for r in rows if r.get(h)]
            excess = [r["excess"] for r in scored if r["excess"] is not None]
            hits = [r["invalidated"] for r in scored if r["invalidated"] is not None]
            entry[h] = {
                "n": len(scored),
                "mean_return_pct": round(100 * sum(r["return"] for r in scored) / len(scored), 2) if scored else None,
                "mean_excess_pct": round(100 * sum(excess) / len(excess), 2) if excess else None,
                "beat_benchmark_pct": round(100 * sum(x > 0 for x in excess) / len(excess), 1) if excess else None,
                "invalidated_pct": round(100 * sum(hits) / len(hits), 1) if hits else None,
            }
        summary[group] = entry
    return summary


def review_past_scans(report_dir: Path, frames: Mapping[str, pd.DataFrame], *,
                      exclude: Iterable[Path] = (), horizons: Sequence[int] = HORIZONS) -> Dict[str, Any]:
    """Score every saved scan point with ``frames`` (daily bars keyed by ticker, plus SPY)."""
    events = unique_events(load_scan_payloads(report_dir, exclude))
    benchmark = frames.get(BENCHMARK)
    scores = [score_event(e, frames.get(e["code"]), benchmark, horizons) for e in events]
    scan_dates = sorted({e["scan_date"] for e in events})
    return {"scans": len(scan_dates), "first_scan": scan_dates[0] if scan_dates else None,
            "events": len(events), "horizons": list(horizons), "summary": summarise(events, scores, horizons)}


_GROUP_LABELS = {
    "en": {"buy_sent_to_llm": "Buy, sent to LLM", "buy_not_sent": "Buy, over the cap",
           "buy_forming": "Buy, forming stroke", "sell": "Sell points"},
    "zh": {"buy_sent_to_llm": "买点·已送 LLM", "buy_not_sent": "买点·超出上限",
           "buy_forming": "买点·笔未完成", "sell": "卖点"},
}


def _cell(value: Any, suffix: str = "") -> str:
    return "-" if value is None else f"{value}{suffix}"


def render_review_markdown(review: Mapping[str, Any], language: str = "zh") -> str:
    en = str(language or "").lower().startswith("en")
    labels = _GROUP_LABELS["en" if en else "zh"]
    horizons = review.get("horizons") or list(HORIZONS)
    if en:
        lines = ["", "## Past scan picks", "",
                 f"- {review['events']} unique points from {review['scans']} earlier scan(s) since "
                 f"{_cell(review.get('first_scan'))}; entry = close on the scan's data date; sell points count a "
                 f"fall as a gain; excess is against {BENCHMARK} over the same bars.",
                 f"- Groups with fewer than {MIN_SAMPLE} scored points are too small to judge.", "",
                 "| Group | Bars | n | Mean return % | Mean excess % | Beat SPY % | Invalidated % |",
                 "|---|---|---|---|---|---|---|"]
    else:
        lines = ["", "## 历史扫描结果回看", "",
                 f"- 自 {_cell(review.get('first_scan'))} 起 {review['scans']} 次扫描中的 {review['events']} 个去重买卖点；"
                 f"入场价为扫描数据日收盘价；卖点以下跌为盈利；超额收益对比同期 {BENCHMARK}。",
                 f"- 有效样本少于 {MIN_SAMPLE} 的分组不足以下结论。", "",
                 "| 分组 | K 线数 | 样本 | 平均收益 % | 平均超额 % | 跑赢 SPY % | 触发失效 % |",
                 "|---|---|---|---|---|---|---|"]
    for group in GROUP_ORDER:
        entry = review["summary"].get(group) or {}
        for h in horizons:
            stats = entry.get(h) or entry.get(str(h)) or {}
            lines.append(
                f"| {labels[group]} | {h} | {stats.get('n', 0)} | {_cell(stats.get('mean_return_pct'))} "
                f"| {_cell(stats.get('mean_excess_pct'))} | {_cell(stats.get('beat_benchmark_pct'))} "
                f"| {_cell(stats.get('invalidated_pct'))} |"
            )
    return "\n".join(lines) + "\n"
