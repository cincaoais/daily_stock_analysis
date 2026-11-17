# -*- coding: utf-8 -*-
"""Chan pre-screen for US index constituents, without any LLM call.

Downloads one year of daily bars for every constituent in a single yfinance
request (OpenD is not used, so the moomoo historical K-line quota is untouched),
builds a daily-only Chan snapshot per ticker, and keeps tickers with a recent,
still-valid daily buy point. Only those go on to the normal LLM analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from src.chan.engine import build_chan_snapshot
from src.chan.service import _LOOKBACK_DAYS

logger = logging.getLogger(__name__)

_UNIVERSE_ALIASES = {
    "sp500": "sp500", "spx": "sp500", "s&p500": "sp500", "snp500": "sp500",
    "nasdaq100": "nasdaq100", "ndx": "nasdaq100", "nasdaq": "nasdaq100", "qqq": "nasdaq100",
}
_UNIVERSE_CACHE_MAX_AGE_DAYS = 7
_STATE_RANK = {"confirmed": 0, "candidate": 1}


def parse_scan_universes(value: str) -> List[str]:
    """``"sp500,ndx"`` -> ``["sp500", "nasdaq100"]``; unknown names raise ValueError."""
    names: List[str] = []
    for raw in str(value or "").split(","):
        key = raw.strip().lower().replace(" ", "").replace("-", "")
        if not key:
            continue
        if key not in _UNIVERSE_ALIASES:
            raise ValueError(f"Unknown scan universe {raw.strip()!r}; use sp500 and/or nasdaq100")
        if _UNIVERSE_ALIASES[key] not in names:
            names.append(_UNIVERSE_ALIASES[key])
    if not names:
        raise ValueError("No scan universe given; use sp500 and/or nasdaq100")
    return names


def load_universe(
    names: Sequence[str],
    cache_dir: Path,
    *,
    fetcher: Optional[Callable[[str], List[str]]] = None,
    now: Optional[datetime] = None,
) -> List[str]:
    """Union of the named constituent lists. A list is re-fetched when its cache is
    older than a week; a failed fetch falls back to a stale cache when there is one."""
    if fetcher is None:
        from src.services.screening.snapshot_us import fetch_index_tickers as fetcher
    now = now or datetime.now()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers: set = set()
    for name in names:
        path = cache_dir / f"universe_{name}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        fresh = cached and (now - datetime.fromisoformat(cached["fetched_at"])).days < _UNIVERSE_CACHE_MAX_AGE_DAYS
        if fresh:
            tickers.update(cached["tickers"])
            continue
        try:
            fetched = fetcher(name)
            path.write_text(json.dumps({"fetched_at": now.isoformat(timespec="seconds"), "tickers": fetched}),
                            encoding="utf-8")
            tickers.update(fetched)
        except Exception as exc:  # noqa: BLE001 - a stale list beats no scan
            if not cached:
                raise RuntimeError(f"Could not fetch the {name} constituent list: {exc}") from exc
            logger.warning("[chan-scan] %s 成分股获取失败，使用 %s 的缓存: %s", name, cached["fetched_at"], exc)
            tickers.update(cached["tickers"])
    return sorted(tickers)


def download_daily_frames(
    tickers: Sequence[str],
    *,
    days: int = _LOOKBACK_DAYS["1d"],
    downloader: Optional[Callable[..., pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """One bulk yfinance request -> ``{ticker: frame[date, open, high, low, close, volume]}``.

    Prices are split/dividend adjusted, like the Futu QFQ daily bars the analysis uses.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    from src.core.trading_calendar import MARKET_TIMEZONE

    today = pd.Timestamp.now(tz=MARKET_TIMEZONE["us"]).normalize()
    yahoo = {t: t.replace(".", "-") for t in tickers}
    data = downloader(
        list(yahoo.values()),
        start=(today - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    frames: Dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return frames
    present = set(data.columns.get_level_values(0)) if isinstance(data.columns, pd.MultiIndex) else set()
    for ticker, symbol in yahoo.items():
        if symbol not in present:
            continue
        raw = data[symbol].dropna(subset=["Close"])
        if raw.empty:
            continue
        frames[ticker] = pd.DataFrame({
            "date": raw.index,
            "open": raw["Open"].to_numpy(),
            "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(),
            "close": raw["Close"].to_numpy(),
            "volume": raw["Volume"].to_numpy() if "Volume" in raw else 0,
        })
    return frames


@dataclass
class ScanResult:
    universes: List[str]
    universe_size: int
    scanned: int
    max_bars_since: int
    max_candidates: int
    points: List[Dict[str, Any]] = field(default_factory=list)  # recent still-valid daily points, both sides
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # one buy point per ticker, ranked
    unusable: List[str] = field(default_factory=list)
    as_of: Optional[str] = None

    @property
    def selected(self) -> List[Dict[str, Any]]:
        return self.candidates[: self.max_candidates]

    @property
    def selected_codes(self) -> List[str]:
        return [row["code"] for row in self.selected]


def scan_frames(
    frames: Mapping[str, Optional[pd.DataFrame]],
    *,
    max_bars_since: int = 5,
    max_candidates: int = 20,
    universes: Iterable[str] = (),
    universe_size: Optional[int] = None,
) -> ScanResult:
    """Build a daily Chan snapshot per ticker and collect recent still-valid points.

    Candidates are ranked confirmed before candidate, then most recent, then the
    smallest distance from the last close down to the invalidation price.
    """
    result = ScanResult(
        universes=list(universes), universe_size=universe_size if universe_size is not None else len(frames),
        scanned=0, max_bars_since=max_bars_since, max_candidates=max_candidates,
    )
    last_bars: List[str] = []
    for code, frame in frames.items():
        level = build_chan_snapshot({"1d": frame}, code, market="us")["levels"]["D"]
        if level["status"] not in ("available", "partial"):
            result.unusable.append(code)
            continue
        result.scanned += 1
        last_bars.append(level["last_bar"])
        last_close = level["last_close"]
        for point in level.get("buy_sell_points") or []:
            if not point.get("still_valid") or point.get("bars_since", 0) > max_bars_since:
                continue
            invalidation = point.get("invalidation_price")
            risk = None
            if invalidation and last_close:
                risk = round(abs(last_close - invalidation) / last_close * 100, 2)
            result.points.append({
                "code": code,
                "evidence_id": point["evidence_id"],
                "type": point["type"],
                "side": point["side"],
                "state": point["state"],
                "dt": point["dt"],
                "price": point["price"],
                "invalidation_price": invalidation,
                "last_close": last_close,
                "risk_to_invalidation_pct": risk,
                "bars_since": point["bars_since"],
                "trend_type": level.get("trend_type"),
                "position_vs_last_zhongshu": level.get("position_vs_last_zhongshu"),
            })

    def rank(row: Dict[str, Any]):
        risk = row["risk_to_invalidation_pct"]
        return (_STATE_RANK[row["state"]], row["bars_since"], risk if risk is not None else float("inf"))

    # Forming points stay in ``points`` for tracking but never become candidates: the
    # analysis buy gate (enforce_chan_buy_gate) only accepts confirmed/candidate points.
    best: Dict[str, Dict[str, Any]] = {}
    for row in result.points:
        if row["side"] != "buy" or row["state"] not in _STATE_RANK:
            continue
        if row["code"] not in best or rank(row) < rank(best[row["code"]]):
            best[row["code"]] = row
    result.candidates = sorted(best.values(), key=lambda r: (rank(r), r["code"]))
    result.as_of = max(last_bars) if last_bars else None
    return result


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def render_scan_markdown(result: ScanResult, language: str = "zh") -> str:
    en = str(language or "").lower().startswith("en")
    sells = [p for p in result.points if p["side"] == "sell"]
    if en:
        lines = [
            f"# Chan daily scan: {', '.join(result.universes)}",
            "",
            f"- Data as of: {_fmt(result.as_of)} (Yahoo daily bars, adjusted); no LLM used in this step",
            f"- Universe: {result.universe_size} tickers, scanned {result.scanned}, unusable {len(result.unusable)}",
            f"- Rule: still-valid confirmed or candidate daily buy point formed within the last "
            f"{result.max_bars_since} bars (points on a still-forming stroke are logged in the JSON only; "
            "the buy gate cannot act on them)",
            f"- Buy-point tickers: {len(result.candidates)}; sent to LLM analysis: {len(result.selected)} "
            f"(cap {result.max_candidates}); recent valid sell points: {len(sells)}",
            "- Not advice. In the repo's 10-symbol event study (evals/chan_event_study) daily buy points "
            "showed no excess return on average; the LLM pass and the buy gate still apply.",
            "",
            "| # | Code | Point | State | Date | Point price | Invalidation | Last close | Risk to invalidation % "
            "| Bars since | Trend | LLM |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            f"# 缠论日线扫描：{', '.join(result.universes)}",
            "",
            f"- 数据截至：{_fmt(result.as_of)}（Yahoo 复权日线）；本步骤不调用 LLM",
            f"- 股票池：{result.universe_size} 只，已扫描 {result.scanned}，不可用 {len(result.unusable)}",
            f"- 规则：最近 {result.max_bars_since} 根日线内出现、仍有效且状态为 confirmed 或 candidate 的日线买点"
            "（笔仍在形成中的买点只记入 JSON，买入闸门不会据此给出买入）",
            f"- 出现买点：{len(result.candidates)} 只；送入 LLM 分析：{len(result.selected)} 只"
            f"（上限 {result.max_candidates}）；近期有效卖点：{len(sells)} 个",
            "- 非投资建议。仓库的 10 只股票事件研究（evals/chan_event_study）中，日线买点平均没有超额收益；"
            "LLM 分析与买入闸门仍然生效。",
            "",
            "| # | 代码 | 买卖点 | 状态 | 日期 | 买点价 | 失效价 | 最新收盘 | 距失效价 % | 已过 K 线 | 走势 | LLM |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    selected = set(result.selected_codes)
    for i, row in enumerate(result.candidates, start=1):
        sent = ("yes" if en else "是") if row["code"] in selected else ("no (over cap)" if en else "否（超出上限）")
        lines.append(
            f"| {i} | {row['code']} | {row['type']} | {row['state']} | {row['dt']} | {_fmt(row['price'])} "
            f"| {_fmt(row['invalidation_price'])} | {_fmt(row['last_close'])} | {_fmt(row['risk_to_invalidation_pct'])} "
            f"| {row['bars_since']} | {_fmt(row['trend_type'])} | {sent} |"
        )
    if not result.candidates:
        lines.append("| - | " + ("no tickers matched" if en else "没有符合条件的股票") + " | | | | | | | | | | |")
    return "\n".join(lines) + "\n"


def scan_output_paths(report_dir: Path, now: Optional[datetime] = None) -> Dict[str, Path]:
    stamp = (now or datetime.now()).strftime("%Y%m%d")
    return {"markdown": report_dir / f"chan_scan_{stamp}.md", "json": report_dir / f"chan_scan_{stamp}.json"}


def save_scan_outputs(result: ScanResult, report_dir: Path, language: str = "zh",
                      now: Optional[datetime] = None, review: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Write ``chan_scan_YYYYMMDD.md`` and a ``.json`` with every recent point (both sides),
    so the hit rate of scan picks can be measured later. ``review`` (scores of earlier
    scans' points) is appended to both when given."""
    from src.chan.scan_review import render_review_markdown

    report_dir.mkdir(parents=True, exist_ok=True)
    paths = scan_output_paths(report_dir, now)
    md_path, json_path = paths["markdown"], paths["json"]
    markdown = render_scan_markdown(result, language)
    if review and review.get("events"):
        markdown += render_review_markdown(review, language)
    md_path.write_text(markdown, encoding="utf-8")
    payload = {
        "universes": result.universes, "as_of": result.as_of, "universe_size": result.universe_size,
        "scanned": result.scanned, "unusable": result.unusable, "max_bars_since": result.max_bars_since,
        "max_candidates": result.max_candidates, "selected": result.selected_codes,
        "candidates": result.candidates, "points": result.points,
    }
    if review is not None:
        payload["review"] = review
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


def run_chan_scan(
    universe_arg: str,
    *,
    max_bars_since: int = 5,
    max_candidates: int = 20,
    report_dir: Path,
    cache_dir: Path,
    language: str = "zh",
) -> ScanResult:
    """Fetch constituents, scan them, write the scan report; returns the result."""
    universes = parse_scan_universes(universe_arg)
    tickers = load_universe(universes, cache_dir)
    logger.info("[chan-scan] %s: %d 只成分股，批量下载日线...", ",".join(universes), len(tickers))
    from src.chan.scan_review import BENCHMARK, review_past_scans

    frames = download_daily_frames(list(tickers) + ([BENCHMARK] if BENCHMARK not in tickers else []))
    missing = sorted(set(tickers) - set(frames))
    scan_input = {code: frame for code, frame in frames.items() if code in set(tickers)}
    result = scan_frames(scan_input, max_bars_since=max_bars_since, max_candidates=max_candidates,
                         universes=universes, universe_size=len(tickers))
    result.unusable = sorted(set(result.unusable) | set(missing))
    try:
        # Score earlier scans with the bars just downloaded; today's file is rewritten below.
        review = review_past_scans(report_dir, frames, exclude=scan_output_paths(report_dir).values())
    except Exception as exc:  # noqa: BLE001 - the review must never block the scan
        logger.warning("[chan-scan] 历史扫描回看失败: %s", exc)
        review = None
    paths = save_scan_outputs(result, report_dir, language, review=review)
    logger.info(
        "[chan-scan] 扫描 %d 只，买点 %d 只，送入 LLM %d 只: %s；报告 %s",
        result.scanned, len(result.candidates), len(result.selected), result.selected_codes, paths["markdown"],
    )
    return result
