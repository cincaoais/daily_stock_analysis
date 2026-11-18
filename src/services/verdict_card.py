# -*- coding: utf-8 -*-
"""
Telegram "verdict card": a short answer to "how is it doing, what does Chan say,
buy or not" for bot replies, with the reasons folded and the full report behind a button.

Two cards share the same Chan summary:

* :func:`render_verdict_card` - from a finished AnalysisResult (LLM verdict, stop, target).
* :func:`render_quick_card` - from the Chan snapshot and a quote only (no LLM, ~2 s),
  sent right after the tap while the full analysis runs.

Cards are Telegram HTML (``parse_mode=HTML``); every dynamic string is escaped and a
plain-text version is returned for the fallback send.
"""

from __future__ import annotations

import html
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# action taxonomy (src/schemas/decision_action) -> (emoji, plain label)
_ACTION_LABELS = {
    "buy": ("🟢", "BUY"),
    "add": ("🟢", "ADD"),
    "hold": ("🔵", "HOLD"),
    "watch": ("🟡", "WAIT"),
    "alert": ("🟡", "WAIT"),
    "reduce": ("🟠", "TRIM"),
    "sell": ("🔴", "SELL"),
    "avoid": ("🔴", "AVOID"),
}
_DECISION_FALLBACK = {"buy": "buy", "sell": "sell", "hold": "watch"}
_LEVEL_NAMES = {"W": "Weekly", "D": "Daily", "60M": "60m", "30M": "30m", "15M": "15m", "5M": "5m"}
_PHASE_NAMES = {"premarket": "pre-market", "intraday": "market open", "postmarket": "after close",
                "closed": "market closed"}
_LIVE_STATES = {"confirmed", "candidate"}  # same states enforce_chan_buy_gate accepts
_NUMBER = re.compile(r"(?<![\w.])\d{1,6}(?:,\d{3})*(?:\.\d+)?")
_MAX_CARD_CHARS = 3800  # Telegram limit is 4096; keep a margin


# ------------------------------------------------------------------ helpers --
def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=False)


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def _fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}" if value < 10000 else f"{value:,.0f}"


def _clip(text: Any, limit: int) -> str:
    """Trim to ``limit`` chars, preferring a sentence or clause boundary."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for mark in (". ", "; ", ": ", ", "):
        pos = cut.rfind(mark)
        if pos >= limit * 0.5:
            return cut[: pos + 1] if mark == ". " else cut[: pos + 1].rstrip(",;: ") + "…"
    return cut.rstrip() + "…"


def price_near(text: Any, reference: Optional[float], tolerance: float = 0.5) -> Optional[float]:
    """First number in ``text`` within ±``tolerance`` of ``reference`` (skips dates, sizes, %).

    Needs a reference price: without one a date or position size could pass as a price.
    """
    if not text or not reference:
        return None
    for match in _NUMBER.finditer(str(text)):
        tail = str(text)[match.end(): match.end() + 1]
        if tail in {"%", "/", "-"}:
            continue
        value = _num(match.group())
        if value is None or value <= 0:
            continue
        if abs(value - reference) <= reference * tolerance:
            return value
    return None


def _short_date(value: Any) -> str:
    """``2026-11-17 13:30`` -> ``Nov 17 13:30``; ``2026-10-20`` -> ``Oct 20``."""
    text = str(value or "").strip()
    for fmt, size, out in (("%Y-%m-%d %H:%M", 16, "%b %d %H:%M"), ("%Y-%m-%d", 10, "%b %d")):
        try:
            return datetime.strptime(text[:size], fmt).strftime(out)
        except ValueError:
            continue
    return text


def action_label(action: Any, decision_type: Any = None) -> Tuple[str, str]:
    key = str(action or "").strip().lower()
    if key not in _ACTION_LABELS:
        key = _DECISION_FALLBACK.get(str(decision_type or "").strip().lower(), "watch")
    return _ACTION_LABELS[key]


# ------------------------------------------------------------ Chan summary --
def _point_text(point: Dict[str, Any]) -> str:
    kind = str(point.get("type") or "?")
    state = str(point.get("state") or "")
    bars = point.get("bars_since")
    when = f"{bars} bars ago" if isinstance(bars, int) else _short_date(point.get("dt"))
    inv = _num(point.get("invalidation_price"))
    side = point.get("side")
    guard = ""
    if inv is not None:
        guard = f" · invalid {'below' if side == 'buy' else 'above'} {_fmt_price(inv)}"
    return f"{kind} {state} ({when}){guard}".replace("  ", " ")


def _valid_points(level: Dict[str, Any], price: Optional[float]) -> List[Dict[str, Any]]:
    points = []
    for point in level.get("buy_sell_points") or []:
        if not isinstance(point, dict) or not point.get("still_valid"):
            continue
        if str(point.get("state")) == "invalidated":
            continue
        inv = _num(point.get("invalidation_price"))
        if price is not None and inv is not None:
            if point.get("side") == "buy" and price < inv:
                continue
            if point.get("side") == "sell" and price > inv:
                continue
        points.append(point)
    return sorted(points, key=lambda p: str(p.get("dt") or ""))


def chan_summary(snapshot: Optional[Dict[str, Any]], price: Optional[float] = None) -> Dict[str, Any]:
    """Operating/confirmation-level reading of a Chan snapshot.

    Returns ``{"lines": [...], "setup": ..., "stop": float|None}`` where setup is
    ``buy``/``sell`` (confirmed or candidate point, the states the buy gate accepts),
    ``buy_forming``/``sell_forming`` (not confirmed yet), ``none`` or ``unknown``.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("levels"), dict):
        return {"lines": ["Chan: not available"], "setup": "unknown", "stop": None}
    levels = snapshot["levels"]
    roles = snapshot.get("level_roles") if isinstance(snapshot.get("level_roles"), dict) else {}
    operating = roles.get("operating") or ("D" if "D" in levels else next(iter(levels), None))
    confirmation = roles.get("confirmation")
    direction = roles.get("direction")
    lines: List[str] = []
    setup, stop = "none", None

    op_level = levels.get(operating) if operating else None
    if not isinstance(op_level, dict) or op_level.get("status") not in (None, "available", "partial"):
        return {"lines": ["Chan: not enough data"], "setup": "unknown", "stop": None}

    head = _LEVEL_NAMES.get(operating, operating)  # the engine trend label reads as a contradiction here
    valid = _valid_points(op_level, price)
    if valid:
        latest = valid[-1]
        side = "buy" if latest.get("side") == "buy" else "sell"
        live = str(latest.get("state")) in _LIVE_STATES
        setup = side if live else f"{side}_forming"
        if setup == "buy":
            stop = _num(latest.get("invalidation_price"))
        lines.append(f"{head}: {_point_text(latest)}")
    else:
        buys = [p for p in op_level.get("buy_sell_points") or [] if isinstance(p, dict) and p.get("side") == "buy"]
        buys.sort(key=lambda p: str(p.get("dt") or ""))
        failed = f" (last {buys[-1].get('type')} {_short_date(buys[-1].get('dt'))} failed)" if buys else ""
        lines.append(f"{head}: no valid buy point{failed}")

    conf_level = levels.get(confirmation) if confirmation else None
    if isinstance(conf_level, dict):
        conf_valid = _valid_points(conf_level, price)
        # A confirmed/candidate point says more than a newer one that is still forming.
        conf_live = [p for p in conf_valid if str(p.get("state")) in _LIVE_STATES]
        shown = (conf_live or conf_valid)[-1:] if conf_valid else []
        if shown:
            lines.append(f"{_LEVEL_NAMES.get(confirmation, confirmation)}: {_point_text(shown[0])}")

    dir_level = levels.get(direction) if direction else None
    if isinstance(dir_level, dict) and dir_level.get("trend_type"):
        lines[0] += f" · {_LEVEL_NAMES.get(direction, direction).lower()} {dir_level['trend_type']}"
    return {"lines": lines, "setup": setup, "stop": stop}


# ------------------------------------------------------------------- cards --
def _keyboard(code: str, report_date: Optional[str]) -> Dict[str, Any]:
    row = []
    if report_date:
        row.append({"text": "📄 Full report", "callback_data": f"/report {code} {report_date}"[:64]})
    row.append({"text": "🔁 Refresh", "callback_data": f"/analyze {code}"[:64]})
    return {"inline_keyboard": [row]}


def _plain(html_text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", html_text))


def _finish(lines: Iterable[str], why: List[str], footer: str,
            keyboard: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body = "\n".join(line for line in lines if line)
    if why:
        body += "\n<blockquote expandable>" + "\n".join(why) + "</blockquote>"
    if footer:
        body += f"\n<i>{footer}</i>"
    if len(body) > _MAX_CARD_CHARS:  # never expected with the clips above; keep the head
        body = body[:_MAX_CARD_CHARS]
        body = re.sub(r"<[^>]*$", "", body)
        body = _plain(body)  # unbalanced tags would be rejected; send it as text
    card = {"text": body, "plain": _plain(body)}
    if keyboard:
        card["reply_markup"] = keyboard
    return card


def render_verdict_card(
    result: Any,
    chan_snapshot: Optional[Dict[str, Any]] = None,
    *,
    report_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Card for a finished analysis. ``report_date`` (YYYYMMDD) enables the 📄 button."""
    dash = getattr(result, "dashboard", None) or {}
    core = dash.get("core_conclusion") or {}
    battle = dash.get("battle_plan") or {}
    sniper = battle.get("sniper_points") or {}
    intel = dash.get("intelligence") or {}
    phase_ctx = (dash.get("phase_decision") or {}).get("phase_context") or {}
    snapshot = getattr(result, "market_snapshot", None) or {}

    raw_code = str(getattr(result, "code", "") or "")  # report files are named with this exact code
    code = raw_code.upper()
    name = str(getattr(result, "name", "") or "")
    price = _num(getattr(result, "current_price", None)) or _num(snapshot.get("price")) or _num(snapshot.get("close"))
    change = _num(getattr(result, "change_pct", None))
    if change is None:
        prev = _num(snapshot.get("prev_close"))
        if price is not None and prev:
            change = (price - prev) / prev * 100

    emoji, label = action_label(getattr(result, "action", None), getattr(result, "decision_type", None))
    score = getattr(result, "sentiment_score", None)
    title = f"{emoji} <b>{_esc(code)}</b>" + (f" {_esc(_clip(name, 24))}" if name and name.upper() != code else "")
    title += f" — <b>{label}</b>" + (f" · score {_esc(score)}/100" if score is not None else "")

    status = f"{_fmt_price(price)}" if price is not None else "price n/a"
    if change is not None:
        status += f" ({change:+.2f}%)"
    trend = getattr(result, "trend_prediction", "")
    if trend:
        status += f" · {_esc(str(trend).lower())}"

    chan = chan_summary(chan_snapshot, price)
    verdict = core.get("one_sentence") or getattr(result, "analysis_summary", "")
    stop = price_near(sniper.get("stop_loss"), price) or chan["stop"]
    target = price_near(sniper.get("take_profit"), price)
    levels = " · ".join(part for part in (
        f"🛑 stop {_fmt_price(stop)}" if stop is not None else "",
        f"🎯 target {_fmt_price(target)}" if target is not None else "",
    ) if part)

    lines = [title, status]
    lines += [f"📐 {_esc(line)}" for line in chan["lines"]]
    if verdict:
        lines.append(f"➡️ {_esc(_clip(verdict, 220))}")
    if levels:
        lines.append(levels)

    why: List[str] = []
    advice = core.get("position_advice") or {}
    if advice.get("no_position"):
        why.append(f"• Not holding: {_esc(_clip(advice['no_position'], 230))}")
    if advice.get("has_position"):
        why.append(f"• Holding: {_esc(_clip(advice['has_position'], 230))}")
    risks = intel.get("risk_alerts") or []
    if risks:
        why.append(f"• ⚠️ {_esc(_clip(risks[0], 170))}")
    catalysts = intel.get("positive_catalysts") or []
    if catalysts:
        why.append(f"• ✨ {_esc(_clip(catalysts[0], 170))}")

    footer_parts = []
    bar_date = phase_ctx.get("effective_daily_bar_date")
    if bar_date:
        footer_parts.append(f"data to {_esc(bar_date)}")
    phase = phase_ctx.get("phase")
    if phase:
        footer_parts.append(_esc(_PHASE_NAMES.get(phase, phase)))
    confidence = getattr(result, "confidence_level", "")
    if confidence:
        footer_parts.append(f"confidence {_esc(str(confidence).lower())}")
    footer_parts.append("not advice")
    return _finish(lines, why, " · ".join(footer_parts), _keyboard(raw_code, report_date))


def render_quick_card(
    code: str,
    chan_snapshot: Optional[Dict[str, Any]],
    *,
    price: Optional[float] = None,
    change_pct: Optional[float] = None,
    name: str = "",
    last_close: bool = False,
) -> Dict[str, Any]:
    """Instant card from Chan + quote only; the LLM verdict follows.

    ``last_close`` marks a stale quote (market not open yet), shown as "last close".
    """
    code = str(code or "").upper()
    if price is None and isinstance(chan_snapshot, dict):
        roles = chan_snapshot.get("level_roles") or {}
        op = (chan_snapshot.get("levels") or {}).get(roles.get("operating") or "D") or {}
        price = _num(op.get("last_close"))
    chan = chan_summary(chan_snapshot, price)
    setup_text = {
        "buy": ("🟢", "Chan buy setup is live"),
        "buy_forming": ("🟡", "Chan buy setup forming, not confirmed — wait"),
        "sell": ("🔴", "Chan sell signal is live — avoid new buys"),
        "sell_forming": ("🟠", "Chan sell signal forming — avoid new buys for now"),
        "none": ("🟡", "No Chan buy setup — wait"),
        "unknown": ("⚪", "Chan reading unavailable"),
    }[chan["setup"]]
    title = f"{setup_text[0]} <b>{_esc(code)}</b>" + (f" {_esc(_clip(name, 24))}" if name else "")
    status = _fmt_price(price) if price is not None else "price n/a"
    notes = [f"{change_pct:+.2f}%"] if change_pct is not None else []
    if last_close and price is not None:
        notes.append("last close")
    if notes:
        status += f" ({', '.join(notes)})"
    lines = [title, status, *[f"📐 {_esc(line)}" for line in chan["lines"]], f"➡️ {setup_text[1]}"]
    if chan["stop"] is not None:
        lines.append(f"🛑 Chan stop {_fmt_price(chan['stop'])}")
    footer = "quick Chan check, no AI yet · full verdict in about a minute · not advice"
    return _finish(lines, [], footer, None)


_fetcher_manager = None
_fetcher_lock = threading.Lock()


def _shared_fetcher_manager():
    """One DataFetcherManager for quick cards (building one takes several seconds)."""
    global _fetcher_manager
    with _fetcher_lock:
        if _fetcher_manager is None:
            from data_provider import DataFetcherManager

            _fetcher_manager = DataFetcherManager()
        return _fetcher_manager


def build_quick_card(code: str) -> Optional[Dict[str, Any]]:
    """Chan snapshot + realtime quote -> quick card; None when there is no Chan data."""
    from data_provider.base import normalize_stock_code
    from src.chan.service import load_chan_snapshot
    from src.core.trading_calendar import get_market_for_stock

    manager = _shared_fetcher_manager()
    market = get_market_for_stock(normalize_stock_code(code))
    snapshot = load_chan_snapshot(manager, code, market=market, name="")
    if not isinstance(snapshot, dict) or (snapshot.get("data_quality") or {}).get("status") == "missing":
        return None
    price = change = None
    name = ""
    stale = False
    try:
        quote = manager.get_realtime_quote(code, log_final_failure=False)
    except Exception:  # noqa: BLE001 - the card falls back to the last daily close
        quote = None
    if quote is not None:
        price = _num(getattr(quote, "price", None))
        change = quote_change_pct(price, _num(getattr(quote, "change_pct", None)),
                                  _num(getattr(quote, "pre_close", None)),
                                  _num(getattr(quote, "change_amount", None)))
        name = str(getattr(quote, "name", "") or "")
        stale = bool(getattr(quote, "is_stale", False))
    return render_quick_card(code, snapshot, price=price, change_pct=change, name=name, last_close=stale)


def quote_change_pct(price: Optional[float], change_pct: Optional[float], pre_close: Optional[float],
                     change_amount: Optional[float]) -> Optional[float]:
    """Provider % change, else derived from the previous close (pre-market quotes omit it)."""
    if change_pct is not None:
        return change_pct
    if price is not None and pre_close:
        return (price - pre_close) / pre_close * 100
    if price is not None and change_amount is not None and price != change_amount:
        return change_amount / (price - change_amount) * 100
    return None


__all__ = ["render_verdict_card", "render_quick_card", "build_quick_card", "chan_summary", "price_near",
           "action_label"]
