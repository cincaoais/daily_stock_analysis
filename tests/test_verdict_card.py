# -*- coding: utf-8 -*-
"""Telegram verdict card: renderer, quick card, pipeline hook and bot buttons. No network."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bot.models import BotResponse
from src.services.verdict_card import (
    action_label,
    chan_summary,
    price_near,
    render_quick_card,
    render_verdict_card,
)
from tests.test_telegram_polling import _client, _FakeTelegram, _sent, _tap, _update


def _point(kind, side, dt, inv, state="confirmed", valid=True, bars=2):
    return {"evidence_id": f"D.{kind}.{dt}", "type": kind, "side": side, "dt": dt, "price": inv + 5,
            "invalidation_price": inv, "state": state, "still_valid": valid, "bars_since": bars}


def _snapshot(daily_points, confirm_points=()):
    return {
        "level_roles": {"direction": "W", "operating": "D", "confirmation": "60M"},
        "data_quality": {"status": "available"},
        "levels": {
            "W": {"status": "available", "trend_type": "up", "buy_sell_points": []},
            "D": {"status": "available", "trend_type": "consolidation", "last_close": 351.91,
                  "buy_sell_points": list(daily_points)},
            "60M": {"status": "available", "buy_sell_points": list(confirm_points)},
        },
    }


AAPL_SNAPSHOT = _snapshot(
    [_point("3B", "buy", "2026-07-23", 301.93, "invalidated", False, 82),
     _point("3B", "buy", "2026-10-20", 334.45, "invalidated", False, 20)],
    [_point("3B", "buy", "2026-11-17 13:30", 347.59, "candidate", True, 3)],
)


def _result(**overrides):
    base = dict(
        code="AAPL", name="Apple", action="watch", decision_type="hold", sentiment_score=48,
        trend_prediction="Sideways", confidence_level="Medium", current_price=351.91, change_pct=None,
        market_snapshot={"price": "351.91", "prev_close": "352.78"}, analysis_summary="",
        dashboard={
            "core_conclusion": {
                "one_sentence": "No valid daily buy point is left, so wait <for> the daily stroke & turn.",
                "position_advice": {"no_position": "Stay out for now.", "has_position": "Hold, do not add."},
            },
            "battle_plan": {"sniper_points": {
                "stop_loss": "347.59 (invalidation_price of 60M.3B.2026-11-17 13:30)",
                "take_profit": "1-2/10 size; 359.37 (60M resistance) then 360.83"}},
            "intelligence": {"risk_alerts": ["2026-11-18: outage"], "positive_catalysts": ["sticky base"]},
            "phase_decision": {"phase_context": {"phase": "premarket", "effective_daily_bar_date": "2026-11-17"}},
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_verdict_card_answers_the_three_questions():
    card = render_verdict_card(_result(), AAPL_SNAPSHOT, report_date="20261118")
    text = card["text"]
    assert text.startswith("🟡 <b>AAPL</b> Apple — <b>WAIT</b> · score 48/100")
    assert "351.91 (-0.25%) · sideways" in text  # change computed from prev_close
    assert "Daily: no valid buy point (last 3B Oct 20 failed) · weekly up" in text
    assert "60m: 3B candidate (3 bars ago) · invalid below 347.59" in text
    assert "&lt;for&gt;" in text and "&amp; turn" in text  # LLM text is escaped
    assert "🛑 stop 347.59 · 🎯 target 359.37" in text  # sizes like 1-2/10 are skipped
    assert "<blockquote expandable>• Not holding: Stay out for now." in text
    assert "data to 2026-11-17 · pre-market · confidence medium · not advice" in text
    assert card["reply_markup"]["inline_keyboard"][0] == [
        {"text": "📄 Full report", "callback_data": "/report AAPL 20261118"},
        {"text": "🔁 Refresh", "callback_data": "/analyze AAPL"}]
    assert "<b>" not in card["plain"] and "<blockquote" not in card["plain"] and "<for>" in card["plain"]
    assert len(text) < 4096


def test_verdict_card_without_snapshot_or_report_file():
    card = render_verdict_card(_result(action="buy", dashboard={}), None, report_date=None)
    assert "🟢 <b>AAPL</b> Apple — <b>BUY</b>" in card["text"] and "Chan: not available" in card["text"]
    assert "blockquote" not in card["text"]
    assert [b["text"] for b in card["reply_markup"]["inline_keyboard"][0]] == ["🔁 Refresh"]


def test_index_report_button_keeps_the_file_name_case():
    card = render_verdict_card(_result(code="sh000016", name=""), None, report_date="20261118")
    assert card["text"].startswith("🟡 <b>SH000016</b> — <b>WAIT</b>")
    assert card["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "/report sh000016 20261118"


def test_chan_summary_valid_buy_and_sell_and_price_guard():
    buy = _snapshot([_point("2B", "buy", "2026-11-14", 340.0, bars=3)])
    assert chan_summary(buy, 351.0) == {
        "lines": ["Daily: 2B confirmed (3 bars ago) · invalid below 340.00 · weekly up"],
        "setup": "buy", "stop": 340.0}
    assert chan_summary(buy, 339.0)["setup"] == "none"  # price already below the invalidation
    sell = _snapshot([_point("2B", "buy", "2026-11-10", 340.0), _point("1S", "sell", "2026-11-17", 360.0)])
    assert chan_summary(sell, 351.0)["setup"] == "sell"
    assert chan_summary({"levels": {"D": {"status": "missing"}}}, 1.0)["setup"] == "unknown"


def test_forming_points_are_not_called_live():
    # NVDA on 2026-11-18: daily 3S forming (valid, not confirmed), 60m 3S confirmed.
    nvda = _snapshot([_point("2B", "buy", "2026-07-29", 189.59, "invalidated", False, 78),
                      _point("3S", "sell", "2026-11-13", 208.30, "forming", True, 2)],
                     [_point("3S", "sell", "2026-11-11 14:30", 213.32, "confirmed", True, 30),
                      _point("3S", "sell", "2026-11-17 16:00", 165.89, "forming", True, 0)])
    summary = chan_summary(nvda, 155.46)
    assert summary["setup"] == "sell_forming" and summary["stop"] is None
    assert summary["lines"] == ["Daily: 3S forming (2 bars ago) · invalid above 208.30 · weekly up",
                                "60m: 3S confirmed (30 bars ago) · invalid above 213.32"]
    card = render_quick_card("NVDA", nvda, price=155.46, change_pct=-0.4355, last_close=True)
    assert "🟠 <b>NVDA</b>" in card["text"] and "Chan sell signal forming — avoid new buys for now" in card["text"]
    assert card["text"].splitlines()[1] == "155.46 (-0.44%, last close)"


def test_price_zone_and_position_size():
    from src.services.verdict_card import position_size, price_zone

    assert price_zone("148.80-150.33 (30M confirmed 1B 30M.1B.2026-11-12 at 148.80)", 155.46) == "148.80–150.33"
    assert price_zone("No buy point; D.3S.2026-11-13 at 170.00", 155.46) == "170.00"  # dates are not ranges
    assert price_zone("1-2/10 trial only", 155.46) is None and price_zone("150-151", None) is None
    assert position_size("Suggested position: 0-2/10 (watch; trial size only)") == "0-2/10"
    assert position_size("2/10 (holders trim toward 2/10; new money 0/10)") == "2/10"
    assert position_size("no size given") is None


def _plan(action, **battle):
    sniper = {"ideal_buy": "Buy 349.67-352.05 on the 60M 3B", "secondary_buy": "347.89-349.67 if a daily 3B forms",
              "stop_loss": "347.59 (60M.3B.2026-11-17 13:30)", "take_profit": "359.37 then 360.83"}
    dashboard = dict(_result().dashboard, battle_plan={
        "sniper_points": sniper, "position_strategy": {"suggested_position": "Suggested position: 2/10"}, **battle})
    lines = render_verdict_card(_result(action=action, dashboard=dashboard), AAPL_SNAPSHOT)["plain"].splitlines()
    return [line for line in lines if line.startswith(("💵", "📉", "🛑"))]


def test_plan_line_is_worded_by_the_verdict():
    assert _plan("buy") == ["💵 Buy near 349.67–352.05 · size 2/10", "🛑 stop 347.59 · 🎯 target 359.37"]
    assert _plan("add")[0] == "💵 Add near 349.67–352.05 · size 2/10"
    assert _plan("watch") == ["💵 No buy now · buy zone 347.89–349.67 only after confirmation · position 2/10",
                              "🛑 stop 347.59 · 🎯 target 359.37"]
    assert _plan("reduce") == ["📉 Trim to 2/10", "🛑 exit below 347.59 · 🎯 rebound target 359.37"]
    assert _plan("sell") == ["📉 Exit the position", "🛑 exit below 347.59"]
    assert _plan("avoid")[0] == "📉 Avoid new buys"


def test_quote_change_pct_falls_back_to_previous_close():
    from src.services.verdict_card import quote_change_pct

    assert quote_change_pct(155.46, 1.5, 156.14, -0.68) == 1.5  # provider value wins
    assert round(quote_change_pct(155.46, None, 156.14, -0.68), 2) == -0.44  # pre-market NVDA
    assert round(quote_change_pct(155.46, None, None, -0.68), 2) == -0.44
    assert quote_change_pct(None, None, 156.14, None) is None

    forming_buy = _snapshot([_point("1B", "buy", "2026-11-17", 300.0, "forming", True, 1)])
    assert chan_summary(forming_buy, 351.0)["setup"] == "buy_forming"
    assert chan_summary(forming_buy, 351.0)["stop"] is None  # no buy stop before the buy gate would accept it
    assert "Chan buy setup forming, not confirmed — wait" in render_quick_card("X", forming_buy)["text"]


def test_price_near_and_action_labels():
    assert price_near("Stop at 347.59 (60M.3B.2026-11-17 13:30)", 351.9) == 347.59
    assert price_near("1-8/10 trial, stop 18.4", 370.0) is None  # nothing near the price
    assert price_near("347.59", None) is None  # no reference, no guess
    assert action_label("reduce") == ("🟠", "TRIM")
    assert action_label("", "sell") == ("🔴", "SELL") and action_label(None, None) == ("🟡", "WAIT")


def test_quick_card_says_no_setup_and_has_no_buttons():
    card = render_quick_card("aapl", AAPL_SNAPSHOT, price=351.91, change_pct=-0.25, name="Apple")
    assert card["text"].splitlines()[:2] == ["🟡 <b>AAPL</b> Apple", "351.91 (-0.25%)"]
    assert "➡️ No Chan buy setup — wait" in card["text"] and "reply_markup" not in card
    live = render_quick_card("X", _snapshot([_point("3B", "buy", "2026-11-17", 300.0)]))
    assert "Chan buy setup is live" in live["text"] and "🛑 Chan stop 300.00" in live["text"]
    assert live["text"].splitlines()[1] == "351.91"  # falls back to the daily close


# ---------------------------------------------------------------- pipeline --
def _pipeline(platform="telegram"):
    from src.core.pipeline import StockAnalysisPipeline

    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace()
    pipeline.source_message = SimpleNamespace(platform=platform)
    pipeline.notifier = SimpleNamespace(_extract_telegram_context_chat_id=lambda: "42")
    pipeline._chan_snapshots = {"AAPL": AAPL_SNAPSHOT}
    return pipeline


def test_pipeline_sends_card_for_telegram_requests_only():
    sender = MagicMock()
    sender.send_telegram_card.return_value = True
    with patch("src.notification_sender.telegram_sender.TelegramSender", return_value=sender):
        assert _pipeline()._try_send_telegram_card(_result(), "AAPL", "20261118") is True
        assert _pipeline("dingtalk")._try_send_telegram_card(_result(), "AAPL", None) is False
    kwargs = sender.send_telegram_card.call_args.kwargs
    assert kwargs["chat_id"] == "42" and "60m: 3B candidate" in sender.send_telegram_card.call_args.args[0]
    assert kwargs["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "/report AAPL 20261118"

    sender.send_telegram_card.return_value = False  # card failed -> caller sends the text report
    with patch("src.notification_sender.telegram_sender.TelegramSender", return_value=sender):
        assert _pipeline()._try_send_telegram_card(_result(), "AAPL", None) is False


# --------------------------------------------------------------- bot side --
def _accepted(code):
    response = BotResponse.markdown_response("✅ submitted")
    response.extra = {"stock_code": code}
    return response


def test_accepted_analyze_sends_quick_card_instead_of_notice():
    api = _FakeTelegram([[_update(1, "/analyze aapl")]])
    replies, codes = [], []
    client = _client(api, [], replies, quick_card=lambda code: codes.append(code) or {
        "text": "<b>AAPL</b>", "plain": "AAPL"})
    client._dispatch = lambda m: _accepted("AAPL")
    client.poll_once()
    assert codes == ["AAPL"] and replies == []
    expected = {"chat_id": "42", "text": "<b>AAPL</b>", "parse_mode": "HTML", "disable_web_page_preview": True}
    assert _sent(api, "sendMessage")[0] == expected


def test_quick_card_failure_or_rejection_keeps_the_text_reply():
    api = _FakeTelegram([[_update(1, "/analyze AAPL"), _update(2, "/analyze ???")]])
    replies = []
    client = _client(api, [], replies, quick_card=lambda code: None)
    responses = iter([_accepted("AAPL"), BotResponse.error_response("bad code")])
    client._dispatch = lambda m: next(responses)
    client.poll_once()
    assert [text for _, text in replies] == ["✅ submitted", "❌ 错误：bad code"]
    assert _sent(api, "sendMessage") == []


def test_html_rejected_card_is_resent_as_plain_text():
    from tests.test_telegram_polling import _Resp

    api = _FakeTelegram([[_update(1, "/analyze AAPL")]],
                        fail={"sendMessage": _Resp({"ok": False, "description": "can't parse entities"}, 400)})
    client = _client(api, [], [], quick_card=lambda code: {"text": "<b>x", "plain": "x"})
    client._dispatch = lambda m: _accepted("AAPL")
    client.poll_once()
    sent = _sent(api, "sendMessage")
    assert sent[0]["parse_mode"] == "HTML" and "parse_mode" not in sent[1] and sent[1]["text"] == "x"


def test_report_button_resends_the_saved_file(tmp_path):
    (tmp_path / "report_20261118_AAPL.md").write_text("# AAPL full report", encoding="utf-8")
    api = _FakeTelegram([[_tap(1, "/report AAPL 20261118"), _tap(2, "/report AAPL 20991231"),
                          _tap(3, "/report ../etc 20261118")]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies, reports_dir=tmp_path)
    client.poll_once()
    assert dispatched == []  # handled by the client, no LLM run
    assert replies[0] == ("42", "# AAPL full report")
    assert "no longer on disk" in replies[1][1] and "not valid" in replies[2][1]
    assert _sent(api, "answerCallbackQuery")[0]["text"] == "Opening the full report…"
