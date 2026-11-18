# -*- coding: utf-8 -*-
"""Telegram long-polling client, /scan command and the market/scan NL intents. No network."""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bot.commands.scan import ScanCommand, parse_scan_args, render_scan_reply
from bot.dispatcher import CommandDispatcher
from bot.models import BotMessage, BotResponse, ChatType
from bot.platforms.telegram_polling import TelegramApiError, TelegramPollingClient


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


class _FakeTelegram:
    """Records calls; ``getUpdates`` returns queued batches."""

    def __init__(self, batches=(), fail=None):
        self.calls, self.batches, self.fail = [], list(batches), fail

    def post(self, url, json=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, json, timeout))
        if self.fail and method in self.fail:
            return self.fail[method]
        if method == "getMe":
            return _Resp({"ok": True, "result": {"username": "dsa_bot"}})
        if method == "getUpdates":
            return _Resp({"ok": True, "result": self.batches.pop(0) if self.batches else []})
        return _Resp({"ok": True, "result": True})


def _update(update_id, text, chat_id=42, age=0):
    return {"update_id": update_id, "message": {
        "message_id": update_id, "date": int(time.time() - age), "text": text,
        "chat": {"id": chat_id, "type": "private"}, "from": {"id": 7, "username": "me"}}}


def _client(api, dispatched, replies, **kwargs):
    client = TelegramPollingClient("TOKEN", "42", session=api,
                                   dispatch=lambda m: dispatched.append(m) or BotResponse.text_response("done"),
                                   reply=lambda chat, text: replies.append((chat, text)) or True,
                                   menu=[{"command": "scan", "description": "Chan scan"}], **kwargs)
    client._workers.submit = lambda fn, *a: fn(*a)  # run jobs inline
    return client


def test_offset_chat_filter_mention_strip_and_reply():
    api = _FakeTelegram([[_update(10, "/help@dsa_bot"), _update(11, "hi", chat_id=999)], [_update(12, "/status")]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies)
    client.prepare()
    assert client.poll_once() == 2
    assert client.poll_once() == 1

    polls = [payload for method, payload, _ in api.calls if method == "getUpdates"]
    assert "offset" not in polls[0] and polls[1]["offset"] == 12  # both updates confirmed, incl. the foreign one
    assert polls[0]["timeout"] == 50 and polls[0]["allowed_updates"] == ["message", "callback_query"]
    assert [m.content for m in dispatched] == ["/help", "/status"]  # chat 999 ignored
    msg = dispatched[0]
    assert msg.platform == "telegram" and msg.chat_id == "42" and msg.chat_type == ChatType.PRIVATE and msg.mentioned
    assert replies == [("42", "done"), ("42", "done")]
    assert [c[0] for c in api.calls[:3]] == ["deleteWebhook", "getMe", "setMyCommands"]
    assert api.calls[2][1] == {"commands": [{"command": "scan", "description": "Chan scan"}]}


def test_menu_hides_agent_commands_without_agent_mode():
    from bot.dispatcher import get_dispatcher, reset_dispatcher
    import bot.platforms.telegram_polling as tp

    client = TelegramPollingClient("T", "42", session=_FakeTelegram())
    with patch("src.config.get_config", return_value=SimpleNamespace(agent_mode=False)):
        names = [c["command"] for c in client._menu_commands()]
    assert names == ["scan", "market", "analyze", "batch", "status", "help"]
    assert all(len(text) <= 256 for _, text, _ in tp.MENU_COMMANDS)
    reset_dispatcher()
    with patch("src.config.get_config", return_value=SimpleNamespace()):
        registered = {c.name for c in get_dispatcher().list_commands(include_hidden=True)}
    reset_dispatcher()
    assert {name for name, _, _ in tp.MENU_COMMANDS} <= registered


def _tap(update_id, data, chat_id=42):
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "data": data, "from": {"id": 7, "username": "me"},
        "message": {"message_id": 99, "chat": {"id": chat_id, "type": "private"}}}}


def _sent(api, method):
    return [payload for m, payload, _ in api.calls if m == method]


@pytest.mark.parametrize("text", ["/analyze", "/a", "/analyze full", "/ANALYZE  full"])
def test_bare_analyze_tap_sends_watchlist_buttons(text):
    api = _FakeTelegram([[_update(1, text)]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies, watchlist=lambda: ["aapl", "HK00700", "AAPL", " "])
    client.poll_once()
    assert dispatched == [] and replies == []
    keyboard = _sent(api, "sendMessage")[0]["reply_markup"]["inline_keyboard"]
    assert keyboard == [
        [{"text": "AAPL", "callback_data": "/analyze AAPL"},
         {"text": "AAPL · full", "callback_data": "/analyze AAPL full"}],
        [{"text": "HK00700", "callback_data": "/analyze HK00700"},
         {"text": "HK00700 · full", "callback_data": "/analyze HK00700 full"}],
    ]


def test_bare_scan_tap_sends_index_buttons_and_args_skip_the_picker():
    api = _FakeTelegram([[_update(1, "/scan"), _update(2, "/scan sp500"), _update(3, "/analyze NVDA")]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies, watchlist=lambda: ["AAPL"])
    client.poll_once()
    assert [m.content for m in dispatched] == ["/scan sp500", "/analyze NVDA"]
    rows = _sent(api, "sendMessage")[0]["reply_markup"]["inline_keyboard"]
    assert [b["callback_data"] for b in rows[0]] == ["/scan sp500", "/scan nasdaq100", "/scan sp500,nasdaq100"]
    assert [b["callback_data"] for b in rows[1]] == [
        "/scan sp500 full 5", "/scan nasdaq100 full 5", "/scan sp500,nasdaq100 full 5"]
    assert all(len(b["callback_data"].encode()) <= 64 for row in rows for b in row)  # Telegram limit
    assert parse_scan_args(rows[1][2]["callback_data"].split()[1:]) == ("sp500,nasdaq100", True, 5)


def test_empty_watchlist_explains_how_to_type_a_code():
    api = _FakeTelegram([[_update(1, "/analyze")]])
    client = _client(api, [], [], watchlist=lambda: [])
    client.poll_once()
    sent = _sent(api, "sendMessage")[0]
    assert "STOCK_LIST is empty" in sent["text"] and "reply_markup" not in sent


def test_button_tap_runs_the_command_and_answers_the_tap():
    api = _FakeTelegram([[_tap(5, "/analyze NVDA full"), _tap(6, "/scan sp500", chat_id=999),
                          _tap(7, "/batch")]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies)
    assert client.poll_once() == 3 and client._offset == 8
    assert [m.content for m in dispatched] == ["/analyze NVDA full"]  # foreign chat and unknown data dropped
    assert dispatched[0].chat_id == "42" and dispatched[0].mentioned
    assert _sent(api, "answerCallbackQuery") == [{"callback_query_id": "cb5", "text": "Running /analyze NVDA full…"}]
    assert replies == [("42", "done")]


def test_old_messages_are_acknowledged_not_executed():
    api = _FakeTelegram([[_update(5, "/scan full", age=7200)]])
    dispatched, replies = [], []
    client = _client(api, dispatched, replies)
    client.poll_once()
    assert dispatched == [] and client._offset == 6
    assert "Skipped an old message" in replies[0][1]


def test_working_notice_only_for_slow_jobs(monkeypatch):
    import bot.platforms.telegram_polling as tp

    monkeypatch.setattr(tp, "WORKING_NOTICE_DELAY_SEC", 0.05)
    replies = []
    slow = TelegramPollingClient("T", "42", session=_FakeTelegram(),
                                 dispatch=lambda m: time.sleep(0.2) or BotResponse.text_response("result"),
                                 reply=lambda chat, text: replies.append(text) or True)
    message = BotMessage(platform="telegram", message_id="1", user_id="7", user_name="me", chat_id="42",
                         chat_type=ChatType.PRIVATE, content="x")
    slow._handle(message)
    assert replies == ["⏳ Working on it…", "result"]

    replies.clear()
    fast = TelegramPollingClient("T", "42", session=_FakeTelegram(), dispatch=lambda m: BotResponse.text_response(""),
                                 reply=lambda chat, text: replies.append(text) or True)
    fast._handle(message)
    time.sleep(0.1)
    assert replies == []  # empty response and no notice


def test_run_forever_backs_off_and_stops_on_bad_token():
    sleeps = []
    api = _FakeTelegram(fail={"getUpdates": _Resp({"ok": False, "description": "Conflict"}, status=409)})
    client = TelegramPollingClient("T", "42", session=api, sleep=lambda s: sleeps.append(s) or (
        len(sleeps) >= 3 and client.stop()))
    client.run_forever()
    assert sleeps == [2, 4, 8]

    bad = TelegramPollingClient("T", "42", session=_FakeTelegram(
        fail={"deleteWebhook": _Resp({"ok": False, "description": "Unauthorized"}, status=401)}),
        sleep=lambda s: pytest.fail("must not retry a bad token"))
    bad.run_forever()
    assert bad._running is False


def test_api_error_does_not_leak_token():
    exc = TelegramApiError("getUpdates", 409, "Conflict")
    assert "TOKEN" not in str(exc) and exc.status == 409


def test_parse_scan_args():
    assert parse_scan_args([]) == ("sp500,nasdaq100", False, 20)
    assert parse_scan_args(["ndx", "full", "5"]) == ("nasdaq100", True, 5)
    assert parse_scan_args(["999"])[2] == 50
    assert ScanCommand().validate_args(["dow30"]) is not None


def test_render_scan_reply():
    row = {"code": "HAL", "type": "1B", "state": "confirmed", "bars_since": 2, "last_close": 30.1,
           "risk_to_invalidation_pct": 4.2}
    result = SimpleNamespace(universes=["sp500"], as_of="2026-11-17", scanned=503, universe_size=503,
                             candidates=[row, row], selected=[row], max_bars_since=5)
    text = render_scan_reply(result, full=False)
    assert "Scanned 503/503; buy points 2; showing top 1." in text
    assert "1. HAL 1B confirmed (2 bars ago, close 30.1, risk 4.2%)" in text and "/scan full" in text


def _routing_dispatcher():
    dispatcher = CommandDispatcher()
    for name in ("market", "scan", "analyze", "ask", "chat"):
        dispatcher.register(SimpleNamespace(name=name, aliases=[], hidden=False, admin_only=False))
    return dispatcher


@pytest.mark.parametrize("parsed,agent_mode,expected", [
    ({"intent": "market"}, False, ("market", [])),
    ({"intent": "scan", "universe": "nasdaq100"}, False, ("scan", ["nasdaq100"])),
    ({"intent": "scan", "universe": "dow"}, False, ("scan", [])),
    ({"intent": "analysis", "codes": ["AAPL", "MSFT"]}, False, ("analyze", ["AAPL"])),
    ({"intent": "analysis", "codes": ["AAPL"], "strategy": "chan"}, True, ("ask", ["AAPL", "chan"])),
    ({"intent": "chat"}, False, None),
    ({"intent": "chat"}, True, ("chat", ["how was it"])),
    ({"intent": "none"}, True, None),
    (None, True, None),
])
def test_route_parsed_intent(parsed, agent_mode, expected):
    route = _routing_dispatcher()._route_parsed_intent(parsed, "how was it", SimpleNamespace(agent_mode=agent_mode))
    assert (None if route is None else (route[0].name, route[1])) == expected


def test_market_intent_does_not_need_agent_mode():
    dispatcher = CommandDispatcher()
    market = SimpleNamespace(name="market", aliases=[], hidden=False, admin_only=False,
                             execute=lambda message, args: BotResponse.text_response("market started"))
    dispatcher.register(market)
    message = BotMessage(platform="telegram", message_id="1", user_id="7", user_name="me", chat_id="42",
                         chat_type=ChatType.PRIVATE, content="how was the snp 500 today?", mentioned=True)
    config = SimpleNamespace(agent_nl_routing=True, agent_mode=False)
    with patch("src.config.get_config", return_value=config), \
            patch.object(CommandDispatcher, "_parse_intent_via_llm_sync", return_value={"intent": "market"}):
        assert dispatcher.dispatch(message).text == "market started"
    assert CommandDispatcher._passes_nl_prefilter("how was the snp 500 today?")
    assert not CommandDispatcher._passes_nl_prefilter("hello")
