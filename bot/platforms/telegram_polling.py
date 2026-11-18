# -*- coding: utf-8 -*-
"""
Telegram long-polling bot client (no public IP, webhook or extra SDK needed).

``getUpdates`` is called with ``timeout=50``: Telegram holds the request open
and answers as soon as a message arrives, so replies start immediately while
the idle loop is one open HTTPS connection. Only ``TELEGRAM_CHAT_ID`` is served.
Replies go out through the existing ``TelegramSender`` (Markdown + 4096 split).
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from bot.models import BotMessage, BotResponse, ChatType

logger = logging.getLogger(__name__)

POLL_TIMEOUT_SEC = 50
MAX_MESSAGE_AGE_SEC = 3600  # older updates (bot was off) are acknowledged, not executed
WORKING_NOTICE_DELAY_SEC = 3.0
_MAX_BACKOFF_SEC = 60

# (command, menu text, needs AGENT_MODE) shown when typing "/" in Telegram
MENU_COMMANDS = (
    ("scan", "Chan buy-point scan of S&P 500 / Nasdaq-100 (pick the index)", False),
    ("market", "Market review for today", False),
    ("analyze", "Analyse a stock (pick from your watchlist)", False),
    ("batch", "Analyse the watchlist (STOCK_LIST)", False),
    ("status", "System and channel status", False),
    ("help", "List all commands", False),
    ("ask", "Agent strategy Q&A on a stock, e.g. /ask AAPL", True),
    ("chat", "Free chat with the agent", True),
    ("research", "Agent deep research on a stock", True),
    ("strategies", "List agent strategies", True),
    ("history", "Agent chat history", True),
)

# A bare menu tap ("/analyze", "/scan") gets buttons instead, since the "/" menu cannot add arguments.
_ANALYZE_TAPS = {"/analyze", "/a", "/analyze full", "/a full"}
_SCAN_TAPS = {"/scan"}
_MAX_PICKER_STOCKS = 20
_SCAN_FULL_PICKS = 5  # "full" runs one LLM analysis per pick
_CALLBACK_PREFIXES = ("/analyze ", "/scan ", "/report ")
_REPORT_CALLBACK = re.compile(r"^/report ([A-Za-z0-9._-]{1,24}) (\d{8})$")
_ANALYZE_WORDS = {"/analyze", "/a"}
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
_SCAN_CHOICES = (("S&P 500", "sp500"), ("Nasdaq-100", "nasdaq100"), ("Both", "sp500,nasdaq100"))


class TelegramPollingClient:
    """Pull updates with long polling and route them through the bot dispatcher."""

    def __init__(
        self,
        bot_token: str,
        allowed_chat_id: str,
        *,
        session: Optional[requests.Session] = None,
        dispatch: Optional[Callable[[BotMessage], BotResponse]] = None,
        reply: Optional[Callable[[str, str], bool]] = None,
        poll_timeout: int = POLL_TIMEOUT_SEC,
        max_workers: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        menu: Optional[List[Dict[str, str]]] = None,
        watchlist: Optional[Callable[[], List[str]]] = None,
        quick_card: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        reports_dir: Optional[Path] = None,
    ):
        if not bot_token or not allowed_chat_id:
            raise ValueError("Telegram polling needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        self._base = f"https://api.telegram.org/bot{bot_token}"
        self._allowed_chat_id = str(allowed_chat_id).strip()
        self._session = session or requests.Session()
        self._dispatch = dispatch or _default_dispatch
        self._reply = reply or _default_reply
        self._poll_timeout = poll_timeout
        self._sleep = sleep
        self._workers = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TelegramBotJob")
        self._offset: Optional[int] = None
        self._failures = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._bot_username = ""
        self._menu = menu
        self._watchlist = watchlist or _default_watchlist
        self._quick_card = quick_card or _default_quick_card
        self._reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR

    # ---------------------------------------------------------------- API --
    def _call(self, method: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        resp = self._session.post(f"{self._base}/{method}", json=payload, timeout=timeout)
        try:
            body = resp.json()
        except ValueError:
            body = {"ok": False, "description": f"HTTP {resp.status_code}"}
        if not body.get("ok"):
            # Never log the URL: it contains the bot token.
            raise TelegramApiError(method, resp.status_code, str(body.get("description", "")))
        return body

    def prepare(self) -> None:
        """Drop any webhook (it blocks getUpdates) and learn the bot username."""
        self._call("deleteWebhook", {"drop_pending_updates": False}, timeout=15)
        me = self._call("getMe", {}, timeout=15).get("result") or {}
        self._bot_username = str(me.get("username") or "")
        logger.info("[Telegram Polling] 已连接 @%s，仅响应 chat_id=%s", self._bot_username, self._allowed_chat_id)
        try:
            commands = self._menu_commands()
            if commands:
                self._call("setMyCommands", {"commands": commands}, timeout=15)
        except Exception as exc:  # noqa: BLE001 - the menu is cosmetic
            logger.warning("[Telegram Polling] setMyCommands 失败: %s", exc)

    def _menu_commands(self) -> List[Dict[str, str]]:
        """Commands for Telegram's "/" menu; agent-only ones only when AGENT_MODE is on."""
        if self._menu is not None:
            return self._menu
        from src.config import get_config

        agent_mode = bool(getattr(get_config(), "agent_mode", False))
        return [{"command": name, "description": text} for name, text, needs_agent in MENU_COMMANDS
                if agent_mode or not needs_agent]

    def poll_once(self) -> int:
        """One ``getUpdates`` round; returns the number of updates handled."""
        payload: Dict[str, Any] = {"timeout": self._poll_timeout,
                                   "allowed_updates": ["message", "callback_query"]}
        if self._offset is not None:
            payload["offset"] = self._offset
        updates = self._call("getUpdates", payload, timeout=self._poll_timeout + 15).get("result") or []
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1  # confirmed on the next getUpdates call
            if "callback_query" in update:
                message = self._parse_callback(update["callback_query"])
            else:
                message = self._parse_update(update)
            if message is not None:
                self._workers.submit(self._handle, message)
        return len(updates)

    # ------------------------------------------------------------ parsing --
    def _parse_update(self, update: Dict[str, Any]) -> Optional[BotMessage]:
        msg = update.get("message")
        if not isinstance(msg, dict):
            return None
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != self._allowed_chat_id:
            logger.warning("[Telegram Polling] 忽略非授权会话 chat_id=%s", chat_id)
            return None
        text = (msg.get("text") or "").strip()
        if not text:
            return None
        age = time.time() - float(msg.get("date") or 0)
        if age > MAX_MESSAGE_AGE_SEC:
            logger.info("[Telegram Polling] 跳过过期消息（%.0f 分钟前）: %s", age / 60, text[:60])
            self._safe_reply(chat_id, f"Skipped an old message ({age / 60:.0f} min ago, sent while I was offline): "
                                      f"\"{text[:80]}\". Send it again if you still want it.")
            return None
        sender = msg.get("from") or {}
        chat_type = {"private": ChatType.PRIVATE, "group": ChatType.GROUP,
                     "supergroup": ChatType.GROUP}.get(chat.get("type"), ChatType.UNKNOWN)
        return BotMessage(
            platform="telegram",
            message_id=str(msg.get("message_id", "")),
            user_id=str(sender.get("id", "")),
            user_name=str(sender.get("username") or sender.get("first_name") or ""),
            chat_id=chat_id,
            chat_type=chat_type,
            content=self._strip_bot_mention(text),
            raw_content=text,
            # The chat is already restricted to the owner, so every message is addressed to the bot.
            mentioned=True,
            timestamp=datetime.fromtimestamp(float(msg.get("date") or time.time())),
            raw_data={"chat_id": chat_id, "message_id": msg.get("message_id")},
        )

    def _strip_bot_mention(self, text: str) -> str:
        """``/help@MyBot args`` -> ``/help args``; ``@MyBot hi`` -> ``hi``."""
        if not self._bot_username:
            return text
        tag = f"@{self._bot_username}"
        head, sep, rest = text.partition(" ")
        if head.lower().endswith(tag.lower()):
            head = head[: -len(tag)]
        return f"{head}{sep}{rest}".strip()

    def _parse_callback(self, query: Dict[str, Any]) -> Optional[BotMessage]:
        """A tapped picker button becomes the command text it carries, e.g. ``/analyze NVDA full``."""
        msg = query.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        data = str(query.get("data") or "").strip()
        if chat_id != self._allowed_chat_id:
            logger.warning("[Telegram Polling] 忽略非授权会话的按钮 chat_id=%s", chat_id)
            return None
        if not data.startswith(_CALLBACK_PREFIXES):
            logger.warning("[Telegram Polling] 忽略未知按钮数据: %s", data[:40])
            return None
        sender = query.get("from") or {}
        return BotMessage(
            platform="telegram",
            message_id=str(msg.get("message_id", "")),
            user_id=str(sender.get("id", "")),
            user_name=str(sender.get("username") or sender.get("first_name") or ""),
            chat_id=chat_id,
            chat_type=ChatType.PRIVATE if chat.get("type") == "private" else ChatType.GROUP,
            content=data,
            raw_content=data,
            mentioned=True,
            timestamp=datetime.now(),
            raw_data={"chat_id": chat_id, "message_id": msg.get("message_id"),
                      "callback_query_id": query.get("id")},
        )

    # ------------------------------------------------------------ pickers --
    def _picker_for(self, text: str) -> Optional[Dict[str, Any]]:
        """Button picker for a bare menu tap, or None to dispatch normally."""
        key = " ".join(text.lower().split())
        if key in _ANALYZE_TAPS:
            return self._analyze_picker()
        if key in _SCAN_TAPS:
            return self._scan_picker()
        return None

    def _analyze_picker(self) -> Dict[str, Any]:
        try:
            codes = [str(c).strip().upper() for c in self._watchlist() if str(c).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Telegram Polling] 读取 STOCK_LIST 失败: %s", exc)
            codes = []
        codes = list(dict.fromkeys(codes))[:_MAX_PICKER_STOCKS]
        if not codes:
            return {"text": "Your STOCK_LIST is empty. Type a code instead, e.g. /analyze AAPL "
                            "(add full for the detailed report)."}
        rows = [[{"text": code, "callback_data": f"/analyze {code}"},
                 {"text": f"{code} · full", "callback_data": f"/analyze {code} full"}] for code in codes]
        return {"text": "Pick a stock (short report, or full for the detailed one).\n"
                        "Other stock: type /analyze TSLA",
                "reply_markup": {"inline_keyboard": rows}}

    @staticmethod
    def _scan_picker() -> Dict[str, Any]:
        plain = [{"text": label, "callback_data": f"/scan {value}"} for label, value in _SCAN_CHOICES]
        full = [{"text": f"{label} · full", "callback_data": f"/scan {value} full {_SCAN_FULL_PICKS}"}
                for label, value in _SCAN_CHOICES]
        return {"text": "Chan scan: pick the index.\n"
                        f"Plain lists the top 20 buy points (~1 min, no LLM). "
                        f"full also runs the LLM analysis on the top {_SCAN_FULL_PICKS}.",
                "reply_markup": {"inline_keyboard": [plain, full]}}

    def _send_picker(self, chat_id: str, picker: Dict[str, Any]) -> None:
        try:
            self._call("sendMessage", {"chat_id": chat_id, **picker}, timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.error("[Telegram Polling] 发送按钮失败: %s", exc)
            self._safe_reply(chat_id, picker["text"])

    def _answer_callback(self, callback_id: Any, text: str) -> None:
        try:
            self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text[:190]}, timeout=10)
        except Exception as exc:  # noqa: BLE001 - the toast is cosmetic
            logger.debug("[Telegram Polling] answerCallbackQuery 失败: %s", exc)

    # ----------------------------------------------------------- handling --
    def _handle(self, message: BotMessage) -> None:
        callback_id = (message.raw_data or {}).get("callback_query_id")
        if callback_id:
            toast = ("Opening the full report…" if message.content.startswith("/report ")
                     else f"Running {message.content}…")
            self._answer_callback(callback_id, toast)
        else:
            picker = self._picker_for(message.content)
            if picker is not None:
                self._send_picker(message.chat_id, picker)
                return
        if message.content.startswith("/report "):
            self._send_saved_report(message.chat_id, message.content)
            return
        self._send_typing(message.chat_id)
        notice = threading.Timer(WORKING_NOTICE_DELAY_SEC, self._safe_reply,
                                 args=(message.chat_id, "⏳ Working on it…"))
        notice.daemon = True
        notice.start()
        try:
            response = self._dispatch(message)
        except Exception as exc:  # noqa: BLE001 - one bad message must not stop the bot
            logger.exception("[Telegram Polling] 处理消息失败: %s", exc)
            response = BotResponse.error_response(f"处理失败: {str(exc)[:100]}")
        finally:
            notice.cancel()
        text = (getattr(response, "text", "") or "").strip()
        if self._send_quick_card(message, response):
            return  # the quick card replaces the "analysis submitted" notice
        if text:
            self._safe_reply(message.chat_id, text)

    # ------------------------------------------------------- verdict cards --
    def _send_quick_card(self, message: BotMessage, response: Any) -> bool:
        """After an accepted /analyze, send the instant Chan card (no LLM)."""
        words = message.content.split()
        if not words or words[0].lower() not in _ANALYZE_WORDS:
            return False
        code = str((getattr(response, "extra", None) or {}).get("stock_code") or "").strip()
        if not code:
            return False  # rejected request: the error text is the reply
        try:
            card = self._quick_card(code)
        except Exception as exc:  # noqa: BLE001 - the plain notice still goes out
            logger.warning("[Telegram Polling] 快速缠论卡片失败 %s: %s", code, exc)
            return False
        return bool(card) and self._send_card(message.chat_id, card)

    def _send_card(self, chat_id: str, card: Dict[str, Any]) -> bool:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": card["text"], "parse_mode": "HTML",
                                   "disable_web_page_preview": True}
        if card.get("reply_markup"):
            payload["reply_markup"] = card["reply_markup"]
        try:
            self._call("sendMessage", payload, timeout=15)
            logger.info("[Telegram Polling] 已发送快速缠论卡片")
            return True
        except TelegramApiError as exc:
            if exc.status != 400 or not card.get("plain"):
                logger.error("[Telegram Polling] 卡片发送失败: %s", exc)
                return False
        except Exception as exc:  # noqa: BLE001
            logger.error("[Telegram Polling] 卡片发送失败: %s", exc)
            return False
        plain = {key: value for key, value in payload.items() if key != "parse_mode"}
        plain["text"] = card["plain"]  # HTML rejected: send the same card as plain text
        try:
            self._call("sendMessage", plain, timeout=15)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[Telegram Polling] 卡片纯文本发送失败: %s", exc)
            return False

    def _send_saved_report(self, chat_id: str, command: str) -> None:
        """📄 button: resend the report file the analysis saved (no new LLM call)."""
        match = _REPORT_CALLBACK.match(command.strip())
        if not match:
            self._safe_reply(chat_id, "That report link is not valid.")
            return
        code, date = match.groups()
        path = self._reports_dir / f"report_{date}_{code}.md"
        if not path.is_file():
            self._safe_reply(chat_id, f"The {code} report from {date} is no longer on disk. "
                                      f"Tap 🔁 Refresh to run the analysis again.")
            return
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("[Telegram Polling] 读取报告失败 %s: %s", path.name, exc)
            self._safe_reply(chat_id, "Could not read that report file.")
            return
        self._safe_reply(chat_id, text or "That report is empty.")

    def _send_typing(self, chat_id: str) -> None:
        try:
            self._call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Telegram Polling] sendChatAction 失败: %s", exc)

    def _safe_reply(self, chat_id: str, text: str) -> None:
        try:
            if not self._reply(chat_id, text):
                logger.error("[Telegram Polling] 回复发送失败")
        except Exception as exc:  # noqa: BLE001
            logger.error("[Telegram Polling] 回复发送异常: %s", exc)

    # -------------------------------------------------------------- loop --
    def run_forever(self) -> None:
        self._running = True
        prepared = False
        while self._running:
            try:
                if not prepared:
                    self.prepare()
                    prepared = True
                self.poll_once()
                self._failures = 0
            except Exception as exc:  # noqa: BLE001 - network errors, 409 conflicts, 5xx
                self._failures += 1
                delay = min(_MAX_BACKOFF_SEC, 2 ** min(self._failures, 6))
                if isinstance(exc, TelegramApiError) and exc.status == 409:
                    logger.warning("[Telegram Polling] 409：另一个进程也在拉取此 Bot 的消息"
                                   "（只能有一个，例如本地和 Docker 不要同时开启），%ss 后重试", delay)
                elif isinstance(exc, TelegramApiError) and exc.status == 401:
                    logger.error("[Telegram Polling] 401：TELEGRAM_BOT_TOKEN 无效，停止拉取")
                    self._running = False
                    break
                else:
                    logger.warning("[Telegram Polling] 拉取失败（第 %d 次）: %s，%ss 后重试", self._failures, exc, delay)
                self._sleep(delay)

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("[Telegram Polling] 客户端已在运行")
            return
        self._thread = threading.Thread(target=self.run_forever, daemon=True, name="TelegramPollingClient")
        self._thread.start()

    def stop(self) -> None:
        self._running = False


class TelegramApiError(RuntimeError):
    def __init__(self, method: str, status: int, description: str):
        super().__init__(f"Telegram {method} failed: HTTP {status} {description}")
        self.status = status


def _default_dispatch(message: BotMessage) -> BotResponse:
    from bot.dispatcher import get_dispatcher

    return get_dispatcher().dispatch(message)


def _default_quick_card(code: str) -> Optional[Dict[str, Any]]:
    from src.services.verdict_card import build_quick_card

    return build_quick_card(code)


def _default_watchlist() -> List[str]:
    from src.config import get_config

    config = get_config()
    config.refresh_stock_list()
    return list(config.stock_list or [])


def _default_reply(chat_id: str, text: str) -> bool:
    from src.config import get_config
    from src.notification_sender.telegram_sender import TelegramSender

    return TelegramSender(get_config()).send_to_telegram(text, chat_id=chat_id)


_client: Optional[TelegramPollingClient] = None


def start_telegram_polling_background(config: Any = None) -> bool:
    """Start the global polling client; returns False when not configured."""
    global _client
    if _client is not None:
        return True
    if config is None:
        from src.config import get_config

        config = get_config()
    token = (getattr(config, "telegram_bot_token", None) or "").strip()
    chat_id = (getattr(config, "telegram_chat_id", None) or "").strip()
    if not token or not chat_id:
        logger.warning("[Telegram Polling] 需要同时配置 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID")
        return False
    _client = TelegramPollingClient(token, chat_id)
    _client.start_background()
    return True


__all__: List[str] = ["TelegramPollingClient", "TelegramApiError", "start_telegram_polling_background"]
