# -*- coding: utf-8 -*-
"""
Telegram long-polling bot client (no public IP, webhook or extra SDK needed).

``getUpdates`` is called with ``timeout=50``: Telegram holds the request open
and answers as soon as a message arrives, so replies start immediately while
the idle loop is one open HTTPS connection. Only ``TELEGRAM_CHAT_ID`` is served.
Replies go out through the existing ``TelegramSender`` (Markdown + 4096 split).
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
    ("scan", "Chan buy-point scan of S&P 500 / Nasdaq-100 (add 'full' for LLM analysis)", False),
    ("market", "Market review for today", False),
    ("analyze", "Analyse one stock, e.g. /analyze AAPL", False),
    ("batch", "Analyse the watchlist (STOCK_LIST)", False),
    ("status", "System and channel status", False),
    ("help", "List all commands", False),
    ("ask", "Agent strategy Q&A on a stock, e.g. /ask AAPL", True),
    ("chat", "Free chat with the agent", True),
    ("research", "Agent deep research on a stock", True),
    ("strategies", "List agent strategies", True),
    ("history", "Agent chat history", True),
)


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
        payload: Dict[str, Any] = {"timeout": self._poll_timeout, "allowed_updates": ["message"]}
        if self._offset is not None:
            payload["offset"] = self._offset
        updates = self._call("getUpdates", payload, timeout=self._poll_timeout + 15).get("result") or []
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1  # confirmed on the next getUpdates call
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

    # ----------------------------------------------------------- handling --
    def _handle(self, message: BotMessage) -> None:
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
        if text:
            self._safe_reply(message.chat_id, text)

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
