# -*- coding: utf-8 -*-
"""
/scan: Chan daily pre-screen of S&P 500 / Nasdaq-100 constituents (src/chan/scan.py).

``/scan`` posts the ranked buy-point picks; ``/scan full`` also runs the normal
LLM analysis on the picks (same as ``python main.py --scan``). Runs in the
background and replies to the chat that asked.
"""

import logging
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from bot.commands.base import BotCommand
from bot.models import BotMessage, BotResponse

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSES = "sp500,nasdaq100"
DEFAULT_MAX_PICKS = 20
DEFAULT_MAX_AGE = 5
_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()


def parse_scan_args(args: List[str]) -> Tuple[str, bool, int]:
    """``[universes] [full] [N]`` in any order -> (universes, full, max_picks)."""
    from src.chan.scan import parse_scan_universes

    universes, full, max_picks = DEFAULT_UNIVERSES, False, DEFAULT_MAX_PICKS
    for arg in args:
        token = arg.strip().lower()
        if token in {"full", "analyze", "analyse", "完整", "分析"}:
            full = True
        elif token.isdigit():
            max_picks = max(1, min(int(token), 50))
        elif token:
            universes = ",".join(parse_scan_universes(token))  # ValueError on unknown names
    return universes, full, max_picks


def render_scan_reply(result, full: bool) -> str:
    """Short chat summary of a ScanResult (the full table stays in reports/)."""
    picks = result.selected
    lines = [
        f"📈 *Chan scan* {','.join(result.universes)} — data {result.as_of or '-'}",
        f"Scanned {result.scanned}/{result.universe_size}; buy points {len(result.candidates)}; "
        f"showing top {len(picks)}.",
    ]
    if not picks:
        lines.append("No confirmed/candidate buy points in the last "
                     f"{result.max_bars_since} bars.")
    for i, row in enumerate(picks, 1):
        risk = row.get("risk_to_invalidation_pct")
        risk_text = f", risk {risk}%" if risk is not None else ""
        lines.append(f"{i}. {row['code']} {row['type']} {row['state']} "
                     f"({row['bars_since']} bars ago, close {row.get('last_close')}{risk_text})")
    lines.append("")
    lines.append("Running LLM analysis on these now; reports follow." if full and picks
                 else "Send `/scan full` to run the LLM analysis on the picks.")
    return "\n".join(lines)


class ScanCommand(BotCommand):

    @property
    def name(self) -> str:
        return "scan"

    @property
    def aliases(self) -> List[str]:
        return ["扫描"]

    @property
    def description(self) -> str:
        return "缠论日线扫描 S&P 500 / Nasdaq-100 买点（full 时再做 LLM 分析）"

    @property
    def usage(self) -> str:
        return "/scan [sp500|nasdaq100] [full] [N]"

    def validate_args(self, args: List[str]) -> Optional[str]:
        try:
            parse_scan_args(args)
        except ValueError as exc:
            return str(exc)
        return None

    def execute(self, message: BotMessage, args: List[str]) -> BotResponse:
        universes, full, max_picks = parse_scan_args(args)
        if not _lock.acquire(blocking=False):
            return BotResponse.text_response("⚠️ A scan is already running; I'll post it when it finishes.")
        thread = threading.Thread(target=self._run, args=(message, universes, full, max_picks),
                                  daemon=True, name="BotChanScan")
        try:
            thread.start()
        except Exception as exc:  # noqa: BLE001
            _lock.release()
            return BotResponse.error_response(f"扫描启动失败: {exc}")
        return BotResponse.text_response(
            f"🔎 Scanning {universes} (about 1 minute)"
            + (f", then analysing up to {max_picks} picks." if full else ".")
        )

    def _run(self, message: BotMessage, universes: str, full: bool, max_picks: int) -> None:
        try:
            from src.chan.scan import run_chan_scan
            from src.config import get_config
            from src.notification import NotificationService

            config = get_config()
            result = run_chan_scan(
                universes,
                max_bars_since=DEFAULT_MAX_AGE,
                max_candidates=max_picks,
                report_dir=_ROOT / "reports",
                cache_dir=_ROOT / "data" / "chan_scan",
                language=getattr(config, "report_language", "zh"),
            )
            NotificationService(source_message=message).send_to_context(render_scan_reply(result, full))
            if full and result.selected_codes:
                from main import StockAnalysisPipeline

                pipeline = StockAnalysisPipeline(config=config, source_message=message,
                                                 query_id=uuid.uuid4().hex, query_source="bot")
                pipeline.run(stock_codes=list(result.selected_codes), dry_run=False, send_notification=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ScanCommand] 扫描失败: %s", exc)
            try:
                from src.notification import NotificationService

                NotificationService(source_message=message).send_to_context(f"❌ Scan failed: {str(exc)[:200]}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            _lock.release()
