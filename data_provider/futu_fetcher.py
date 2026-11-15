# -*- coding: utf-8 -*-
"""Futu / moomoo OpenD market-data fetcher.

Read-only HK/US quote and K-line adapter for DSA. Trading APIs are
intentionally not imported or exposed here.
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .opend_sdk import load_opend_sdk, opend_sdk_brand
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float
from .us_index_mapping import is_us_stock_code

logger = logging.getLogger(__name__)

# Futu get_market_snapshot returns update_time as a naive "yyyy-MM-dd HH:mm:ss"
# string; per the official docs HK and A-share quotes use Beijing time (UTC+8).
# DSA's _parse_realtime_timestamp treats naive values as UTC, so we attach the
# market-local offset here to keep provider_timestamp / stale_seconds correct.
_HK_UPDATE_TIME_OFFSET = timezone(timedelta(hours=8))

# futu-api's synchronous OpenQuoteContext constructor retries forever while
# OpenD is unreachable, so connect asynchronously and wait a bounded time for
# the protocol handshake (status READY), then skip Futu for a cooldown window.
_CONNECT_TIMEOUT_SECONDS = 5.0
_CONNECT_POLL_SECONDS = 0.05
_CONNECT_RETRY_COOLDOWN_SECONDS = 300.0

# Futu/moomoo caps one request_history_kline page at 1000 rows and pages with
# page_req_key. Keep the loop bounded: one stock/timeframe should not fan out
# into an unbounded history request (historical-kline quota is per stock).
_MAX_HISTORY_PAGES = 5
# OpenD pages run oldest-first, so a window larger than the page cap would drop
# the newest bars. Clamp the default intraday window (calendar days) to what the
# cap can hold: about 13 half-hour bars per US session.
_MAX_INTRADAY_WINDOW_DAYS = {"30m": 365, "60m": 730}
_KLINE_KTYPE_NAMES = {
    "1d": "K_DAY",
    "1w": "K_WEEK",
    "60m": "K_60M",
    "30m": "K_30M",
}

# US snapshot/kline timestamps are US Eastern local time (DST-aware); HK and
# A-share timestamps are Beijing time (UTC+8).
_US_EASTERN_FALLBACK = timezone(timedelta(hours=-5), "EST")


@lru_cache(maxsize=1)
def _us_eastern_timezone():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - Windows without tzdata: keep a fixed offset
        logger.warning("[Futu] America/New_York 时区不可用，回退固定 UTC-5")
        return _US_EASTERN_FALLBACK


def _first_env_value(*names: str) -> Optional[str]:
    """Return the first non-empty env value; the Futu name wins when both set."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _provider_timestamp(value: Any, market: str = "hk") -> Optional[str]:
    """Normalize an OpenD timestamp to an offset-aware ISO string."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        tzinfo = _us_eastern_timezone() if market == "us" else _HK_UPDATE_TIME_OFFSET
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed.isoformat()


def _hk_provider_timestamp(value: Any) -> Optional[str]:
    """Normalize Futu HK snapshot update_time to an offset-aware ISO string."""
    return _provider_timestamp(value, market="hk")


def _hk_symbol(stock_code: str) -> Optional[str]:
    code = (stock_code or "").strip().upper()
    if code.endswith(".HK"):
        digits = code[:-3]
    elif code.startswith("HK"):
        digits = code[2:]
    elif code.isdigit() and 1 <= len(code) <= 5:
        digits = code
    else:
        return None
    if not digits.isdigit() or not 1 <= len(digits) <= 5:
        return None
    return f"HK.{digits.zfill(5)}"


def _opend_symbol(stock_code: str) -> Optional[str]:
    """Map a DSA code to an OpenD symbol, or None for unsupported markets.

    HK keeps the existing ``HK.00700`` form; US tickers become ``US.AAPL``.
    A-share and every other market fail closed instead of guessing an exchange.
    """
    hk_symbol = _hk_symbol(stock_code)
    if hk_symbol:
        return hk_symbol
    code = (stock_code or "").strip().upper()
    if is_us_stock_code(code):
        return f"US.{code}"
    return None


def _opend_market(stock_code: str) -> str:
    """Return the OpenD market label for a supported stock code."""
    if _hk_symbol(stock_code):
        return "hk"
    code = (stock_code or "").strip().upper()
    if is_us_stock_code(code):
        return "us"
    return "hk"


class FutuFetcher(BaseFetcher):
    """Futu OpenD adapter for DSA's existing provider contract."""

    name = "FutuFetcher"
    priority = int(os.getenv("FUTU_PRIORITY", "2"))
    allow_empty_daily_data = False
    SUPPORTED_TIMEFRAMES = frozenset(_KLINE_KTYPE_NAMES)

    def __init__(self) -> None:
        self._ctx = None
        self._ctx_lock = threading.Lock()
        self._available: Optional[bool] = None
        self._retry_after = 0.0
        config = None
        try:
            from src.config import get_config
            config = get_config()
        except Exception:
            pass
        env_host = _first_env_value("FUTU_OPEND_HOST", "MOOMOO_OPEND_HOST")
        env_port = _first_env_value("FUTU_OPEND_PORT", "MOOMOO_OPEND_PORT")
        self._host = (getattr(config, "futu_opend_host", None) or env_host or "127.0.0.1").strip() or "127.0.0.1"
        try:
            self._port = int(getattr(config, "futu_opend_port", None) or env_port or 11111)
        except (TypeError, ValueError):
            self._port = 11111

    @staticmethod
    def has_configured_endpoint() -> bool:
        if _first_env_value("FUTU_OPEND_HOST", "MOOMOO_OPEND_HOST"):
            return True
        try:
            from src.config import get_config
            return bool((getattr(get_config(), "futu_opend_host", None) or "").strip())
        except Exception:
            return False

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            load_opend_sdk()
            self._available = self.has_configured_endpoint()
        except ImportError:
            self._available = False
            logger.warning("[Futu] moomoo-api / futu-api 未安装")
        return self._available

    def _in_connect_cooldown(self) -> bool:
        return self._ctx is None and time.monotonic() < self._retry_after

    def is_available_for_request(self, capability: str = "") -> bool:
        return self._is_available() and not self._in_connect_cooldown()

    def _get_ctx(self):
        if not self._is_available():
            return None
        if self._ctx is not None:
            return self._ctx
        if self._in_connect_cooldown():
            return None
        with self._ctx_lock:
            if self._ctx is None and not self._in_connect_cooldown():
                self._ctx = self._connect()
        return self._ctx

    def _connect(self):
        """Open a quote context, or return None within a bounded time."""
        ctx = None
        try:
            sdk = load_opend_sdk()
            ctx = sdk.OpenQuoteContext(host=self._host, port=self._port, is_async_connect=True)
            deadline = time.monotonic() + _CONNECT_TIMEOUT_SECONDS
            while ctx.status != sdk.ContextStatus.READY:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"OpenD 未在 {_CONNECT_TIMEOUT_SECONDS:g}s 内就绪 (status={ctx.status})"
                    )
                time.sleep(_CONNECT_POLL_SECONDS)
            logger.info(
                "[Futu] OpenQuoteContext 初始化成功 (%s): %s:%s",
                opend_sdk_brand(sdk),
                self._host,
                self._port,
            )
            return ctx
        except Exception as exc:
            # close() turns off the SDK's background auto-reconnect.
            if ctx is not None:
                self._close_ctx(ctx)
            self._retry_after = time.monotonic() + _CONNECT_RETRY_COOLDOWN_SECONDS
            logger.warning(
                "[Futu] 连接 OpenD %s:%s 失败，%d 分钟内跳过 Futu 并回退其他数据源: %s",
                self._host,
                self._port,
                _CONNECT_RETRY_COOLDOWN_SECONDS // 60,
                exc,
            )
            return None

    @staticmethod
    def _close_ctx(ctx: Any) -> None:
        try:
            ctx.close()
        except Exception:
            pass

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        symbol = _opend_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ret, data = ctx.get_market_snapshot([symbol])
            if ret == 0 and data is not None and not data.empty:
                return str(data.iloc[0].get("name") or "").strip() or None
        except Exception as exc:
            logger.debug("[Futu] stock name failed(%s): %s", symbol, exc)
        return None

    def get_stock_basicinfo(self, stock_code: str):
        """Return static security metadata for one HK symbol."""
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_stock_basicinfo(ft.Market.HK, code_list=[symbol])
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] static info failed(%s): %s", symbol, exc)
            return None

    def request_trading_days(self, start_date: str, end_date: str):
        """Return HK trading days in the requested range."""
        ctx = self._get_ctx()
        if ctx is None:
            return []
        try:
            ft = load_opend_sdk()
            ret, data = ctx.request_trading_days(
                ft.TradeDateMarket.HK, start=start_date, end=end_date
            )
            return data if ret == ft.RET_OK else []
        except Exception as exc:
            logger.warning("[Futu] trading days failed: %s", exc)
            return []

    def get_trading_days(self, start_date: str, end_date: str):
        """Return normalized HK trading-day records."""
        rows = self.request_trading_days(start_date, end_date)
        return [
            {
                "date": row.get("time"),
                "trade_date_type": row.get("trade_date_type"),
            }
            for row in rows
            if isinstance(row, dict) and row.get("time")
        ]

    def get_company_profile(self, stock_code: str):
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_company_profile(symbol)
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] company profile failed(%s): %s", symbol, exc)
            return None

    def get_financials_statements(self, stock_code: str, statement_type: Optional[int] = None, num: int = 8):
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_financials_statements(
                symbol, statement_type=statement_type, num=num
            )
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] financial statements failed(%s): %s", symbol, exc)
            return None

    def get_corporate_actions_dividends(self, stock_code: str):
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_corporate_actions_dividends(symbol)
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] dividends failed(%s): %s", symbol, exc)
            return None

    def get_corporate_actions_stock_splits(self, stock_code: str, num: int = 50):
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_corporate_actions_stock_splits(symbol, num=num)
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] stock splits failed(%s): %s", symbol, exc)
            return None

    def get_capital_flow(self, stock_code: str, **kwargs):
        symbol = _hk_symbol(stock_code)
        ctx = self._get_ctx()
        if not symbol or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_capital_flow(symbol, **kwargs)
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] capital flow failed(%s): %s", symbol, exc)
            return None

    def get_owner_plate(self, stock_codes):
        symbols = [_hk_symbol(code) for code in stock_codes]
        symbols = [symbol for symbol in symbols if symbol]
        ctx = self._get_ctx()
        if not symbols or ctx is None:
            return None
        try:
            ft = load_opend_sdk()
            ret, data = ctx.get_owner_plate(symbols)
            return data if ret == ft.RET_OK else None
        except Exception as exc:
            logger.warning("[Futu] owner plate failed: %s", exc)
            return None

    def close(self) -> None:
        if self._ctx is not None:
            self._close_ctx(self._ctx)
            self._ctx = None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        symbol = _opend_symbol(stock_code)
        if not symbol:
            return None
        market = _opend_market(stock_code)
        ctx = self._get_ctx()
        if ctx is None:
            return None
        try:
            ret, rows = ctx.get_market_snapshot([symbol])
            if ret != 0 or rows is None or rows.empty:
                return None
            q = rows.iloc[0]
            price = safe_float(q.get("last_price"))
            if price is None or price <= 0:
                return None
            prev = safe_float(q.get("prev_close_price"))
            change_pct = safe_float(q.get("change_rate"))
            volume = int(q.get("volume") or 0)
            turnover = safe_float(q.get("turnover"))
            turnover_rate = safe_float(q.get("turnover_rate"))
            return UnifiedRealtimeQuote(
                code=stock_code,
                name=str(q.get("name") or ""),
                source=RealtimeSource.FUTU,
                price=price,
                change_pct=change_pct,
                change_amount=round(price - prev, 4) if prev else None,
                volume=volume or None,
                amount=turnover,
                turnover_rate=turnover_rate,
                volume_ratio=safe_float(q.get("volume_ratio")),
                amplitude=safe_float(q.get("amplitude")),
                open_price=safe_float(q.get("open_price")),
                high=safe_float(q.get("high_price")),
                low=safe_float(q.get("low_price")),
                pre_close=prev,
                pe_ratio=safe_float(q.get("pe_ttm_ratio")) or safe_float(q.get("pe_ratio")),
                pb_ratio=safe_float(q.get("pb_ratio")),
                total_mv=safe_float(q.get("total_market_val")),
                circ_mv=safe_float(q.get("circular_market_val")),
                provider_timestamp=_provider_timestamp(q.get("update_time"), market=market),
            )
        except Exception as exc:
            logger.warning("[Futu] realtime quote failed(%s): %s", symbol, exc)
            return None

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Existing daily path: single page, unadjusted prices, K_DAY. Kept
        # intentionally unchanged; multi-timeframe QFQ lives in get_kline_data.
        symbol = _opend_symbol(stock_code)
        if not symbol:
            return pd.DataFrame()
        ctx = self._get_ctx()
        if ctx is None:
            return pd.DataFrame()
        try:
            ft = load_opend_sdk()
            ret, data, _ = ctx.request_history_kline(
                symbol,
                start=start_date,
                end=end_date,
                ktype=ft.KLType.K_DAY,
                autype=ft.AuType.NONE,
                max_count=1000,
            )
            if ret != ft.RET_OK or data is None or data.empty:
                return pd.DataFrame()
            return data.rename(columns={"time_key": "date", "turnover": "amount"})
        except Exception as exc:
            logger.warning("[Futu] daily data failed(%s): %s", symbol, exc)
            return pd.DataFrame()

    def get_kline_data(
        self,
        stock_code: str,
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch forward-adjusted multi-timeframe K-lines from OpenD."""
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise DataFetchError(f"[{self.name}] 不支持的 timeframe: {timeframe!r}")
        symbol = _opend_symbol(stock_code)
        if not symbol:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        ctx = self._get_ctx()
        if ctx is None:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        ft = load_opend_sdk()
        ktype = getattr(ft.KLType, _KLINE_KTYPE_NAMES[timeframe])
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            window_days = max(int(days), 1) * 2
            window_days = min(window_days, _MAX_INTRADAY_WINDOW_DAYS.get(timeframe, window_days))
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=window_days)
            start_date = start_dt.strftime("%Y-%m-%d")

        frames = []
        page_req_key = None
        for _ in range(_MAX_HISTORY_PAGES):
            try:
                ret, data, page_req_key = ctx.request_history_kline(
                    symbol,
                    start=start_date,
                    end=end_date,
                    ktype=ktype,
                    autype=ft.AuType.QFQ,
                    max_count=1000,
                    page_req_key=page_req_key,
                )
            except Exception as exc:
                logger.warning("[Futu] kline failed(%s, %s): %s", symbol, timeframe, exc)
                if not frames:
                    raise DataFetchError(f"[{self.name}] {symbol} {timeframe}: {exc}") from exc
                break
            if ret != ft.RET_OK:
                if not frames:
                    raise DataFetchError(f"[{self.name}] {symbol} {timeframe}: ret={ret}")
                logger.warning("[Futu] kline page failed(%s, %s): ret=%s", symbol, timeframe, ret)
                break
            if data is not None and not data.empty:
                frames.append(data)
            if not page_req_key:
                break
        else:
            if page_req_key:
                logger.warning(
                    "[Futu] kline %s %s 超过 %s 页上限，最新的 K 线可能缺失，请缩小时间窗口",
                    symbol, timeframe, _MAX_HISTORY_PAGES,
                )

        if not frames:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        return self._normalize_kline_data(pd.concat(frames, ignore_index=True), timeframe)

    @staticmethod
    def _normalize_kline_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Return exactly STANDARD_COLUMNS, market-local naive datetimes."""
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        out = df.copy().rename(columns={"time_key": "date", "turnover": "amount"})
        if "date" not in out.columns:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if getattr(out["date"].dt, "tz", None) is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        if timeframe in {"1d", "1w"}:
            out["date"] = out["date"].dt.normalize()
        for col in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "pct_chg" not in out.columns and "close" in out.columns:
            out["pct_chg"] = out["close"].pct_change() * 100
        for col in STANDARD_COLUMNS:
            if col not in out.columns:
                out[col] = None
        out = out[STANDARD_COLUMNS].dropna(subset=["date", "close"])
        out = out.sort_values("date", kind="stable")
        out = out.drop_duplicates(subset=["date"], keep="last")
        return out.reset_index(drop=True)

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        rename = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        out = out.rename(columns=rename)
        if "pct_chg" not in out.columns:
            out["pct_chg"] = out["close"].pct_change() * 100
        for col in STANDARD_COLUMNS:
            if col not in out.columns:
                out[col] = None
        return out[STANDARD_COLUMNS]