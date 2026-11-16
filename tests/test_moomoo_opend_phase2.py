# -*- coding: utf-8 -*-
"""Phase 2 tests: moomoo/Futu HK+US market data and multi-timeframe K-lines.

All SDK interaction is an in-process fake module; no network or OpenD needed.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_provider import opend_sdk
from data_provider.base import (
    DataFetcherManager,
    DataFetchError,
    STANDARD_COLUMNS,
)
from data_provider.futu_fetcher import (
    FutuFetcher,
    _KLINE_KTYPE_NAMES,
    _MAX_HISTORY_PAGES,
    _opend_market,
    _opend_symbol,
    _provider_timestamp,
)
from src.config import Config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fake SDK / context
# ---------------------------------------------------------------------------


class _FakeContextStatus:
    READY = "READY"
    WAIT_RECONNECT = "WAIT_RECONNECT"


class _FakeKLType:
    K_DAY = "K_DAY"
    K_WEEK = "K_WEEK"
    K_60M = "K_60M"
    K_30M = "K_30M"


class _FakeAuType:
    NONE = "NONE"
    QFQ = "QFQ"


class _FakeQuoteContext:
    def __init__(self, response_factory=None):
        self.calls = []
        self._response_factory = response_factory

    def request_history_kline(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        if self._response_factory is None:
            return 0, pd.DataFrame(), None
        return self._response_factory(len(self.calls), kwargs)

    def close(self):
        pass


def _fake_sdk_module(name: str = "futu") -> types.ModuleType:
    module = types.ModuleType(name)
    module.ContextStatus = _FakeContextStatus
    module.KLType = _FakeKLType
    module.AuType = _FakeAuType
    module.RET_OK = 0
    module.OpenQuoteContext = _FakeQuoteContext
    return module


def _install_fake_sdk(testcase, module=None, *, name="futu"):
    module = module or _fake_sdk_module(name)
    patcher = patch.dict(
        sys.modules,
        {name: module, "moomoo": None, "futu": module if name == "futu" else None},
    )
    patcher.start()
    if hasattr(testcase, "addCleanup"):
        testcase.addCleanup(patcher.stop)
    else:
        testcase._patchers = getattr(testcase, "_patchers", [])
        testcase._patchers.append(patcher)
    opend_sdk.reset_opend_sdk_cache()
    if hasattr(testcase, "addCleanup"):
        testcase.addCleanup(opend_sdk.reset_opend_sdk_cache)
    return module


def _make_fetcher(testcase, ctx=None, module=None):
    _install_fake_sdk(testcase, module)
    env = {"FUTU_OPEND_HOST": "127.0.0.1", "FUTU_OPEND_PORT": "11111"}
    with patch.dict(os.environ, env, clear=False), patch(
        "src.config.get_config", side_effect=RuntimeError("no config")
    ):
        fetcher = FutuFetcher()
    fetcher._available = True
    fetcher._ctx = ctx
    return fetcher


def _rows(*timestamps, closes=None):
    closes = closes or [100.0 + index for index in range(len(timestamps))]
    return pd.DataFrame(
        {
            "time_key": list(timestamps),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000] * len(timestamps),
            "turnover": [value * 1000 for value in closes],
        }
    )


# ---------------------------------------------------------------------------
# SDK loader
# ---------------------------------------------------------------------------


class TestOpendsdkLoader:
    def teardown_method(self):
        opend_sdk.reset_opend_sdk_cache()

    def test_prefers_moomoo_when_both_are_importable(self):
        moomoo = _fake_sdk_module("moomoo")
        futu = _fake_sdk_module("futu")
        with patch.dict(sys.modules, {"moomoo": moomoo, "futu": futu}):
            opend_sdk.reset_opend_sdk_cache()
            assert opend_sdk.load_opend_sdk() is moomoo
            assert opend_sdk.opend_sdk_brand() == "moomoo"

    def test_falls_back_to_futu(self):
        futu = _fake_sdk_module("futu")
        with patch.dict(sys.modules, {"moomoo": None, "futu": futu}):
            opend_sdk.reset_opend_sdk_cache()
            assert opend_sdk.load_opend_sdk() is futu
            assert opend_sdk.opend_sdk_brand() == "futu"

    def test_raises_import_error_when_neither_is_installed(self):
        with patch.dict(sys.modules, {"moomoo": None, "futu": None}):
            opend_sdk.reset_opend_sdk_cache()
            with pytest.raises(ImportError):
                opend_sdk.load_opend_sdk()

    def test_cache_invalidates_when_injected_module_is_replaced(self):
        first = _fake_sdk_module("futu")
        second = _fake_sdk_module("futu")
        with patch.dict(sys.modules, {"moomoo": None, "futu": first}):
            opend_sdk.reset_opend_sdk_cache()
            assert opend_sdk.load_opend_sdk() is first
            sys.modules["futu"] = second
            assert opend_sdk.load_opend_sdk() is second


# ---------------------------------------------------------------------------
# Config aliases and endpoint detection
# ---------------------------------------------------------------------------


class TestEndpointAliases:
    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    @patch.object(Config, "_parse_stock_email_groups", return_value=[])
    def test_moomoo_aliases_are_accepted(self, _groups, _yaml, _setup_env):
        with patch.dict(
            os.environ,
            {"MOOMOO_OPEND_HOST": "10.0.0.5", "MOOMOO_OPEND_PORT": "22222"},
            clear=True,
        ):
            config = Config._load_from_env()
        assert config.futu_opend_host == "10.0.0.5"
        assert config.futu_opend_port == 22222

    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    @patch.object(Config, "_parse_stock_email_groups", return_value=[])
    def test_futu_values_win_when_both_are_set(self, _groups, _yaml, _setup_env):
        with patch.dict(
            os.environ,
            {
                "FUTU_OPEND_HOST": "127.0.0.1",
                "FUTU_OPEND_PORT": "11111",
                "MOOMOO_OPEND_HOST": "10.0.0.5",
                "MOOMOO_OPEND_PORT": "22222",
            },
            clear=True,
        ):
            config = Config._load_from_env()
        assert config.futu_opend_host == "127.0.0.1"
        assert config.futu_opend_port == 11111

    def test_fetcher_init_reads_moomoo_alias(self):
        with patch.dict(
            os.environ,
            {"MOOMOO_OPEND_HOST": "10.0.0.5", "MOOMOO_OPEND_PORT": "22222"},
            clear=True,
        ), patch("src.config.get_config", side_effect=RuntimeError("no config")):
            fetcher = FutuFetcher()
            assert FutuFetcher.has_configured_endpoint() is True
        assert fetcher._host == "10.0.0.5"
        assert fetcher._port == 22222

    def test_futu_env_wins_over_moomoo_in_fetcher_init(self):
        with patch.dict(
            os.environ,
            {
                "FUTU_OPEND_HOST": "127.0.0.1",
                "FUTU_OPEND_PORT": "11111",
                "MOOMOO_OPEND_HOST": "10.0.0.5",
                "MOOMOO_OPEND_PORT": "22222",
            },
            clear=True,
        ), patch("src.config.get_config", side_effect=RuntimeError("no config")):
            fetcher = FutuFetcher()
        assert fetcher._host == "127.0.0.1"
        assert fetcher._port == 11111

    @patch("src.config.get_config")
    def test_no_endpoint_is_not_configured(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(futu_opend_host=None)
        with patch.dict(os.environ, {}, clear=True):
            assert FutuFetcher.has_configured_endpoint() is False


# ---------------------------------------------------------------------------
# Symbol builder and timestamps
# ---------------------------------------------------------------------------


class TestOpenDSymbols:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("hk00700", "HK.00700"),
            ("00700", "HK.00700"),
            ("HK00700", "HK.00700"),
            ("700", "HK.00700"),
            ("AAPL", "US.AAPL"),
            ("aapl", "US.AAPL"),
            ("BRK.B", "US.BRK.B"),
            ("600519", None),
            ("SH600519", None),
            ("SPX", None),
            ("^GSPC", None),
        ],
    )
    def test_symbol_builder_is_market_aware_and_fails_closed(self, code, expected):
        assert _opend_symbol(code) == expected

    def test_market_label(self):
        assert _opend_market("hk00700") == "hk"
        assert _opend_market("AAPL") == "us"


class TestMarketAwareTimestamp:
    def test_hk_timestamp_keeps_beijing_offset(self):
        assert _provider_timestamp("2026-08-24 10:30:00", "hk") == "2026-08-24T10:30:00+08:00"

    def test_us_timestamp_uses_edt_in_summer(self):
        assert _provider_timestamp("2026-07-15 10:30:00", "us") == "2026-07-15T10:30:00-04:00"

    def test_us_timestamp_uses_est_in_winter(self):
        assert _provider_timestamp("2026-01-15 10:30:00", "us") == "2026-01-15T10:30:00-05:00"


# ---------------------------------------------------------------------------
# Futu K-line
# ---------------------------------------------------------------------------


class TestFutuKline:
    def teardown_method(self):
        for patcher in reversed(getattr(self, "_patchers", [])):
            patcher.stop()
        opend_sdk.reset_opend_sdk_cache()

    def test_timeframe_maps_to_ktype_and_uses_qfq(self):
        for timeframe, ktype in _KLINE_KTYPE_NAMES.items():
            ctx = _FakeQuoteContext(
                lambda call, kwargs: (0, _rows("2026-08-24 10:00:00"), None)
            )
            fetcher = _make_fetcher(self, ctx=ctx)
            df = fetcher.get_kline_data(
                "AAPL", timeframe, start_date="2026-01-01", end_date="2026-08-24"
            )
            assert not df.empty
            symbol, kwargs = ctx.calls[0]
            assert symbol == "US.AAPL"
            assert kwargs["ktype"] == ktype
            assert kwargs["autype"] == "QFQ"
            assert kwargs["max_count"] == 1000

    def test_pagination_collects_pages_and_dedupes_frame(self):
        pages = {
            1: _rows(
                "2026-08-24 10:00:00",
                "2026-08-24 09:30:00",
                "2026-08-24 10:00:00",
                closes=[102.0, 101.0, 103.0],
            ),
            2: _rows("2026-08-24 10:30:00", closes=[104.0]),
        }

        def factory(call, kwargs):
            return 0, pages[call], ("next" if call == 1 else None)

        ctx = _FakeQuoteContext(factory)
        fetcher = _make_fetcher(self, ctx=ctx)
        df = fetcher.get_kline_data(
            "HK00700", "30m", start_date="2026-01-01", end_date="2026-08-24"
        )

        assert len(ctx.calls) == 2
        assert ctx.calls[0][1]["page_req_key"] is None
        assert ctx.calls[1][1]["page_req_key"] == "next"
        assert list(df.columns) == STANDARD_COLUMNS
        assert list(df["date"]) == [
            pd.Timestamp("2026-08-24 09:30:00"),
            pd.Timestamp("2026-08-24 10:00:00"),
            pd.Timestamp("2026-08-24 10:30:00"),
        ]
        assert list(df["close"]) == [101.0, 103.0, 104.0]
        assert str(df["date"].dtype).startswith("datetime64")
        assert pd.api.types.is_numeric_dtype(df["close"])
        assert df["date"].is_monotonic_increasing

    def test_page_loop_is_capped(self):
        calls = {"count": 0}

        def factory(call, kwargs):
            calls["count"] += 1
            day = 1 + call
            return 0, _rows(f"2026-08-{day:02d} 10:00:00", closes=[100.0 + call]), "next"

        ctx = _FakeQuoteContext(factory)
        fetcher = _make_fetcher(self, ctx=ctx)
        df = fetcher.get_kline_data(
            "HK00700", "60m", start_date="2026-01-01", end_date="2026-08-24"
        )
        assert calls["count"] == _MAX_HISTORY_PAGES
        assert len(df) == _MAX_HISTORY_PAGES

    def test_daily_path_is_single_page_qfq_for_us_and_unadjusted_for_hk(self):
        for code, autype in (("AAPL", "QFQ"), ("HK00700", "NONE")):
            ctx = _FakeQuoteContext(
                lambda call, kwargs: (0, _rows("2026-08-24"), "should-not-page")
            )
            fetcher = _make_fetcher(self, ctx=ctx)
            df = fetcher.get_daily_data(code, start_date="2026-01-01", end_date="2026-08-24")

            assert not df.empty
            assert len(ctx.calls) == 1
            _, kwargs = ctx.calls[0]
            assert kwargs["ktype"] == "K_DAY"
            assert kwargs["autype"] == autype, code
            assert "page_req_key" not in kwargs
            assert list(df.columns[: len(STANDARD_COLUMNS)]) == STANDARD_COLUMNS

    def test_a_share_returns_empty_without_connecting(self):
        fetcher = _make_fetcher(self, ctx=_FakeQuoteContext())
        assert fetcher.get_kline_data("600519", "1d").empty
        assert fetcher._ctx.calls == []

    def test_unsupported_timeframe_raises(self):
        fetcher = _make_fetcher(self, ctx=_FakeQuoteContext())
        with pytest.raises(DataFetchError):
            fetcher.get_kline_data("AAPL", "4h")


# ---------------------------------------------------------------------------
# BaseFetcher default
# ---------------------------------------------------------------------------


class _DailyOnlyFetcher:
    name = "DailyOnlyFetcher"
    priority = 1
    SUPPORTED_TIMEFRAMES = frozenset({"1d"})

    def __init__(self):
        self.calls = []

    def get_daily_data(self, stock_code, start_date=None, end_date=None, days=365):
        self.calls.append((stock_code, start_date, end_date, days))
        return pd.DataFrame([{"date": pd.Timestamp("2026-08-24"), "open": 1.0, **{
            column: 2.0 for column in STANDARD_COLUMNS if column not in {"date", "open"}
        }}])

    def get_kline_data(self, stock_code, timeframe="1d", start_date=None, end_date=None, days=365):
        from data_provider.base import BaseFetcher

        return BaseFetcher.get_kline_data(
            self, stock_code, timeframe=timeframe,
            start_date=start_date, end_date=end_date, days=days,
        )


class TestBaseFetcherDefault:
    def test_daily_delegates_to_get_daily_data_and_subsets_columns(self):
        fetcher = _DailyOnlyFetcher()
        df = fetcher.get_kline_data("600519", "1d", days=10)
        assert fetcher.calls == [("600519", None, None, 10)]
        assert list(df.columns) == STANDARD_COLUMNS

    def test_non_daily_timeframe_raises(self):
        with pytest.raises(DataFetchError):
            _DailyOnlyFetcher().get_kline_data("600519", "60m")


# ---------------------------------------------------------------------------
# DataFetcherManager K-line routing
# ---------------------------------------------------------------------------


class _KlineFetcher:
    def __init__(self, name, priority, result=None, error=None, timeframes=None):
        self.name = name
        self.priority = priority
        self.result = result
        self.error = error
        self.SUPPORTED_TIMEFRAMES = timeframes or frozenset({"1d", "1w", "60m", "30m"})
        self.calls = []

    def is_available_for_request(self, capability=""):
        return True

    def get_kline_data(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _manager(fetchers):
    return DataFetcherManager(fetchers=fetchers)


def _kline_frame():
    return pd.DataFrame([{"date": pd.Timestamp("2026-08-24"), "open": 1.0, "high": 2.0,
                          "low": 0.5, "close": 1.5, "volume": 100, "amount": 150.0,
                          "pct_chg": 1.0}])


class TestManagerKlineRouting:
    def setup_method(self):
        DataFetcherManager.reset_daily_source_health()

    def test_falls_back_from_failing_futu_to_yfinance(self):
        futu = _KlineFetcher("FutuFetcher", 2, error=RuntimeError("opend down"))
        yfinance = _KlineFetcher("YfinanceFetcher", 4, result=_kline_frame())
        manager = _manager([futu, yfinance])

        df, source = manager.get_kline_data("AAPL", "60m")

        assert source == "YfinanceFetcher"
        assert len(df) == 1
        assert len(futu.calls) == 1
        assert len(yfinance.calls) == 1

    def test_timeframe_filter_skips_daily_only_fetcher(self):
        futu = _KlineFetcher("FutuFetcher", 2, result=_kline_frame(), timeframes={"1d"})
        yfinance = _KlineFetcher("YfinanceFetcher", 4, result=_kline_frame())
        manager = _manager([futu, yfinance])

        _, source = manager.get_kline_data("AAPL", "60m")

        assert source == "YfinanceFetcher"
        assert futu.calls == []

    def test_all_empty_or_failed_raises(self):
        futu = _KlineFetcher("FutuFetcher", 2, error=RuntimeError("opend down"))
        yfinance = _KlineFetcher("YfinanceFetcher", 4, result=pd.DataFrame(columns=STANDARD_COLUMNS))
        manager = _manager([futu, yfinance])
        with pytest.raises(DataFetchError):
            manager.get_kline_data("AAPL", "60m")

    def test_unsupported_timeframe_rejected_before_routing(self):
        manager = _manager([_KlineFetcher("YfinanceFetcher", 4, result=_kline_frame())])
        with pytest.raises(DataFetchError):
            manager.get_kline_data("AAPL", "4h")


# ---------------------------------------------------------------------------
# US routing
# ---------------------------------------------------------------------------


class _DailyFetcher:
    def __init__(self, name, priority, result=None):
        self.name = name
        self.priority = priority
        self.result = result
        self.calls = []

    def is_available_for_request(self, capability=""):
        return True

    def get_daily_data(self, stock_code, start_date=None, end_date=None, days=365):
        self.calls.append(stock_code)
        return self.result if self.result is not None else _kline_frame()

    def get_realtime_quote(self, stock_code):
        self.calls.append(stock_code)
        quote = MagicMock()
        quote.has_basic_data.return_value = True
        for field in DataFetcherManager._SUPPLEMENT_FIELDS:
            setattr(quote, field, 1.0)
        return quote


class TestUSRouting:
    def setup_method(self):
        DataFetcherManager.reset_daily_source_health()

    def test_us_daily_prefers_futu_even_when_yfinance_has_lower_priority(self):
        futu = _DailyFetcher("FutuFetcher", 2)
        yfinance = _DailyFetcher("YfinanceFetcher", 0)
        manager = _manager([yfinance, futu])

        df, source = manager.get_daily_data("AAPL")

        assert source == "FutuFetcher"
        assert len(df) == 1
        assert futu.calls == ["AAPL"]
        assert yfinance.calls == []

    def test_us_daily_without_futu_uses_normal_chain(self):
        yfinance = _DailyFetcher("YfinanceFetcher", 4)
        manager = _manager([yfinance])

        _, source = manager.get_daily_data("AAPL")

        assert source == "YfinanceFetcher"
        assert yfinance.calls == ["AAPL"]

    def test_us_index_never_routes_to_futu(self):
        futu = _DailyFetcher("FutuFetcher", 2)
        yfinance = _DailyFetcher("YfinanceFetcher", 4)
        manager = _manager([futu, yfinance])

        _, source = manager.get_daily_data("SPX")

        assert source == "YfinanceFetcher"
        assert futu.calls == []
        assert yfinance.calls == ["SPX"]

    @patch("data_provider.futu_fetcher.FutuFetcher.has_configured_endpoint", return_value=True)
    @patch("src.config.get_config")
    def test_us_realtime_uses_futu_first_when_configured(self, mock_get_config, _mock_endpoint):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance",
            realtime_cache_ttl=None,
        )
        futu = _DailyFetcher("FutuFetcher", 2)
        yfinance = _DailyFetcher("YfinanceFetcher", 4)
        manager = _manager([futu, yfinance])
        manager._enrich_realtime_quote = lambda quote, **kwargs: quote

        quote = manager.get_realtime_quote("AAPL")

        assert quote is not None
        assert futu.calls == ["AAPL"]
        assert yfinance.calls == []

    @patch("data_provider.futu_fetcher.FutuFetcher.has_configured_endpoint", return_value=False)
    @patch("src.config.get_config")
    def test_us_realtime_skips_unconfigured_futu(self, mock_get_config, _mock_endpoint):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance",
            realtime_cache_ttl=None,
        )
        futu = _DailyFetcher("FutuFetcher", 2)
        yfinance = _DailyFetcher("YfinanceFetcher", 4)
        manager = _manager([futu, yfinance])
        manager._enrich_realtime_quote = lambda quote, **kwargs: quote

        quote = manager.get_realtime_quote("AAPL")

        assert quote is not None
        assert futu.calls == []
        assert yfinance.calls == ["AAPL"]


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


class TestDataCapabilityAdvertisement:
    def test_futu_advertises_us_quote_and_daily_but_financials_stay_hk(self):
        from src.services.data_capability_service import _PROVIDER_DEFINITION_MAP

        futu = _PROVIDER_DEFINITION_MAP["futu"]
        assert set(futu.dataset_markets["quote.realtime"]) == {"hk", "us"}
        assert set(futu.dataset_markets["kline.daily"]) == {"hk", "us"}
        assert set(futu.dataset_markets["financial.snapshot"]) == {"hk"}
