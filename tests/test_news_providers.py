# -*- coding: utf-8 -*-
"""Ticker-keyed news providers (src/news_providers.py) and their SearchService wiring."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src import news_providers
from src.news_providers import (
    FinnhubNewsProvider,
    YahooFinanceNewsProvider,
    build_ticker_news_providers,
    parse_ticker_news_providers,
)
from src.search_service import SearchResponse, SearchResult, SearchService


def _recent(hours=2):
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _yahoo_item(title, hours=2, url="https://example.com/a"):
    return {"content": {
        "title": title,
        "summary": title + " summary",
        "pubDate": _recent(hours).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": url},
    }}


def test_parse_provider_list():
    assert parse_ticker_news_providers(None) == ["yahoo", "finnhub"]
    assert parse_ticker_news_providers(" Finnhub ,yahoo,bogus,yahoo") == ["finnhub", "yahoo"]
    assert parse_ticker_news_providers("") == []


def test_build_skips_finnhub_without_key():
    assert [p.name for p in build_ticker_news_providers(["yahoo", "finnhub"])] == ["YahooFinanceNews"]
    names = [p.name for p in build_ticker_news_providers(["finnhub", "yahoo"], finnhub_api_key="k")]
    assert names == ["FinnhubNews", "YahooFinanceNews"]


def test_config_parser_matches_module_parser():
    from src.config import _parse_news_ticker_providers

    for value in (None, "", "yahoo", "finnhub, yahoo", "x,yahoo"):
        assert _parse_news_ticker_providers(value) == parse_ticker_news_providers(value)


def test_yahoo_parses_current_and_legacy_payloads():
    legacy = {"title": "Old shape", "link": "https://example.com/b", "publisher": "AP",
              "providerPublishTime": int(_recent().timestamp())}
    broken = {"content": {"title": "", "canonicalUrl": {"url": "https://example.com/c"}}}
    provider = YahooFinanceNewsProvider()
    fake_ticker = SimpleNamespace(news=[_yahoo_item("New shape"), legacy, broken])
    with patch.object(YahooFinanceNewsProvider, "_yahoo_symbol", return_value="AAPL"), \
            patch("yfinance.Ticker", return_value=fake_ticker) as ticker:
        response = provider.search_by_code("AAPL")

    ticker.assert_called_once_with("AAPL")
    assert response.success
    assert [r.title for r in response.results] == ["New shape", "Old shape"]
    assert response.results[0].source == "Reuters"
    assert response.results[1].published_date.endswith("+00:00")


def test_ticker_provider_without_stock_code_fails_cleanly():
    response = YahooFinanceNewsProvider().search("Apple news", max_results=3)
    assert not response.success
    assert "stock_code" in response.error_message


def test_finnhub_only_covers_us_and_sorts_newest_first():
    provider = FinnhubNewsProvider("secret")
    assert provider.search_by_code("HK00700").success is False
    assert provider.search_by_code("600519").success is False

    older, newer = int(_recent(30).timestamp()), int(_recent(1).timestamp())
    payload = [
        {"headline": "Older", "url": "https://x/1", "source": "MarketWatch", "summary": "s", "datetime": older},
        {"headline": "Newer", "url": "https://x/2", "source": "Reuters", "summary": "s", "datetime": newer},
        {"headline": "", "url": "https://x/3", "datetime": newer},
    ]
    http = MagicMock()
    http.json.return_value = payload
    with patch.object(news_providers.requests, "get", return_value=http) as get:
        response = provider.search_by_code("AAPL", days=3)

    params = get.call_args.kwargs["params"]
    assert params["symbol"] == "AAPL" and params["token"] == "secret"
    assert [r.title for r in response.results] == ["Newer", "Older"]


def test_finnhub_http_error_is_a_failed_response():
    http = MagicMock()
    http.raise_for_status.side_effect = news_providers.requests.HTTPError("403 Forbidden")
    with patch.object(news_providers.requests, "get", return_value=http):
        response = FinnhubNewsProvider("bad").search_by_code("AAPL")
    assert not response.success and "403" in response.error_message


# ---------------------------------------------------------------------------
# SearchService wiring
# ---------------------------------------------------------------------------

def _fake_ticker_provider(results, name="FakeTicker"):
    provider = MagicMock()
    provider.name = name
    provider.is_available = True
    provider._execute_search.return_value = SearchResponse(
        query="q", results=results, provider=name, success=bool(results)
    )
    return provider


def _apple_result(title="Apple unveils new iPhone lineup"):
    return SearchResult(
        title=title, snippet="Apple Inc. (AAPL) said on Tuesday...", url="https://example.com/apple",
        source="Reuters", published_date=_recent().isoformat(),
    )


def test_service_is_available_with_only_ticker_providers():
    service = SearchService(ticker_news_providers=["yahoo"])
    assert service.is_available
    assert not service.has_keyword_search
    assert not SearchService().is_available


def test_stock_news_uses_ticker_provider_first_and_passes_code():
    service = SearchService()
    fake = _fake_ticker_provider([_apple_result()])
    service._ticker_providers = [fake]

    response = service.search_stock_news("AAPL", "Apple", max_results=3)

    assert response.success and response.provider == "FakeTicker"
    assert fake._execute_search.call_args.kwargs["stock_code"] == "AAPL"


def test_stock_news_with_focus_keywords_skips_ticker_providers():
    service = SearchService()
    fake = _fake_ticker_provider([_apple_result()])
    service._ticker_providers = [fake]

    service.search_stock_news("market", "US market", max_results=3, focus_keywords=["Fed", "rates"])

    fake._execute_search.assert_not_called()


def test_empty_ticker_result_falls_through_to_keyword_engines():
    service = SearchService()
    service._ticker_providers = [_fake_ticker_provider([])]
    keyword = MagicMock()
    keyword.name = "FakeKeyword"
    keyword.is_available = True
    keyword.search.return_value = SearchResponse(
        query="q", results=[_apple_result("Apple shares rise after earnings")], provider="FakeKeyword", success=True
    )
    service._providers = [keyword]

    response = service.search_stock_news("AAPL", "Apple", max_results=3)

    keyword.search.assert_called()
    assert response.provider == "FakeKeyword"


def test_comprehensive_intel_takes_latest_news_from_ticker_provider():
    service = SearchService()
    service._ticker_providers = [_fake_ticker_provider([_apple_result()])]

    with patch("src.search_service.time.sleep"):
        intel = service.search_comprehensive_intel("AAPL", "Apple", max_searches=3)

    assert list(intel) == ["latest_news"]
    assert intel["latest_news"].provider == "FakeTicker"


@pytest.mark.parametrize("value, expected", [(None, ["yahoo", "finnhub"]), ("", []), ("finnhub", ["finnhub"])])
def test_config_reads_news_ticker_providers(monkeypatch, value, expected):
    from src.config import Config

    if value is None:
        monkeypatch.delenv("NEWS_TICKER_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("NEWS_TICKER_PROVIDERS", value)
    Config.reset_instance()
    try:
        assert Config.get_instance().news_ticker_providers == expected
    finally:
        Config.reset_instance()
