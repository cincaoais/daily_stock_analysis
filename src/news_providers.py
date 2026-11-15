# -*- coding: utf-8 -*-
"""Ticker-keyed news providers (Yahoo Finance, Finnhub).

Unlike the keyword web-search providers in ``src.search_service``, these look news
up by stock symbol. They plug into ``SearchService`` through the same
``BaseSearchProvider`` contract and receive the stock code as the ``stock_code``
search keyword; the shared freshness filter, ranking and admission rules still apply.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

from src.search_service import BaseSearchProvider, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

YAHOO_PROVIDER_NAME = "YahooFinanceNews"
FINNHUB_PROVIDER_NAME = "FinnhubNews"
SUPPORTED_TICKER_NEWS_PROVIDERS = ("yahoo", "finnhub")

_FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_HTTP_TIMEOUT_SECONDS = 10
_NO_KEY = "no-key-required"


class TickerNewsProvider(BaseSearchProvider):
    """Base class for providers that need a stock code rather than a free-text query."""

    def _do_search(  # type: ignore[override]
        self,
        query: str,
        api_key: str,
        max_results: int,
        days: int = 7,
        stock_code: Optional[str] = None,
        **_ignored: Any,
    ) -> SearchResponse:
        if not stock_code:
            return self._failure(query, "ticker news provider needs a stock_code")
        try:
            results = self._fetch(stock_code, api_key, max_results=max_results, days=days)
        except requests.RequestException as exc:
            return self._failure(query, "%s request failed: %s" % (self.name, exc))
        if results is None:
            return self._failure(query, "%s does not cover %s" % (self.name, stock_code))
        return SearchResponse(query=query, results=results[:max_results], provider=self.name, success=True)

    def search_by_code(self, stock_code: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """Look news up by stock code (the query text is only used for logging)."""
        return self._execute_search(
            f"{stock_code} news", max_results=max_results, days=days, stock_code=stock_code
        )

    def _fetch(self, stock_code: str, api_key: str, *, max_results: int, days: int) -> Optional[List[SearchResult]]:
        """Return results, or None when this provider does not cover the symbol."""
        raise NotImplementedError

    def _failure(self, query: str, message: str) -> SearchResponse:
        return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=message)


def _iso_from_epoch(value: Any) -> Optional[str]:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class YahooFinanceNewsProvider(TickerNewsProvider):
    """Yahoo Finance news via ``yfinance`` (no key). Covers US, HK and A-share symbols."""

    def __init__(self) -> None:
        super().__init__([_NO_KEY], YAHOO_PROVIDER_NAME)
        self._symbol_converter = None
        self._converter_lock = threading.Lock()

    def _yahoo_symbol(self, stock_code: str) -> str:
        with self._converter_lock:
            if self._symbol_converter is None:
                from data_provider.yfinance_fetcher import YfinanceFetcher

                self._symbol_converter = YfinanceFetcher()
        return self._symbol_converter._convert_stock_code(stock_code)

    def _fetch(self, stock_code: str, api_key: str, *, max_results: int, days: int) -> Optional[List[SearchResult]]:
        import yfinance as yf

        symbol = self._yahoo_symbol(stock_code)
        items = yf.Ticker(symbol).news or []
        return [r for r in (_yahoo_item_to_result(item) for item in items) if r is not None]


def _yahoo_item_to_result(item: Dict[str, Any]) -> Optional[SearchResult]:
    """Parse both the current ``{"content": {...}}`` and the older flat yfinance shapes."""
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content is not None:
        title = content.get("title") or ""
        url = ((content.get("canonicalUrl") or {}).get("url")
               or (content.get("clickThroughUrl") or {}).get("url") or "")
        source = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
        snippet = content.get("summary") or content.get("description") or ""
        published = content.get("pubDate") or content.get("displayTime")
    else:
        title = item.get("title") or ""
        url = item.get("link") or ""
        source = item.get("publisher") or "Yahoo Finance"
        snippet = item.get("summary") or ""
        published = _iso_from_epoch(item.get("providerPublishTime"))
    if not title or not url:
        return None
    return SearchResult(title=title, snippet=snippet or title, url=url, source=source, published_date=published)


class FinnhubNewsProvider(TickerNewsProvider):
    """Finnhub company news. The free plan covers North American companies only."""

    def __init__(self, api_key: str) -> None:
        super().__init__([api_key], FINNHUB_PROVIDER_NAME)

    def _fetch(self, stock_code: str, api_key: str, *, max_results: int, days: int) -> Optional[List[SearchResult]]:
        from data_provider.us_index_mapping import is_us_stock_code

        symbol = (stock_code or "").strip().upper()
        if not is_us_stock_code(symbol):
            return None
        today = datetime.now(timezone.utc).date()
        params = {
            "symbol": symbol,
            "from": (today - timedelta(days=max(int(days), 1))).isoformat(),
            "to": today.isoformat(),
            "token": api_key,
        }
        response = requests.get(_FINNHUB_COMPANY_NEWS_URL, params=params, timeout=_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise requests.RequestException("unexpected Finnhub payload: %s" % str(payload)[:200])
        payload.sort(key=lambda row: row.get("datetime") or 0, reverse=True)
        results = []
        for row in payload:
            title, url = row.get("headline") or "", row.get("url") or ""
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                snippet=row.get("summary") or title,
                url=url,
                source=row.get("source") or "Finnhub",
                published_date=_iso_from_epoch(row.get("datetime")),
            ))
        return results


def parse_ticker_news_providers(value: Any) -> List[str]:
    """Normalise ``NEWS_TICKER_PROVIDERS`` into known provider names, keeping order."""
    if value is None:
        return list(SUPPORTED_TICKER_NEWS_PROVIDERS)
    items: Iterable[str] = value.split(",") if isinstance(value, str) else value
    names: List[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name in SUPPORTED_TICKER_NEWS_PROVIDERS and name not in names:
            names.append(name)
    return names


def build_ticker_news_providers(names: Iterable[str], *, finnhub_api_key: Optional[str] = None) -> List[TickerNewsProvider]:
    """Instantiate providers in the given order; Finnhub is skipped without a key."""
    providers: List[TickerNewsProvider] = []
    for name in names:
        if name == "yahoo":
            providers.append(YahooFinanceNewsProvider())
        elif name == "finnhub" and finnhub_api_key:
            providers.append(FinnhubNewsProvider(finnhub_api_key))
    return providers


def ticker_news_kwargs_from_config(config: Any) -> Dict[str, Any]:
    """SearchService constructor kwargs for ticker news, read from the app config."""
    return {
        "ticker_news_providers": list(getattr(config, "news_ticker_providers", None) or []),
        "finnhub_api_key": getattr(config, "finnhub_api_key", None),
    }
