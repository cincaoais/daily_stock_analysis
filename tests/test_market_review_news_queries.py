# -*- coding: utf-8 -*-
"""Market review news queries follow the review language (no Chinese query for an English US review)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.market_profile import US_PROFILE
from src.market_analyzer import MarketAnalyzer


def _analyzer(language):
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.profile = US_PROFILE
    analyzer.region = "us"
    analyzer.search_service = MagicMock()
    analyzer.search_service.search_stock_news.return_value = SimpleNamespace(results=[])
    analyzer._get_review_language = lambda: language
    analyzer._log_context = lambda: "region=us"
    return analyzer


@pytest.mark.parametrize("language,expected", [
    ("en", ["US stock market", "S&P 500 NASDAQ"]),
    ("zh", ["美股 大盘", "US stock market", "S&P 500 NASDAQ"]),
])
def test_us_review_queries_follow_review_language(language, expected):
    analyzer = _analyzer(language)
    analyzer.search_market_news()
    sent = [" ".join(call.kwargs["focus_keywords"]) for call in analyzer.search_service.search_stock_news.call_args_list]
    assert sent == expected
