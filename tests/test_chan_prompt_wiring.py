# -*- coding: utf-8 -*-
"""The Chan snapshot reaches the default prompt, the single agent and multi-agent workers."""

from unittest.mock import MagicMock, patch

from src.agent.agents.technical_agent import TechnicalAgent
from src.agent.executor import AgentExecutor
from src.agent.protocols import AgentContext
from src.agent.tools.registry import ToolRegistry
from src.analyzer import GeminiAnalyzer

SNAPSHOT = {
    "schema_version": "chan_snapshot/1.0",
    "symbol": "AAPL",
    "data_quality": {"status": "available", "engine": "czsc", "warnings": []},
    "levels": {
        "D": {
            "status": "available",
            "trend_type": "up",
            "buy_sell_points": [
                {"evidence_id": "D.3B.2026-10-20", "type": "3B", "dt": "2026-10-20",
                 "price": 338.7, "invalidation_price": 334.99, "still_valid": True},
            ],
        },
    },
    "cross_level": {"resonance": "unknown"},
}


def _context(**extra):
    context = {"code": "AAPL", "stock_name": "Apple", "date": "2026-11-13", "today": {}}
    context.update(extra)
    return context


def _analyzer():
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        return GeminiAnalyzer()


def test_default_prompt_includes_snapshot_section():
    prompt = _analyzer()._format_prompt(_context(chan_snapshot=SNAPSHOT), "Apple", None)
    assert "缠论结构快照" in prompt
    assert '"evidence_id":"D.3B.2026-10-20"' in prompt
    assert '"invalidation_price":334.99' in prompt


def test_default_prompt_without_snapshot_has_no_section():
    prompt = _analyzer()._format_prompt(_context(), "Apple", None)
    assert "缠论结构快照" not in prompt
    assert "chan_snapshot" not in prompt


def test_single_agent_user_message_includes_snapshot_section():
    executor = AgentExecutor(ToolRegistry(), MagicMock(), max_steps=1)
    message = executor._build_user_message(
        "Analyze AAPL", {"stock_code": "AAPL", "chan_snapshot": SNAPSHOT}
    )
    assert "缠论结构快照" in message
    assert "D.3B.2026-10-20" in message


def test_multi_agent_worker_gets_rendered_snapshot_not_truncated_json():
    agent = TechnicalAgent(tool_registry=MagicMock(), llm_adapter=MagicMock())
    agent.memory = None
    ctx = AgentContext(query="分析 AAPL", stock_code="AAPL")
    ctx.set_data("chan_snapshot", SNAPSHOT)
    ctx.set_data("realtime_quote", {"price": 356.27})

    with patch.object(TechnicalAgent, "_build_memory_context", return_value=""):
        injected = agent._inject_cached_data(ctx)

    assert "缠论结构快照" in injected
    assert "[Pre-fetched: chan_snapshot]" not in injected
    assert "[Pre-fetched: realtime_quote]" in injected
    assert "D.3B.2026-10-20" in injected


def test_technical_agent_prompt_puts_chan_structure_first():
    prompt = TechnicalAgent(tool_registry=MagicMock(), llm_adapter=MagicMock()).system_prompt(
        AgentContext(query="分析 AAPL", stock_code="AAPL")
    )
    assert "get_chan_structure" in prompt
    assert "invalidation_price" in prompt
    assert "get_chan_structure" in TechnicalAgent.tool_names
