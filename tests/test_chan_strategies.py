# -*- coding: utf-8 -*-
"""Built-in strategies and default policies are Chan-theory only."""

import re
from pathlib import Path

import pytest

from src.agent.protocols import AgentContext
from src.agent.skills import defaults
from src.agent.skills.base import SkillManager
from src.agent.skills.router import SkillRouter

STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "strategies"
EXPECTED = {
    "chan_theory",
    "chan_divergence_reversal",
    "chan_second_buy",
    "chan_third_buy",
    "chan_zhongshu_oscillation",
    "chan_multi_level",
}
_CJK = re.compile(r"[一-鿿]")
_LEGACY_WORDING = ("MA5", "MA10", "MA20", "乖离率", "多头排列", "Bullish alignment", "Bias from")


def _builtin_skills():
    manager = SkillManager()
    manager.load_builtin_strategies()
    return {skill.name: skill for skill in manager.list_skills()}


def test_builtin_strategies_are_exactly_the_chan_set():
    assert {path.stem for path in STRATEGIES_DIR.glob("*.yaml")} == EXPECTED
    assert set(_builtin_skills()) == EXPECTED


def test_every_strategy_reads_the_code_computed_structure():
    for name, skill in _builtin_skills().items():
        assert "get_chan_structure" in skill.required_tools, name
        assert "缠论结构快照" in skill.instructions, name
        assert not any(word in skill.instructions for word in _LEGACY_WORDING), name
        assert all(1 <= rule <= 7 for rule in skill.core_rules), name


def test_default_and_regime_routing_point_at_chan_strategies():
    assert defaults.PRIMARY_DEFAULT_SKILL_ID == "chan_theory"
    assert defaults.DEFAULT_ACTIVE_SKILL_IDS == ("chan_theory",)
    assert defaults.DEFAULT_ROUTER_SKILL_IDS == ("chan_theory",)
    regimes = defaults.REGIME_SKILL_IDS
    assert set(regimes) == {"trending_up", "trending_down", "sideways", "volatile", "sector_hot"}
    assert "chan_third_buy" in regimes["trending_up"]
    assert "chan_divergence_reversal" in regimes["trending_down"]
    assert "chan_zhongshu_oscillation" in regimes["sideways"]
    assert all(set(ids) <= EXPECTED for ids in regimes.values())


def test_default_policies_are_chan_rules():
    for text in (
        defaults.CORE_TRADING_SKILL_POLICY_ZH,
        defaults.CORE_TRADING_SKILL_POLICY_EN,
        defaults.TECHNICAL_SKILL_RULES_EN,
    ):
        assert not any(word in text for word in _LEGACY_WORDING)
        assert "invalidation_price" in text
    assert "缠论结构快照" in defaults.CORE_TRADING_SKILL_POLICY_ZH
    assert not _CJK.search(defaults.CORE_TRADING_SKILL_POLICY_EN)
    assert not _CJK.search(defaults.TECHNICAL_SKILL_RULES_EN)
    assert defaults.get_default_trading_skill_policy(explicit_skill_selection=True) == ""


def _ctx_with_daily(**daily):
    ctx = AgentContext(query="分析 AAPL", stock_code="AAPL")
    level = {"status": "available", "divergence": {"present": False}}
    level.update(daily)
    ctx.set_data("chan_snapshot", {"levels": {"D": level}})
    return ctx


@pytest.mark.parametrize(
    "daily, expected",
    [
        ({"trend_type": "up"}, "trending_up"),
        ({"trend_type": "down"}, "trending_down"),
        ({"trend_type": "consolidation"}, "sideways"),
        ({"trend_type": "up", "divergence": {"present": True}}, "volatile"),
        ({"trend_type": "undetermined"}, None),
        ({"trend_type": "up", "status": "missing"}, None),
    ],
)
def test_router_reads_regime_from_daily_chan_structure(daily, expected):
    assert SkillRouter._detect_chan_regime(_ctx_with_daily(**daily)) == expected


def test_router_without_snapshot_has_no_chan_regime():
    assert SkillRouter._detect_chan_regime(AgentContext(query="x", stock_code="AAPL")) is None
