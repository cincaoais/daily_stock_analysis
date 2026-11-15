# -*- coding: utf-8 -*-
"""
Shared defaults for trading skills.

This module centralises:
1. The default active skill set used by agent entrypoints
2. The fallback skill subset used by the multi-agent router
3. Common prompt fragments that previously drifted across multiple files
4. Helper utilities for skill-specific agent naming
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "strategies"

SKILL_AGENT_PREFIX = "skill_"
LEGACY_STRATEGY_AGENT_PREFIX = "strategy_"
SKILL_CONSENSUS_AGENT_NAME = "skill_consensus"
LEGACY_STRATEGY_CONSENSUS_AGENT_NAME = "strategy_consensus"

CORE_TRADING_SKILL_POLICY_ZH = """## 默认技能基线（必须严格遵守）

当前激活的 skills 可以补充细化分析视角，但默认风险控制和交易节奏必须遵守以下缠论基线。

### 1. 结构先行（只用快照）
- 笔、中枢、背驰、买卖点一律取自程序计算的“缠论结构快照”（或 `get_chan_structure` 工具结果）
- 不得自行从 K 线重算结构，也不得凭记忆补充；快照未提供的结构写“快照未提供”，线段未计算时不得描述线段

### 2. 级别明确
- 任何结论都要说明级别：周线定方向，日线定买卖点，60/30 分钟定进出场
- 不同级别的结论不得混用；级别数据缺失或不完整时必须说明并降低置信度

### 3. 只在买卖点操作
- 只有仍然有效的一、二、三类买卖点才构成操作依据
- 没有有效买卖点时结论为“观望/持有观察”，不得因为单日涨跌追涨杀跌

### 4. 背驰判断
- 背驰以快照的 MACD 面积比较为准；未确认的背驰只能写“形成中，待确认”
- 趋势背驰的转折级别高于盘整背驰，盘整背驰只看回到中枢

### 5. 失效价即止损
- 买入建议必须给出止损，并锚定对应买卖点的失效价（`invalidation_price`）
- 买点失效（价格击穿失效价）后原判断作废，不得继续持有等待

### 6. 多级别联立
- 大级别与操作级别同向才可用正常仓位；逆大级别只做小级别反弹并减仓
- 各级别方向矛盾且没有有效买卖点时，不做方向性结论

### 7. 风险排查与数据局限
- 减持公告、业绩预亏、监管处罚、行业政策利空、大额解禁等重大利空优先于技术结构
- 估值明显偏高或行情/新闻数据过期、缺失时，在风险点中说明并下调置信度
"""

CORE_TRADING_SKILL_POLICY_EN = """## Default Skill Baseline (must be followed strictly)

The currently activated skills may refine the analysis perspective, but the default risk controls and trading rhythm must follow this Chan theory baseline.

### 1. Structure first (snapshot only)
- Take strokes, pivots, divergence and buy/sell points only from the code-computed Chan structure snapshot (or the `get_chan_structure` tool result)
- Never recompute structure from K-lines or recall it from memory; write "not in snapshot" for anything missing, and do not describe segments when they are not computed

### 2. State the level
- Every conclusion names its level: weekly sets direction, daily sets buy/sell points, 60/30-minute sets entries and exits
- Do not mix conclusions across levels; when a level is missing or partial, say so and lower confidence

### 3. Act only at buy/sell points
- Only class 1, 2 or 3 buy/sell points that are still valid justify an action
- With no valid point the conclusion is "Watch / hold and observe"; do not chase a single day's move

### 4. Divergence
- Divergence follows the snapshot's MACD-area comparison; an unconfirmed divergence is "forming, awaiting confirmation"
- A trend divergence outranks a range divergence, which only targets a return to the pivot

### 5. The invalidation price is the stop
- Every buy idea carries a stop-loss anchored to the point's `invalidation_price`
- Once a point is invalidated the thesis is void; do not keep holding and waiting

### 6. Multi-level alignment
- Use a normal position only when the higher level agrees with the operating level; against the higher level, trade the smaller-level bounce with reduced size
- When levels conflict and no point is valid, give no directional call

### 7. Risk screening and data limits
- Material negatives (shareholder reduction, expected losses, regulatory penalties, negative industry policy, large share unlocks) outrank technical structure
- When valuation is clearly elevated, or quote/news data is stale or missing, state it in the risk points and lower confidence
"""

TECHNICAL_SKILL_RULES_EN = """## Default Skill Baseline

Treat the currently activated skills as the primary analysis lens, but keep the
following Chan theory controls as the shared baseline:

- Structural facts (strokes, pivots, divergence, buy/sell points) come only from the
  code-computed Chan structure snapshot or `get_chan_structure`; never recompute them
- Name the level for every claim: weekly = direction, daily = buy/sell point, 60/30m = timing
- Act only on buy/sell points that are still valid; no valid point -> hold off
- Anchor the stop-loss to the point's `invalidation_price`; an invalidated point voids the thesis
- Unconfirmed strokes or divergence are "forming"; partial or missing levels lower confidence
"""


def get_default_trading_skill_policy(*, explicit_skill_selection: bool) -> str:
    """Return the legacy default trading baseline only for implicit/default runs.

    When a caller explicitly chooses a skill (via request payload or config),
    analysis should follow that selected skill alone instead of silently
    layering the old bull-trend baseline on top.
    """
    if explicit_skill_selection:
        return ""
    return CORE_TRADING_SKILL_POLICY_ZH


def get_default_technical_skill_policy(*, explicit_skill_selection: bool) -> str:
    """Return the technical-agent baseline only for implicit/default runs."""
    if explicit_skill_selection:
        return ""
    return TECHNICAL_SKILL_RULES_EN


@lru_cache(maxsize=1)
def _load_builtin_skill_catalog() -> tuple[object, ...]:
    try:
        from src.agent.skills.base import load_skills_from_directory

        return tuple(load_skills_from_directory(_BUILTIN_SKILLS_DIR))
    except Exception:
        return ()


def _coerce_priority(value: object, default: int = 100) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_available_ids(available_skill_ids: Optional[Iterable[str]]) -> List[str]:
    normalized: List[str] = []
    if available_skill_ids is None:
        return normalized
    for skill_id in available_skill_ids:
        if isinstance(skill_id, str):
            cleaned = skill_id.strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
    return normalized


def _normalize_skill_inputs(
    skills: Optional[Iterable[object]],
    available_skill_ids: Optional[Iterable[str]] = None,
) -> tuple[List[object], List[str]]:
    normalized_available = _normalize_available_ids(available_skill_ids)

    if skills is None:
        return list(_load_builtin_skill_catalog()), normalized_available

    skill_pool: List[object] = []
    for item in skills:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in normalized_available:
                normalized_available.append(cleaned)
            continue
        if item is not None:
            skill_pool.append(item)
    return skill_pool, normalized_available


def _sort_skill_pool(skills: Iterable[object]) -> List[object]:
    return sorted(
        skills,
        key=lambda skill: (
            _coerce_priority(getattr(skill, "default_priority", 100)),
            str(getattr(skill, "display_name", "") or getattr(skill, "name", "")),
            str(getattr(skill, "name", "")),
        ),
    )


def _iter_candidate_skills(
    skills: Optional[Iterable[object]],
    *,
    available_skill_ids: Optional[Iterable[str]] = None,
    user_invocable_only: bool = True,
) -> tuple[List[object], List[str]]:
    skill_pool, normalized_available = _normalize_skill_inputs(skills, available_skill_ids)
    available_lookup = set(normalized_available)

    candidates: List[object] = []
    for skill in _sort_skill_pool(skill_pool):
        skill_id = str(getattr(skill, "name", "")).strip()
        if not skill_id:
            continue
        if user_invocable_only and not bool(getattr(skill, "user_invocable", True)):
            continue
        if available_lookup and skill_id not in available_lookup:
            continue
        candidates.append(skill)

    return candidates, normalized_available


def _slice_skill_ids(skill_ids: List[str], max_count: Optional[int]) -> List[str]:
    if max_count is None:
        return skill_ids
    return skill_ids[:max_count]


def _pick_primary_default_skill_id(candidates: List[object]) -> str:
    preferred = [
        str(getattr(skill, "name", "")).strip()
        for skill in candidates
        if bool(getattr(skill, "default_active", False))
    ]
    if preferred:
        return preferred[0]

    fallback = [str(getattr(skill, "name", "")).strip() for skill in candidates]
    if fallback:
        return fallback[0]

    return ""


def get_default_active_skill_ids(
    skills: Optional[Iterable[object]] = None,
    max_count: Optional[int] = None,
    available_skill_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    candidates, normalized_available = _iter_candidate_skills(
        skills,
        available_skill_ids=available_skill_ids,
    )
    default_skill_id = _pick_primary_default_skill_id(candidates)
    if default_skill_id:
        return _slice_skill_ids([default_skill_id], max_count)

    return _slice_skill_ids(normalized_available[:1], max_count)


def get_default_router_skill_ids(
    skills: Optional[Iterable[object]] = None,
    max_count: Optional[int] = None,
    available_skill_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    candidates, normalized_available = _iter_candidate_skills(
        skills,
        available_skill_ids=available_skill_ids,
    )
    preferred = [
        str(getattr(skill, "name", "")).strip()
        for skill in candidates
        if bool(getattr(skill, "default_router", False))
    ]
    if preferred:
        return _slice_skill_ids(preferred, max_count)

    return get_default_active_skill_ids(
        candidates,
        max_count=max_count,
        available_skill_ids=normalized_available,
    )


def get_regime_skill_ids(
    regime: str,
    skills: Optional[Iterable[object]] = None,
    max_count: Optional[int] = None,
    available_skill_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    candidates, normalized_available = _iter_candidate_skills(
        skills,
        available_skill_ids=available_skill_ids,
    )
    regime_name = (regime or "").strip().lower()
    if regime_name:
        matched = []
        for skill in candidates:
            market_regimes = getattr(skill, "market_regimes", None) or []
            normalized_regimes = {
                str(item).strip().lower()
                for item in market_regimes
                if str(item).strip()
            }
            if regime_name in normalized_regimes:
                matched.append(str(getattr(skill, "name", "")).strip())
        if matched:
            return _slice_skill_ids(matched, max_count)

    return get_default_router_skill_ids(
        candidates,
        max_count=max_count,
        available_skill_ids=normalized_available,
    )


def get_primary_default_skill_id(
    skills: Optional[Iterable[object]] = None,
    available_skill_ids: Optional[Iterable[str]] = None,
) -> str:
    defaults = get_default_active_skill_ids(skills, max_count=1, available_skill_ids=available_skill_ids)
    return defaults[0] if defaults else ""


def _build_regime_skill_ids(skills: Iterable[object]) -> Dict[str, List[str]]:
    regime_map: Dict[str, List[str]] = {}
    for skill in _sort_skill_pool(skills):
        skill_id = str(getattr(skill, "name", "")).strip()
        if not skill_id:
            continue
        for regime in getattr(skill, "market_regimes", None) or []:
            regime_name = str(regime).strip().lower()
            if not regime_name:
                continue
            regime_map.setdefault(regime_name, []).append(skill_id)
    return regime_map


DEFAULT_ACTIVE_SKILL_IDS: tuple[str, ...] = tuple(get_default_active_skill_ids())
DEFAULT_ROUTER_SKILL_IDS: tuple[str, ...] = tuple(get_default_router_skill_ids())
PRIMARY_DEFAULT_SKILL_ID = get_primary_default_skill_id()
REGIME_SKILL_IDS: Dict[str, List[str]] = _build_regime_skill_ids(_load_builtin_skill_catalog())


def build_skill_agent_name(skill_id: str) -> str:
    return f"{SKILL_AGENT_PREFIX}{skill_id}"


def extract_skill_id(agent_name: Optional[str]) -> Optional[str]:
    if not agent_name or not isinstance(agent_name, str):
        return None
    for prefix in (SKILL_AGENT_PREFIX, LEGACY_STRATEGY_AGENT_PREFIX):
        if agent_name.startswith(prefix):
            return agent_name[len(prefix):]
    return None


def is_skill_agent_name(agent_name: Optional[str]) -> bool:
    return extract_skill_id(agent_name) is not None


def is_skill_consensus_name(agent_name: Optional[str]) -> bool:
    return agent_name in {SKILL_CONSENSUS_AGENT_NAME, LEGACY_STRATEGY_CONSENSUS_AGENT_NAME}
