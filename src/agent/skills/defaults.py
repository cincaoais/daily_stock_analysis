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

CORE_TRADING_SKILL_POLICY_ZH = """## 默认技能基线：缠论（必须严格遵守）

激活的 skills 可以细化视角，但最终动作由以下规则决定。

### 1. 证据：只用缠论结构快照
- 笔、中枢、背驰、买卖点一律取自程序计算的“缠论结构快照”（或 `get_chan_structure` 工具结果）；不得从 K 线重算、不得凭记忆补充、不得引入快照中没有的缠论概念。
- 每条结构判断都要带 `evidence_id`（或笔 id）；快照没有的内容写“快照未提供”；`segment` 为 `not_supported` 时不得描述线段。
- 不得编造数字：正文中不写概率、胜率、百分比置信度；不用固定百分比止损或目标。

### 2. 级别分工
- 按快照 `level_roles` 分工：`direction` 级别定方向，`operating` 级别给买卖点，`confirmation` 级别定进出场时机。每条结论都写明级别。
- 力度（背驰）只能在同一级别内比较；大级别只用于确认方向（区间套），不做跨级别力度比较。
- `status` 不是 `available` 的级别属于数据局限：必须说明并降低置信度。

### 3. 稳定性状态（`state`）
- `forming`：笔尚未完成，可能完全改变，只能观察。
- `candidate`：位于最新一笔已完成的笔上，仍可能被重画；只有方向级别不相反时才可计划小仓试探，并写明“候选，待确认”。
- `confirmed`：其后已有新的一笔，结构已固定；`still_valid=true` 时可作为操作依据。
- `invalidated`：价格已击穿 `invalidation_price`，只能作为历史参考，原判断作废。
- 不得在文字中升级状态（不得把 candidate 说成已确认）。

### 4. 背驰是预警，不是指令
- 说明背驰时必须写出两段比较笔（`stroke_id` 与 `compared_stroke_id`）及 MACD 面积比；快照写无背驰就是没有背驰。
- 单独的背驰不能得出买入/加仓或卖出，只代表“准备”；要操作需等二买/三买（或二卖/三卖），或确认级别的确认。
- `range_divergence`（盘整背驰）不给方向，只预期回到中枢；`trend_divergence`（趋势背驰）才可能标志趋势结束。

### 5. 决策闸门（任一不通过 → 观望/持有观察）
1. 操作级别数据可用？
2. 方向级别不与交易相反（若相反，必须减仓并写明）？
3. 操作级别或确认级别存在 `state` 为 confirmed 或 candidate、且 `still_valid=true` 的买点？
4. 现价仍在该买点 `invalidation_price` 之上，且到失效价的距离与仓位相称？
5. 没有足以压倒结构的重大利空或事件？
全部通过才可给出买入/加仓。卖出同理使用卖点；持仓所依赖的买点变为 invalidated 时，持仓者应减仓/卖出。

### 6. 先写失效，再写入场
- 止损 = 对应买卖点的 `invalidation_price`（一买/二买：一买低点；三买：中枢 ZG；卖点对称）。目标位取自 `key_levels` 或中枢边界，不用固定百分比。

### 7. 辅助数据只作背景
- 均线、量能、筹码、资金流、估值和新闻只作背景：可以否决或降低置信度，不能单独产生买卖结论。
- 减持公告、业绩预亏、监管处罚、行业政策利空、大额解禁等重大利空优先于技术结构；行情或新闻过期、缺失时降低置信度。
"""

CORE_TRADING_SKILL_POLICY_EN = """## Default Skill Baseline: Chan theory (must be followed strictly)

Activated skills may refine the view; these rules decide the final action.

### 1. Evidence: the Chan structure snapshot only
- Strokes, pivots, divergence and buy/sell points come only from the code-computed Chan structure snapshot (or the `get_chan_structure` tool result). Never recompute them from K-lines, never recall them from memory, never introduce Chan concepts the snapshot does not contain.
- Cite the `evidence_id` (or stroke id) for every structural claim. Write "not in snapshot" for anything missing. `segment` = `not_supported` means no segment claims.
- No invented numbers: no probabilities, win rates or percentage confidence in prose; no fixed-percent stops or targets.

### 2. Level roles
- Follow the snapshot's `level_roles`: the `direction` level sets the bias, the `operating` level supplies the buy/sell point, the `confirmation` level times the entry. Name the level for every claim.
- Compare strength (divergence) only within one level; a higher level only confirms direction, never strength.
- A level whose `status` is not `available` is a data limitation: state it and lower confidence.

### 3. Stability states (`state`)
- `forming`: the stroke is unfinished and may change completely. Watch only.
- `candidate`: on the newest finished stroke, which can still be redrawn. At most a planned small trial position when the direction level does not oppose it; label it "candidate, awaiting confirmation".
- `confirmed`: a newer stroke exists after it, so the structure is fixed. Actionable while `still_valid=true`.
- `invalidated`: price broke `invalidation_price`. History only; the original thesis is void.
- Never upgrade a state in prose (do not call a candidate confirmed).

### 4. Divergence is a warning, not an instruction
- When citing divergence, name both compared strokes (`stroke_id` and `compared_stroke_id`) and the MACD-area ratio. If the snapshot reports none, there is none.
- Divergence alone never justifies buy/add or sell; it means "prepare". Acting needs a 2B/3B (or 2S/3S), or confirmation on the confirmation level.
- `range_divergence` gives no direction, only an expected return toward the pivot. Only `trend_divergence` can mark the end of a trend.

### 5. Decision gates (any failure -> watch / hold and observe)
1. Is the operating level's data usable?
2. Does the direction level not oppose the trade (if it does, reduce size and say so)?
3. Is there a buy point on the operating or confirmation level with `state` confirmed or candidate and `still_valid=true`?
4. Is the current price still above that point's `invalidation_price`, with a distance that fits the position size?
5. Is there no material negative news or event that overrides the structure?
Only when all pass may the action be buy/add. Sells mirror this with sell points; holders reduce or sell when the buy point they rely on becomes invalidated.

### 6. Invalidation before entry
- Stop-loss = the point's `invalidation_price` (1B/2B: the 1B low; 3B: pivot ZG; sells mirrored). Targets come from `key_levels` or pivot bounds, never from a fixed percentage.

### 7. Secondary data is context only
- Moving averages, volume, chips, fund flow, valuation and news are context: they may veto or lower confidence, never create a buy or sell on their own.
- Material negatives (shareholder reduction, expected losses, regulatory penalties, negative industry policy, large share unlocks) outrank structure. Stale or missing quote/news data lowers confidence.
"""

TECHNICAL_SKILL_RULES_EN = """## Default Skill Baseline

Treat the currently activated skills as the primary analysis lens, but keep the
following Chan theory controls as the shared baseline:

- Structural facts (strokes, pivots, divergence, buy/sell points) come only from the
  code-computed Chan structure snapshot or `get_chan_structure`; never recompute them,
  never add Chan concepts the snapshot lacks, cite the `evidence_id` for each claim
- Use `level_roles`: direction level = bias, operating level = buy/sell point,
  confirmation level = timing; compare divergence strength only within one level
- `state`: forming = watch only; candidate = can still be redrawn, small trial at most;
  confirmed = fixed; invalidated = history only. Never upgrade a state in prose
- Divergence is a warning, not a signal; range_divergence gives no direction
- Act only on a confirmed/candidate point with `still_valid=true`; no such point -> hold off
- Anchor the stop-loss to the point's `invalidation_price`; no invented probabilities or
  fixed-percent stops; partial or missing levels lower confidence
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
