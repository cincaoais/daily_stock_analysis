# -*- coding: utf-8 -*-
"""
TechnicalAgent — technical & price analysis specialist.

Responsible for:
- Fetching realtime quotes and historical K-line data
- Running technical indicators (trend, MA, volume, pattern)
- Producing a structured opinion on trend/momentum/support-resistance
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    agent_name = "technical"
    max_steps = 6
    tool_names = [
        "get_realtime_quote",
        "get_daily_history",
        "analyze_trend",
        "get_chan_structure",
        "calculate_ma",
        "get_volume_analysis",
        "analyze_pattern",
        "get_chip_distribution",
        "get_analysis_context",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Skills\n\n{self.skill_instructions}\n"
        baseline = ""
        if self.technical_skill_policy:
            baseline = f"\n{self.technical_skill_policy}\n"

        return f"""\
You are a **Technical Analysis Agent** specialising in Chinese A-shares, \
Hong Kong stocks, and US equities.

Your task: perform a thorough technical analysis of the given stock and \
output a structured JSON opinion.

## Workflow (execute stages in order)
1. Fetch realtime quote + daily history (if not already provided)
2. Read the Chan structure snapshot (pre-fetched `chan_snapshot`, otherwise call
   `get_chan_structure`): per level, the trend type, current stroke, pivots, divergence
   and still-valid buy/sell points. This is the primary evidence; never recompute it.
3. Run trend analysis (MA alignment, MACD, RSI) as supporting indicators only
4. Analyse volume and chip distribution
5. Derive `signal` from the Chan structure using `level_roles`: a still-valid buy point
   with `state` confirmed on the operating level and a direction level that does not
   oppose it supports buy; `candidate` caps the signal at buy with lower confidence;
   `forming` points and divergence alone mean hold. A still-valid sell point or an
   invalidated buy point supports sell; no valid point means hold. Set `stop_loss` to the
   relevant point's `invalidation_price`, never a fixed percentage.

{baseline}
{skills}
## Output Format
Return **only** a JSON object (no markdown fences):
{{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary",
  "key_levels": {{
    "support": <float>,
    "resistance": <float>,
    "stop_loss": <float>
  }},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  "volume_status": "heavy|normal|light",
  "pattern": "<detected pattern or none>",
  "chan_trend_type": "up|down|consolidation|undetermined (daily level, from the snapshot)",
  "chan_point": "<evidence_id of the still-valid buy/sell point used, or none>",
  "chan_point_state": "forming|candidate|confirmed|invalidated|none"
}}
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"Perform technical analysis on stock **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("Use your tools to fetch any missing data, then output the JSON opinion.")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Parse the JSON opinion from the LLM response."""
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[TechnicalAgent] failed to parse opinion JSON")
            return None

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            key_levels={
                k: float(v) for k, v in parsed.get("key_levels", {}).items()
                if isinstance(v, (int, float))
            },
            raw_data=parsed,
        )

