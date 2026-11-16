# -*- coding: utf-8 -*-
"""Automated scoring of one analysis result against a frozen Chan scenario.

Pure functions only (no LLM, no network), so they can re-score saved runs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set

_EVIDENCE_RE = re.compile(r"\b(?:W|D|60M|30M)\.[123][BS]\.\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?")
# A number with a percent sign next to a probability/win-rate/confidence word, either order.
_PROB_WORDS = r"(?:概率|胜率|可能性|置信度|probability|chance|odds|win rate|likelihood|confidence)"
_INVENTED_PROB_RE = re.compile(
    rf"{_PROB_WORDS}[^。.;；\n]{{0,12}}\d{{1,3}}(?:\.\d+)?\s*%|\d{{1,3}}(?:\.\d+)?\s*%[^。.;；\n]{{0,6}}{_PROB_WORDS}",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

TEXT_FIELDS = (
    "analysis_summary", "key_points", "risk_warning", "buy_reason", "trend_analysis",
    "technical_analysis", "short_term_outlook", "medium_term_outlook", "operation_advice",
)


def snapshot_evidence_ids(snapshot: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for level in (snapshot.get("levels") or {}).values():
        for point in level.get("buy_sell_points") or []:
            if point.get("evidence_id"):
                ids.add(point["evidence_id"])
    return ids


def unknown_evidence_ids(cited: Iterable[str], known: Set[str]) -> List[str]:
    """Cited ids not in the snapshot. A date-only form of a known intraday id
    (``60M.1S.2026-11-12`` for ``60M.1S.2026-11-12 09:30``) counts as known."""
    date_only = {k.split(" ")[0] for k in known}
    return sorted(c for c in cited if c not in known and c not in date_only)


def _all_text(result: Dict[str, Any]) -> str:
    parts: List[str] = [str(result.get(k) or "") for k in TEXT_FIELDS]
    parts.append(json.dumps(result.get("dashboard") or {}, ensure_ascii=False))
    return "\n".join(parts)


def _stop_loss_text(result: Dict[str, Any]) -> str:
    dashboard = result.get("dashboard") or {}
    sniper = ((dashboard.get("battle_plan") or {}).get("sniper_points") or {})
    return str(sniper.get("stop_loss") or "")


def _prices_in(text: str) -> List[float]:
    out = []
    for raw in _NUMBER_RE.findall(text):
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def stop_is_anchored(stop_text: str, invalidation_prices: Iterable[float], tol: float = 0.01) -> Optional[bool]:
    """True when a number in the stop-loss text is within ``tol`` of an invalidation price."""
    targets = [p for p in invalidation_prices if p]
    if not targets or not stop_text.strip():
        return None
    return any(abs(v - t) / t <= tol for v in _prices_in(stop_text) for t in targets)


def score_result(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Score one result dict (``AnalysisResult.to_dict()``) against a case's expectations."""
    expect = case.get("expect") or {}
    snapshot = case["context"].get("chan_snapshot") or {}
    text = _all_text(result)
    known_ids = snapshot_evidence_ids(snapshot)
    cited = set(_EVIDENCE_RE.findall(text))
    decision = str(result.get("decision_type") or "").lower()
    confidence = str(result.get("confidence_level") or "").lower()

    allowed = expect.get("allowed_decisions")
    invalidations = [
        float(p["invalidation_price"])
        for level in (snapshot.get("levels") or {}).values()
        for p in level.get("buy_sell_points") or []
        if p.get("still_valid") and p.get("invalidation_price") is not None
    ]
    scores = {
        "success": bool(result.get("success", True)),
        "decision": decision,
        "decision_allowed": None if not allowed else decision in allowed,
        "confidence_not_high": None if not expect.get("confidence_not_high") else confidence not in ("high", "高", "높음"),
        "cites_evidence": bool(cited & known_ids) if known_ids else None,
        "invented_evidence_ids": unknown_evidence_ids(cited, known_ids),
        "invented_probability": bool(_INVENTED_PROB_RE.search(text)),
        "stop_anchored": stop_is_anchored(_stop_loss_text(result), invalidations) if decision == "buy" else None,
        "chan_gate": ((result.get("dashboard") or {}).get("chan_gate") or {}).get("outcome"),
    }
    must_mention = expect.get("must_mention") or []
    if must_mention:
        lowered = text.lower()
        scores["mentions_required"] = all(any(alt.lower() in lowered for alt in item.split("|")) for item in must_mention)
    checks = [v for k, v in scores.items()
              if k in ("decision_allowed", "confidence_not_high", "cites_evidence", "stop_anchored", "mentions_required")
              and v is not None]
    violations = int(scores["invented_probability"]) + len(scores["invented_evidence_ids"])
    scores["pass"] = scores["success"] and all(checks) and violations == 0
    return scores
