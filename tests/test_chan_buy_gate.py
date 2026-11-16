# -*- coding: utf-8 -*-
"""Post-generation Chan gate: a buy needs a still-valid confirmed/candidate buy point."""

from src.analyzer import AnalysisResult, enforce_chan_buy_gate


def _result(decision_type="buy", advice="买入", confidence="高", price=340.0, language="zh"):
    return AnalysisResult(
        code="AAPL",
        name="Apple",
        sentiment_score=78,
        trend_prediction="看多",
        operation_advice=advice,
        decision_type=decision_type,
        confidence_level=confidence,
        report_language=language,
        current_price=price,
        dashboard={"core_conclusion": {"one_sentence": "原始结论"}},
    )


def _point(state="confirmed", still_valid=True, side="buy", invalidation=334.99, kind="3B"):
    return {
        "evidence_id": "D.%s.2026-10-20" % kind, "type": kind, "side": side,
        "state": state, "still_valid": still_valid, "invalidation_price": invalidation,
    }


def _snapshot(*points, roles=None):
    return {
        "level_roles": roles or {"direction": "W", "operating": "D", "confirmation": "60M"},
        "levels": {"D": {"status": "available", "buy_sell_points": list(points)}},
    }


def test_buy_without_supporting_point_is_downgraded_to_watch():
    result = _result()
    enforce_chan_buy_gate(result, _snapshot())
    assert result.decision_type == "hold"
    assert result.operation_advice == "观望"
    assert 45 <= result.sentiment_score <= 59
    assert result.dashboard["chan_gate"]["outcome"] == "downgraded_to_watch"
    assert "缠论快照" in result.risk_warning


def test_forming_invalidated_sell_or_broken_points_do_not_support_a_buy():
    for point, price in (
        (_point(state="forming"), 340.0),
        (_point(state="invalidated", still_valid=False), 340.0),
        (_point(side="sell", kind="3S"), 340.0),
        (_point(), 330.0),  # price already below the invalidation price
    ):
        result = _result(price=price)
        enforce_chan_buy_gate(result, _snapshot(point))
        assert result.decision_type == "hold", point


def test_confirmed_point_leaves_the_buy_untouched():
    result = _result()
    enforce_chan_buy_gate(result, _snapshot(_point()))
    assert result.decision_type == "buy"
    assert result.confidence_level == "高"
    assert result.dashboard["chan_gate"] == {
        "applied": False, "outcome": "supported", "supporting_points": ["D.3B.2026-10-20"]}


def test_candidate_only_caps_confidence_and_warns():
    result = _result(language="en", advice="Buy", confidence="High")
    enforce_chan_buy_gate(result, _snapshot(_point(state="candidate")))
    assert result.decision_type == "buy"
    assert result.confidence_level == "Medium"
    assert "candidate" in result.risk_warning
    assert result.dashboard["chan_gate"]["outcome"] == "candidate_only"


def test_gate_ignores_non_buy_calls_and_missing_snapshot():
    hold = _result(decision_type="hold", advice="持有")
    enforce_chan_buy_gate(hold, _snapshot())
    assert hold.decision_type == "hold" and "chan_gate" not in hold.dashboard

    no_snapshot = _result()
    enforce_chan_buy_gate(no_snapshot, None)
    assert no_snapshot.decision_type == "buy"

    unusable = _result()
    enforce_chan_buy_gate(unusable, _snapshot(roles={"direction": None, "operating": None, "confirmation": None}))
    assert unusable.decision_type == "buy"
