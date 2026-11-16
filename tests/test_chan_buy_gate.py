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


def test_confirmation_level_point_alone_becomes_watch_and_keeps_entry_plan():
    point = dict(_point(), level="60M", evidence_id="60M.3B.2026-11-12 15:30")
    snapshot = _snapshot()
    snapshot["levels"]["60M"] = {"status": "available", "buy_sell_points": [point]}
    result = _result()
    plan = {"sniper_points": {"ideal_buy": "354.41", "stop_loss": "347.59"}}
    result.dashboard["battle_plan"] = plan
    enforce_chan_buy_gate(result, snapshot)
    assert result.decision_type == "hold"
    assert result.operation_advice == "观望"
    assert result.dashboard["chan_gate"] == {
        "applied": True, "outcome": "confirmation_level_only", "supporting_points": ["60M.3B.2026-11-12 15:30"]}
    assert "60M.3B.2026-11-12 15:30" in result.risk_warning
    assert result.dashboard["battle_plan"] == plan


def test_capital_flow_downgrade_applies_to_a_shares_only():
    from src.analyzer import stabilize_decision_with_structure

    unsupported = {"capital_flow": {"status": "not_supported", "data": {}}}
    us = _result(language="en", advice="Buy", confidence="Medium")
    stabilize_decision_with_structure(us, None, unsupported)
    assert us.decision_type == "buy" and us.operation_advice == "Buy"
    assert us.dashboard["decision_stability"]["applied"] is False

    cn = _result()
    cn.code = "600519"
    stabilize_decision_with_structure(cn, None, unsupported)
    assert cn.decision_type == "hold"


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
