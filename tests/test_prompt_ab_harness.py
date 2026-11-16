# -*- coding: utf-8 -*-
"""Prompt A/B harness: cases, metrics and the runner with a fake analyzer (no LLM)."""

from types import SimpleNamespace

from evals.prompt_ab.cases import build_cases
from evals.prompt_ab.metrics import score_result, stop_is_anchored
from evals.prompt_ab.run_ab import run_cases, summarise
from src.analyzer import AnalysisResult, enforce_chan_buy_gate


def _cases():
    return {c["id"]: c for c in build_cases()}


def test_cases_are_built_from_the_fixture_and_edit_only_their_scenario():
    cases = _cases()
    assert set(cases) == {
        "aapl_real", "no_valid_points", "forming_only", "confirmed_3b_aligned",
        "range_divergence_1b", "partial_daily_data", "bullish_news_bearish_structure",
    }
    for case in cases.values():
        assert case["context"]["chan_snapshot"]["schema_version"] == "chan_snapshot/1.1"
    none = cases["no_valid_points"]["context"]["chan_snapshot"]
    assert not any(p["still_valid"] for lv in none["levels"].values() for p in lv.get("buy_sell_points", []))
    daily_points = cases["confirmed_3b_aligned"]["context"]["chan_snapshot"]["levels"]["D"]["buy_sell_points"]
    (point,) = [p for p in daily_points if p["still_valid"]]
    assert point["state"] == "confirmed" and point["invalidation_price"] < point["price"]


def test_metrics_flag_invented_numbers_ids_and_unanchored_stops():
    case = _cases()["confirmed_3b_aligned"]
    point = [p for p in case["context"]["chan_snapshot"]["levels"]["D"]["buy_sell_points"] if p["still_valid"]][0]
    good = {
        "decision_type": "buy", "confidence_level": "Medium",
        "analysis_summary": f"Daily 3B {point['evidence_id']} confirmed.",
        "dashboard": {"battle_plan": {"sniper_points": {"stop_loss": f"{point['invalidation_price']}"}}},
    }
    assert score_result(case, good)["pass"] is True

    bad = dict(good, analysis_summary="Bottom divergence probability about 75%, see D.2B.2020-01-01.",
               dashboard={"battle_plan": {"sniper_points": {"stop_loss": "1.00 (-3%)"}}})
    scores = score_result(case, bad)
    assert scores["invented_probability"] is True
    assert scores["invented_evidence_ids"] == ["D.2B.2020-01-01"]
    assert scores["stop_anchored"] is False
    assert scores["pass"] is False

    assert stop_is_anchored("止损 100.5 元", [100.0]) is True
    assert stop_is_anchored("", [100.0]) is None


def test_runner_applies_gate_and_summarises_noise():
    cases = [c for c in build_cases() if c["id"] == "no_valid_points"]

    def fake_analyze(context, news):
        return AnalysisResult(code="AAPL", name="Apple", sentiment_score=75, trend_prediction="看多",
                              operation_advice="买入", decision_type="buy", confidence_level="高",
                              report_language="zh", current_price=context["today"]["close"], dashboard={})

    records = run_cases(fake_analyze, cases, repeats=2, gate=enforce_chan_buy_gate)
    assert len(records) == 2
    assert all(r["pre_gate_decision"] == "buy" and r["score"]["decision"] == "hold" for r in records)
    summary = summarise(records)["no_valid_points"]
    assert summary["gate_changed"] == 2 and summary["self_agreement"] == 1.0
    assert summary["decisions"] == {"hold": 2}


def test_runner_accepts_plain_objects():
    cases = [c for c in build_cases() if c["id"] == "forming_only"]
    obj = SimpleNamespace(decision_type="hold", success=True,
                          to_dict=lambda: {"decision_type": "hold", "confidence_level": "Low"})
    records = run_cases(lambda ctx, news: obj, cases, repeats=1)
    assert records[0]["score"]["decision_allowed"] is True
