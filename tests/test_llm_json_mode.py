# -*- coding: utf-8 -*-
"""LLM_JSON_MODE: response_format on the analysis call, with a one-shot retry when rejected."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _mod in ("litellm", "google.generativeai", "google.genai", "anthropic"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest

from src.llm.errors import call_with_json_mode_fallback, is_json_mode_unsupported


@pytest.fixture(autouse=True)
def _llm_usage_hmac_env(monkeypatch):
    monkeypatch.setenv("LLM_USAGE_HMAC_SECRET", "test-usage-hmac-secret")
    monkeypatch.setenv("LLM_USAGE_HMAC_KEY_VERSION", "test-v1")


def test_json_mode_rejection_is_detected_but_timeouts_are_not():
    assert is_json_mode_unsupported(Exception("response_format json_object is not supported"))
    assert is_json_mode_unsupported(Exception("This model does not support JSON mode"))
    assert not is_json_mode_unsupported(Exception("Request timed out while waiting for json"))
    assert not is_json_mode_unsupported(Exception("rate limit exceeded"))


def test_fallback_retries_once_without_response_format():
    seen = []

    def call(kwargs):
        seen.append(dict(kwargs))
        if "response_format" in kwargs:
            raise Exception("response_format is not supported by this model")
        return "ok"

    assert call_with_json_mode_fallback(call, {"model": "m", "response_format": {"type": "json_object"}},
                                        model="m") == "ok"
    assert ["response_format" in k for k in seen] == [True, False]


def test_fallback_reraises_other_errors_and_calls_without_json_mode_untouched():
    def boom(kwargs):
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        call_with_json_mode_fallback(boom, {"response_format": {"type": "json_object"}}, model="m")

    def reject(kwargs):
        raise Exception("response_format unsupported")

    with pytest.raises(Exception, match="unsupported"):
        call_with_json_mode_fallback(reject, {"model": "m"}, model="m")  # no response_format: no retry


def _analyzer():
    cfg = MagicMock()
    cfg.litellm_model = "analysis-route"
    cfg.litellm_fallback_models = []
    cfg.llm_model_list = [{"model_name": "analysis-route", "litellm_params": {"model": "openai/test-model"}}]
    cfg.generation_backend = "litellm"
    cfg.generation_fallback_backend = ""
    from src.analyzer import GeminiAnalyzer

    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._config_override = cfg
    return analyzer


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


@pytest.mark.parametrize("json_mode", [True, False])
def test_analysis_call_sends_response_format_only_in_json_mode(json_mode):
    analyzer = _analyzer()
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append(dict(kwargs))
        return _response('{"ok": true}')

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch):
        text, _, _ = analyzer._call_litellm_impl("prompt json", {"json_mode": json_mode, "max_output_tokens": 64},
                                                 system_prompt="system")
    assert text == '{"ok": true}'
    assert len(calls) == 1
    assert (calls[0].get("response_format") == {"type": "json_object"}) is json_mode


def test_analysis_call_drops_response_format_when_model_rejects_it():
    analyzer = _analyzer()
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append(dict(kwargs))
        if "response_format" in kwargs:
            raise Exception("Invalid parameter: response_format json_object is not supported for this model")
        return _response('{"ok": true}')

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch):
        text, _, _ = analyzer._call_litellm_impl("prompt json", {"json_mode": True, "max_output_tokens": 64},
                                                 system_prompt="system")
    assert text == '{"ok": true}'
    assert ["response_format" in c for c in calls][-2:] == [True, False]
    assert "response_format" not in calls[-1]


@pytest.mark.parametrize("raw, expected", [("16384", 16384), ("", 8192), ("12", 1024)])
def test_config_reads_analysis_max_output_tokens(monkeypatch, raw, expected):
    from src.config import Config

    monkeypatch.setenv("LLM_ANALYSIS_MAX_OUTPUT_TOKENS", raw)
    Config.reset_instance()
    try:
        assert Config._load_from_env().llm_analysis_max_output_tokens == expected
    finally:
        Config.reset_instance()


def test_analysis_max_output_tokens_ignores_non_int_config():
    from src.analyzer import _analysis_max_output_tokens

    assert _analysis_max_output_tokens(SimpleNamespace(llm_analysis_max_output_tokens=16384)) == 16384
    assert _analysis_max_output_tokens(MagicMock()) == 8192
    assert _analysis_max_output_tokens(SimpleNamespace()) == 8192


@pytest.mark.parametrize("raw, expected", [("yes", True), ("false", False)])
def test_config_reads_llm_json_mode(monkeypatch, raw, expected):
    from src.config import Config

    monkeypatch.setenv("LLM_JSON_MODE", raw)
    Config.reset_instance()
    try:
        assert Config._load_from_env().llm_json_mode is expected
    finally:
        Config.reset_instance()
