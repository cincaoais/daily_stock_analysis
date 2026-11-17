# -*- coding: utf-8 -*-
"""LLM_ANALYSIS_TIMEOUT_SEC: a stalled stream fails over to the next model, not the same one."""

import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _mod in ("litellm", "google.generativeai", "google.genai", "anthropic"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest


@pytest.fixture(autouse=True)
def _llm_usage_hmac_env(monkeypatch):
    monkeypatch.setenv("LLM_USAGE_HMAC_SECRET", "test-usage-hmac-secret")
    monkeypatch.setenv("LLM_USAGE_HMAC_KEY_VERSION", "test-v1")


@pytest.mark.parametrize("raw,expected", [(None, 180), ("0", 0), ("90", 90), ("9999", 3600), ("-5", 0), ("x", 180)])
def test_config_reads_analysis_timeout(monkeypatch, raw, expected):
    from src.config import Config

    if raw is None:
        monkeypatch.delenv("LLM_ANALYSIS_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("LLM_ANALYSIS_TIMEOUT_SEC", raw)
    with patch("src.config.setup_env"):
        assert Config._load_from_env().llm_analysis_timeout_sec == expected


def test_analysis_timeout_helper_ignores_non_int_config():
    from src.analyzer import _analysis_timeout_sec

    assert _analysis_timeout_sec(SimpleNamespace(llm_analysis_timeout_sec=120)) == 120
    assert _analysis_timeout_sec(SimpleNamespace(llm_analysis_timeout_sec=0)) == 0
    assert _analysis_timeout_sec(MagicMock()) == 0
    assert _analysis_timeout_sec(SimpleNamespace()) == 0


def _analyzer():
    cfg = MagicMock()
    cfg.litellm_model = "primary-route"
    cfg.litellm_fallback_models = ["fallback-route"]
    cfg.llm_model_list = [
        {"model_name": "primary-route", "litellm_params": {"model": "openai/primary"}},
        {"model_name": "fallback-route", "litellm_params": {"model": "openai/fallback"}},
    ]
    cfg.generation_backend = "litellm"
    cfg.generation_fallback_backend = ""
    from src.analyzer import GeminiAnalyzer

    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._config_override = cfg
    return analyzer


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


class _SlowStream:
    """Yields one chunk, then trickles; records close()."""

    def __init__(self, delay):
        self.delay = delay
        self.closed = False

    def __iter__(self):
        yield _chunk('{"ok": ')
        while True:
            time.sleep(self.delay)
            yield _chunk(" ")

    def close(self):
        self.closed = True


def test_stalled_stream_moves_to_next_model_without_same_model_retry():
    analyzer = _analyzer()
    calls = []
    slow = _SlowStream(delay=0.02)

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream")), kwargs.get("timeout")))
        if model == "primary-route":
            return slow
        return iter([_chunk('{"ok": true}')])

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch):
        text, model_used, _ = analyzer._call_litellm_impl(
            "prompt json", {"max_output_tokens": 64, "timeout": 0.1, "stream_budget_sec": 0.1},
            system_prompt="system", stream=True,
        )
    assert text == '{"ok": true}'
    assert [(m, s) for m, s, _ in calls] == [("primary-route", True), ("fallback-route", True)]
    assert all(t == 0.1 for _, _, t in calls)  # per-read HTTP timeout is forwarded
    assert slow.closed is True


def test_without_budget_a_slow_stream_still_completes():
    analyzer = _analyzer()

    def stream():
        yield _chunk('{"ok": ')
        time.sleep(0.05)
        yield _chunk("true}")

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=lambda model, kwargs, **_: stream()):
        text, _, _ = analyzer._call_litellm_impl("prompt json", {"max_output_tokens": 64},
                                                 system_prompt="system", stream=True)
    assert text == '{"ok": true}'


def test_a_read_timeout_after_the_budget_counts_as_deadline():
    analyzer = _analyzer()
    calls = []

    class _TimesOut:
        def __iter__(self):
            time.sleep(0.15)
            raise TimeoutError("read timed out")

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream"))))
        if model == "primary-route":
            return _TimesOut()
        return _response('{"ok": true}') if not kwargs.get("stream") else iter([_chunk('{"ok": true}')])

    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch):
        text, _, _ = analyzer._call_litellm_impl(
            "prompt json", {"max_output_tokens": 64, "timeout": 0.1, "stream_budget_sec": 0.1},
            system_prompt="system", stream=True,
        )
    assert text == '{"ok": true}'
    assert calls == [("primary-route", True), ("fallback-route", True)]
