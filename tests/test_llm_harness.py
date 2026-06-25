"""Unit tests for LLM harness (no live API calls)."""
from unittest.mock import MagicMock, patch

import pytest

from app.llm_harness import LLMResult, call_llm


def test_llm_result_ok_when_no_error():
    assert LLMResult(content="hello").ok is True
    assert LLMResult(error=None).ok is True


def test_llm_result_not_ok_when_error():
    assert LLMResult(error="timeout").ok is False


def test_call_llm_success_returns_content():
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.invoke.return_value = MagicMock(content="  answer  ")

    result = call_llm(llm, "prompt", caller="test", max_retries=0)

    assert result.ok
    assert result.content == "answer"
    llm.invoke.assert_called_once_with("prompt")


def test_call_llm_retries_then_succeeds():
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.invoke.side_effect = [RuntimeError("transient"), MagicMock(content="ok")]

    with patch("app.llm_harness.time.sleep"):
        result = call_llm(llm, "prompt", caller="test", max_retries=1)

    assert result.ok
    assert result.content == "ok"
    assert llm.invoke.call_count == 2


def test_call_llm_exhausted_returns_error():
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.invoke.side_effect = RuntimeError("down")

    with patch("app.llm_harness.time.sleep"):
        result = call_llm(llm, "prompt", caller="test", max_retries=1)

    assert not result.ok
    assert "down" in result.error
    assert llm.invoke.call_count == 2


def test_call_llm_logs_request_id(caplog):
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.invoke.return_value = MagicMock(content="x")

    with patch("app.llm_harness.get_request_id", return_value="req-abc"):
        with caplog.at_level("INFO"):
            call_llm(llm, "prompt", caller="test", max_retries=0)

    assert any("request_id=req-abc" in r.message for r in caplog.records)
