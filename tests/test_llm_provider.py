import json

from backend.app.config import Settings
from backend.app.integrations.llm_provider import QwenProvider, build_llm_provider
from backend.app.models.llm import LLMRequest


def _request() -> LLMRequest:
    return LLMRequest(
        prompt="Ответьте только по контексту.",
        fallback_text="Безопасный ответ из БЗ.",
        provider="qwen",
        model="qwen-plus",
        task_type="answer_generation",
        session_id="test-session",
        user_role="guest",
    )


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "Ответ Qwen по БЗ."}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            }
        ).encode("utf-8")


def test_qwen_provider_calls_openai_compatible_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["authorization"] = http_request.headers["Authorization"]
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    settings = Settings(
        llm_provider="qwen",
        qwen_base_url="https://workspace.example/compatible-mode/v1/",
        qwen_api_key="test-secret",
        llm_request_timeout_seconds=9,
    )

    result = build_llm_provider(settings).generate(_request())

    assert result.success is True
    assert result.text == "Ответ Qwen по БЗ."
    assert result.provider == "qwen"
    assert result.total_tokens == 19
    assert captured["url"] == "https://workspace.example/compatible-mode/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-secret"
    assert captured["timeout"] == 9
    assert captured["payload"]["model"] == "qwen-plus"
    assert captured["payload"]["enable_thinking"] is False


def test_qwen_provider_fails_closed_without_key():
    provider = QwenProvider(
        Settings(
            llm_provider="qwen",
            qwen_base_url="https://workspace.example/compatible-mode/v1",
        )
    )

    result = provider.generate(_request())

    assert result.success is False
    assert result.text == "Безопасный ответ из БЗ."
    assert result.error == "QWEN_API_KEY is not configured"
