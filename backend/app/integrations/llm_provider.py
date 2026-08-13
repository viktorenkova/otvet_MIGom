from __future__ import annotations

import json
import time
from typing import Protocol
import urllib.error
import urllib.request

from backend.app.config import Settings
from backend.app.models.llm import LLMRequest, LLMResult


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> LLMResult:
        ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLMProvider:
    def generate(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        output_tokens = estimate_tokens(request.fallback_text)
        input_tokens = estimate_tokens(request.prompt)
        return LLMResult(
            text=request.fallback_text,
            provider="mock",
            model=request.model,
            task_type=request.task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class LiteLLMProxyProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, request: LLMRequest) -> LLMResult:
        models = [request.model]
        if request.fallback_model and request.fallback_model != request.model:
            models.append(request.fallback_model)

        last_error: str | None = None
        for model in models:
            result = self._try_model(request, model)
            if result.success:
                return result
            last_error = result.error

        return LLMResult(
            text=request.fallback_text,
            provider="litellm",
            model=models[-1],
            task_type=request.task_type,
            input_tokens=estimate_tokens(request.prompt),
            output_tokens=estimate_tokens(request.fallback_text),
            total_tokens=estimate_tokens(request.prompt) + estimate_tokens(request.fallback_text),
            estimated_cost_usd=0.0,
            latency_ms=0,
            success=False,
            error=last_error or "LiteLLM request failed",
        )

    def _try_model(self, request: LLMRequest, model: str) -> LLMResult:
        started = time.perf_counter()
        if not self.settings.litellm_proxy_url:
            return LLMResult(
                text=request.fallback_text,
                provider="litellm",
                model=model,
                task_type=request.task_type,
                success=False,
                error="LITELLM_PROXY_URL is not configured",
            )

        url = self.settings.litellm_proxy_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Вы консультант MIGTORG. Отвечайте на русском, на вы, делово и понятно. "
                        "Не добавляйте факты, которых нет в контексте. Не обещайте возвраты, отмену штрафа, "
                        "передачу лота, выигрыш или подтверждение платежа."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0.2,
        }
        if "gpt-5.6" in model:
            # Chat Completions is retained for LiteLLM compatibility. Make the
            # effective reasoning explicit so latency/cost do not silently
            # change with GPT-5.6 defaults.
            payload["reasoning_effort"] = self.settings.llm_reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.settings.litellm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.litellm_api_key}"

        try:
            http_request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(http_request, timeout=self.settings.llm_request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return LLMResult(
                text=request.fallback_text,
                provider="litellm",
                model=model,
                task_type=request.task_type,
                success=False,
                error=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        text = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", request.fallback_text)
        )
        cost = float(data.get("response_cost") or data.get("_hidden_params", {}).get("response_cost") or 0.0)
        return LLMResult(
            text=text,
            provider="litellm",
            model=model,
            task_type=request.task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class QwenProvider:
    """Direct client for Alibaba Cloud Model Studio's OpenAI-compatible API."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, request: LLMRequest) -> LLMResult:
        models = [request.model]
        if request.fallback_model and request.fallback_model != request.model:
            models.append(request.fallback_model)

        last_result: LLMResult | None = None
        for model in models:
            last_result = self._try_model(request, model)
            if last_result.success:
                return last_result
        return last_result or self._failure(request, request.model, "Qwen request failed")

    def _failure(
        self,
        request: LLMRequest,
        model: str,
        error: str,
        latency_ms: int = 0,
    ) -> LLMResult:
        input_tokens = estimate_tokens(request.prompt)
        output_tokens = estimate_tokens(request.fallback_text)
        return LLMResult(
            text=request.fallback_text,
            provider="qwen",
            model=model,
            task_type=request.task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=0.0,
            latency_ms=latency_ms,
            success=False,
            error=error,
        )

    def _try_model(self, request: LLMRequest, model: str) -> LLMResult:
        started = time.perf_counter()
        if not self.settings.qwen_base_url:
            return self._failure(request, model, "QWEN_BASE_URL is not configured")
        if not self.settings.qwen_api_key:
            return self._failure(request, model, "QWEN_API_KEY is not configured")

        url = self.settings.qwen_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Вы консультант MIGTORG. Отвечайте на русском, на вы, делово и понятно. "
                        "Используйте только факты из переданного контекста базы знаний. "
                        "Если контекста недостаточно, прямо скажите об этом. Не обещайте возврат, "
                        "отмену штрафа, передачу лота, выигрыш или подтверждение платежа."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 700,
            # Answer rewriting is a low-complexity task; thinking only adds
            # latency and spends the trial quota without improving grounding.
            "enable_thinking": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }

        try:
            http_request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(
                http_request,
                timeout=self.settings.llm_request_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return self._failure(
                request,
                model,
                f"Qwen HTTP {exc.code}",
                int((time.perf_counter() - started) * 1000),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return self._failure(
                request,
                model,
                str(exc),
                int((time.perf_counter() - started) * 1000),
            )

        choice = (data.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            return self._failure(
                request,
                model,
                "Qwen returned an empty response",
                int((time.perf_counter() - started) * 1000),
            )

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        return LLMResult(
            text=text.strip(),
            provider="qwen",
            model=model,
            task_type=request.task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "litellm":
        return LiteLLMProxyProvider(settings)
    if settings.llm_provider == "qwen":
        return QwenProvider(settings)
    return MockLLMProvider()
