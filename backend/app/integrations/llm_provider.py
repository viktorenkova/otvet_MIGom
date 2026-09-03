from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Lock
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


def estimate_cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    return round(
        (
            input_tokens * settings.llm_input_cost_per_million_usd
            + output_tokens * settings.llm_output_cost_per_million_usd
        )
        / 1_000_000,
        8,
    )


def _correlation_id(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24] if session_id else ""


@dataclass
class _ProviderGuard:
    max_concurrency: int
    semaphore: BoundedSemaphore = field(init=False)
    failures: int = 0
    opened_until: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.semaphore = BoundedSemaphore(self.max_concurrency)

    def enter(self) -> str:
        with self.lock:
            if self.opened_until > time.monotonic():
                return "circuit_open"
        if not self.semaphore.acquire(blocking=False):
            return "concurrency_limit"
        return ""

    def leave(self) -> None:
        self.semaphore.release()

    def record_success(self) -> None:
        with self.lock:
            self.failures = 0
            self.opened_until = 0.0

    def record_failure(self, settings: Settings) -> None:
        with self.lock:
            self.failures += 1
            if self.failures >= settings.llm_circuit_failure_threshold:
                self.opened_until = time.monotonic() + settings.llm_circuit_cooldown_seconds


_GUARDS: dict[tuple[str, str, int], _ProviderGuard] = {}
_GUARDS_LOCK = Lock()


def _provider_guard(provider: str, endpoint: str, max_concurrency: int) -> _ProviderGuard:
    key = (provider, endpoint, max_concurrency)
    with _GUARDS_LOCK:
        return _GUARDS.setdefault(key, _ProviderGuard(max_concurrency))


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
            environment="dev",
            correlation_id=_correlation_id(request.session_id),
        )


class LiteLLMProxyProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.guard = _provider_guard("litellm", settings.litellm_proxy_url, settings.llm_max_concurrency)

    def generate(self, request: LLMRequest) -> LLMResult:
        guard_error = self.guard.enter()
        if guard_error:
            return self._failure(request, request.model, guard_error)
        deadline = time.monotonic() + self.settings.llm_total_timeout_seconds
        models = [request.model]
        if request.fallback_model and request.fallback_model != request.model:
            models.append(request.fallback_model)

        last_result: LLMResult | None = None
        try:
            for model in models:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_result = self._failure(request, model, "total_timeout")
                    break
                last_result = self._try_model(
                    request,
                    model,
                    min(float(self.settings.llm_request_timeout_seconds), remaining),
                )
                if last_result.success:
                    self.guard.record_success()
                    return last_result
            self.guard.record_failure(self.settings)
            return last_result or self._failure(request, models[-1], "LiteLLM request failed")
        finally:
            self.guard.leave()

    def _failure(self, request: LLMRequest, model: str, error: str, latency_ms: int = 0) -> LLMResult:
        input_tokens = estimate_tokens(request.prompt)
        output_tokens = estimate_tokens(request.fallback_text)
        return LLMResult(
            text=request.fallback_text,
            provider="litellm",
            model=model,
            task_type=request.task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=0.0,
            latency_ms=latency_ms,
            success=False,
            error=error,
            environment=self.settings.llm_environment,
            fallback_used=True,
            correlation_id=_correlation_id(request.session_id),
        )

    def _try_model(self, request: LLMRequest, model: str, timeout_seconds: float) -> LLMResult:
        started = time.perf_counter()
        if not self.settings.litellm_proxy_url:
            return self._failure(request, model, "LITELLM_PROXY_URL is not configured")

        url = self.settings.litellm_proxy_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Вы консультант MIGTORG. Отвечайте на русском, на вы, делово и понятно. "
                        "Давайте прямой ответ в 2–4 коротких предложениях, без повторов и сведений, не относящихся к вопросу. "
                        "Не добавляйте факты, которых нет в контексте, и никогда не называйте имена сотрудников. "
                        "Не обещайте возвраты, отмену штрафа, передачу лота, выигрыш или подтверждение платежа."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0.2,
            "max_tokens": self.settings.llm_max_output_tokens,
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
            with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return self._failure(request, model, str(exc), int((time.perf_counter() - started) * 1000))

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
        if cost <= 0:
            cost = estimate_cost(self.settings, input_tokens, output_tokens)
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
            environment=self.settings.llm_environment,
            correlation_id=_correlation_id(request.session_id),
        )


class QwenProvider:
    """Direct client for Alibaba Cloud Model Studio's OpenAI-compatible API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.guard = _provider_guard("qwen", settings.qwen_base_url, settings.llm_max_concurrency)

    def generate(self, request: LLMRequest) -> LLMResult:
        guard_error = self.guard.enter()
        if guard_error:
            return self._failure(request, request.model, guard_error)
        deadline = time.monotonic() + self.settings.llm_total_timeout_seconds
        models = [request.model]
        if request.fallback_model and request.fallback_model != request.model:
            models.append(request.fallback_model)

        last_result: LLMResult | None = None
        try:
            for model in models:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_result = self._failure(request, model, "total_timeout")
                    break
                last_result = self._try_model(
                    request,
                    model,
                    min(float(self.settings.llm_request_timeout_seconds), remaining),
                )
                if last_result.success:
                    self.guard.record_success()
                    return last_result
            self.guard.record_failure(self.settings)
            return last_result or self._failure(request, request.model, "Qwen request failed")
        finally:
            self.guard.leave()

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
            environment=self.settings.llm_environment,
            fallback_used=True,
            correlation_id=_correlation_id(request.session_id),
        )

    def _try_model(self, request: LLMRequest, model: str, timeout_seconds: float) -> LLMResult:
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
                        "Давайте прямой ответ в 2–4 коротких предложениях, без повторов и сведений, не относящихся к вопросу. "
                        "Используйте только факты из переданного контекста базы знаний и никогда не называйте имена сотрудников. "
                        "Если контекста недостаточно, прямо скажите об этом. Не обещайте возврат, "
                        "отмену штрафа, передачу лота, выигрыш или подтверждение платежа."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0.1,
            "max_tokens": self.settings.llm_max_output_tokens,
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
                timeout=timeout_seconds,
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
            estimated_cost_usd=estimate_cost(self.settings, input_tokens, output_tokens),
            latency_ms=int((time.perf_counter() - started) * 1000),
            environment=self.settings.llm_environment,
            correlation_id=request.session_id,
        )


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "litellm":
        return LiteLLMProxyProvider(settings)
    if settings.llm_provider == "qwen":
        return QwenProvider(settings)
    if settings.llm_provider == "mock" and not settings.llm_enabled:
        return MockLLMProvider()
    raise ValueError(f"Unsupported enabled LLM provider: {settings.llm_provider}")
