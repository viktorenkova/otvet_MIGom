from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Protocol
from urllib import error, parse, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from backend.app.config import Settings


class StatusPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    freshness: str | None = None
    allowed_actions: list[str] = Field(default_factory=list, max_length=30)

@dataclass(frozen=True)
class StatusResult:
    success: bool
    kind: str
    status: str = ""
    description: str = ""
    freshness: str | None = None
    allowed_actions: list[str] = field(default_factory=list)
    error_code: str = ""
    received_at: str | None = None


class StatusProvider(Protocol):
    def fetch(self, kind: str, user_id: str, reference_id: str, access_token: str) -> StatusResult:
        ...


class DisabledStatusProvider:
    def fetch(self, kind: str, user_id: str, reference_id: str, access_token: str) -> StatusResult:
        return StatusResult(False, kind, error_code="status_provider_unavailable")


class InternalApiStatusProvider:
    ALLOWED_KINDS = {"lot", "bid", "auction", "payment", "tariff", "documents", "transfer"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, kind: str, user_id: str, reference_id: str, access_token: str) -> StatusResult:
        if kind not in self.ALLOWED_KINDS:
            return StatusResult(False, kind, error_code="unsupported_status_kind")
        if not self.settings.internal_status_api_url:
            return StatusResult(False, kind, error_code="status_provider_unavailable")
        if not user_id or not reference_id or not access_token:
            return StatusResult(False, kind, error_code="missing_status_parameters")
        query = parse.urlencode({"reference_id": reference_id})
        url = f"{self.settings.internal_status_api_url.rstrip('/')}/v1/status/{kind}?{query}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "X-MIGTORG-User": user_id,
        }
        req = request.Request(url, headers=headers, method="GET")
        timeout = min(4.0, self.settings.internal_status_timeout_seconds)
        from backend.app.bot.architecture_decision import decision_context
        import time
        ctx = decision_context.get()
        if ctx:
            timeout = min(timeout, ctx["deadline"] - time.monotonic() - 0.2)
        if timeout <= 0:
            return StatusResult(False, kind, error_code="deadline_exceeded")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read(65537)
                if len(raw) > 65536:
                    return StatusResult(False, kind, error_code="invalid_status_payload")
                body = StatusPayload.model_validate(json.loads(raw.decode("utf-8")))
                if body.freshness:
                    stamp = datetime.fromisoformat(body.freshness.replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        raise ValueError("freshness must have timezone")
        except error.HTTPError as exc:
            code = "not_found" if exc.code == 404 else "forbidden" if exc.code in {401,403} else "upstream_error"
            return StatusResult(False, kind, error_code=code)
        except (ValidationError, ValueError, UnicodeDecodeError):
            return StatusResult(False, kind, error_code="invalid_status_payload")
        except (error.URLError, TimeoutError, OSError):
            return StatusResult(False, kind, error_code="upstream_unavailable")
        if body.status.strip().casefold() in {"", "unknown", "неизвестно"}:
            return StatusResult(False, kind, error_code="unknown_status", received_at=datetime.now(timezone.utc).isoformat())
        return StatusResult(
            success=True,
            kind=kind,
            status=body.status,
            description=body.description,
            freshness=body.freshness,
            allowed_actions=body.allowed_actions,
            received_at=datetime.now(timezone.utc).isoformat(),
        )


def build_status_provider(settings: Settings) -> StatusProvider:
    if settings.internal_status_api_enabled:
        return InternalApiStatusProvider(settings)
    return DisabledStatusProvider()
