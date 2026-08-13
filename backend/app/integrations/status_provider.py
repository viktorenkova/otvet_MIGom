from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Protocol
from urllib import error, parse, request

from backend.app.config import Settings


@dataclass(frozen=True)
class StatusResult:
    success: bool
    kind: str
    status: str = ""
    description: str = ""
    freshness: str | None = None
    allowed_actions: list[str] = field(default_factory=list)
    error_code: str = ""


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
        query = parse.urlencode({"reference_id": reference_id})
        url = f"{self.settings.internal_status_api_url.rstrip('/')}/v1/status/{kind}?{query}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "X-MIGTORG-User": user_id,
        }
        req = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=self.settings.internal_status_timeout_seconds) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            code = "not_found" if exc.code == 404 else "forbidden" if exc.code == 403 else "upstream_error"
            return StatusResult(False, kind, error_code=code)
        except (error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            return StatusResult(False, kind, error_code="upstream_unavailable")

        return StatusResult(
            success=True,
            kind=kind,
            status=str(body.get("status") or "unknown"),
            description=str(body.get("description") or "Статус получен из системы MIGTORG."),
            freshness=str(body.get("freshness") or datetime.now(timezone.utc).isoformat()),
            allowed_actions=[str(item) for item in body.get("allowed_actions", []) if str(item)],
        )


def build_status_provider(settings: Settings) -> StatusProvider:
    if settings.internal_status_api_enabled:
        return InternalApiStatusProvider(settings)
    return DisabledStatusProvider()

