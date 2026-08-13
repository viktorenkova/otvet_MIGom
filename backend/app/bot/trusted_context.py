from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import time
from typing import Any


class TrustedContextError(ValueError):
    """A client supplied an invalid, expired, or unsupported context token."""


@dataclass(frozen=True)
class TrustedContext:
    user_id: str
    scopes: tuple[str, ...]
    expires_at: int
    email: str | None = None
    phone: str | None = None


def _decode_part(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as exc:
        raise TrustedContextError("malformed_token") from exc


def verify_trusted_context_token(
    token: str,
    secret: str,
    *,
    issuer: str = "migtorg-site",
    now: int | None = None,
) -> TrustedContext:
    """Verify a compact HMAC token issued by the MIGTORG site backend.

    Wire format: base64url(JSON payload) + '.' + base64url(HMAC-SHA256(payload)).
    The deliberately small format avoids trusting browser-controlled context and
    keeps the integration independent of a JWT package.
    """
    if not secret:
        raise TrustedContextError("trusted_context_not_configured")
    try:
        payload_part, signature_part = token.split(".", 1)
    except ValueError as exc:
        raise TrustedContextError("malformed_token") from exc

    expected = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    supplied = _decode_part(signature_part)
    if not hmac.compare_digest(expected, supplied):
        raise TrustedContextError("invalid_signature")

    try:
        payload: dict[str, Any] = json.loads(_decode_part(payload_part).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrustedContextError("malformed_payload") from exc

    current_time = int(time.time()) if now is None else int(now)
    if str(payload.get("iss") or "") != issuer:
        raise TrustedContextError("invalid_issuer")
    expires_at = int(payload.get("exp") or 0)
    if expires_at <= current_time:
        raise TrustedContextError("expired_token")
    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise TrustedContextError("missing_subject")
    raw_scopes = payload.get("scopes", [])
    if not isinstance(raw_scopes, list):
        raise TrustedContextError("invalid_scopes")
    scopes = tuple(sorted({str(scope) for scope in raw_scopes if str(scope)}))
    return TrustedContext(
        user_id=user_id,
        scopes=scopes,
        expires_at=expires_at,
        email=str(payload.get("email") or "").strip() or None,
        phone=str(payload.get("phone") or "").strip() or None,
    )

