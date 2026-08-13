from typing import Literal

from pydantic import BaseModel, Field


UserRole = Literal["guest", "authorized"]
PageType = Literal[
    "public_site",
    "registration",
    "login",
    "tariffs",
    "lot_catalog",
    "lot_card",
    "balance",
    "payments",
    "won_lots",
    "transferred_lots",
    "support",
]


class UserContext(BaseModel):
    role: UserRole | None = None
    is_authorized: bool = False
    user_id: str | None = None
    page_type: PageType | str | None = "public_site"
    lot_id: str | None = None
    user_email: str | None = None
    user_phone: str | None = None
    session_id: str | None = None
    trusted: bool = False
    trusted_scopes: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    def resolved_role(self) -> UserRole:
        if self.role in ("guest", "authorized"):
            return self.role
        return "authorized" if self.is_authorized else "guest"
