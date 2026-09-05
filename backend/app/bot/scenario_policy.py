"""One access boundary for retrieval, decisions, actions and response cards."""
from typing import Any


def scenario_allowed(scenario: Any, role: str) -> bool:
    # Product-owner decision 2026-09-05: the current knowledge base contains
    # public reference information only. A role must never hide an active
    # knowledge scenario; authorization applies to real actions and personal
    # data integrations instead.
    return bool(scenario and scenario.status == "active")


def article_allowed(article: Any, role: str) -> bool:
    return bool(article)


def action_allowed(action: Any, role: str) -> bool:
    from backend.app.config import get_settings
    from backend.app.bot.scenario_engine import get_scenario

    # Status actions are a dormant extension point. They must not be offered or
    # accepted until a real protected integration is explicitly enabled.
    if action.type == "fetch_status" and not get_settings().internal_status_api_enabled:
        return False
    if action.requires_auth and role != "authorized":
        return False
    return not action.scenario_id or scenario_allowed(get_scenario(action.scenario_id), role)
