"""One access boundary for retrieval, decisions, actions and response cards."""
from typing import Any


def scenario_allowed(scenario: Any, role: str) -> bool:
    return bool(scenario and role in scenario.roles and scenario.status == "active")


def article_allowed(article: Any, role: str) -> bool:
    if article.roles:
        return role in article.roles
    # Legacy public articles have no scenario roles.
    return article.section in ({"public", "guest"} if role == "guest" else {"public", "guest", "authorized"})


def action_allowed(action: Any, role: str) -> bool:
    from backend.app.bot.scenario_engine import get_scenario
    if action.requires_auth and role != "authorized":
        return False
    return not action.scenario_id or scenario_allowed(get_scenario(action.scenario_id), role)
