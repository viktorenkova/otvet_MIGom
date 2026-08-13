from backend.app.models.user_context import UserContext, UserRole


def resolve_role(context: UserContext | None) -> UserRole:
    if context is None:
        return "guest"
    return context.resolved_role()
