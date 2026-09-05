from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from backend.app.bot.scenario_engine import get_scenario
from backend.app.models.chat import ChatAction


@dataclass(frozen=True)
class NavigationChoice:
    id: str
    label: str
    target_node_id: str | None = None
    scenario_id: str | None = None


@dataclass(frozen=True)
class NavigationNode:
    id: str
    label: str
    prompt: str
    parent_id: str | None
    choices: tuple[NavigationChoice, ...]


@dataclass(frozen=True)
class GuidedNavigation:
    version: str
    root_node_id: str
    nodes: dict[str, NavigationNode]

    def node(self, node_id: str) -> NavigationNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise ValueError(f"Unknown guided navigation node: {node_id}") from exc

    def actions(self, node_id: str) -> list[ChatAction]:
        node = self.node(node_id)
        actions = [self._choice_action(node, choice) for choice in node.choices]
        if node.parent_id:
            actions.append(
                ChatAction(
                    id=self._action_id(node.id, "back"),
                    type="guided_choice",
                    label="Назад",
                    payload={
                        "kind": "back",
                        "navigation_version": self.version,
                        "node_id": node.id,
                        "target_node_id": node.parent_id,
                    },
                )
            )
        actions.append(
            ChatAction(
                id=self._action_id(node.id, "free_text"),
                type="guided_choice",
                label="Написать вопрос своими словами",
                payload={
                    "kind": "free_text",
                    "navigation_version": self.version,
                    "node_id": node.id,
                },
            )
        )
        return actions

    def resolve_action(self, action_id: str) -> ChatAction | None:
        for node_id in self.nodes:
            for action in self.actions(node_id):
                if action.id == action_id:
                    return action
        return None

    def _choice_action(self, node: NavigationNode, choice: NavigationChoice) -> ChatAction:
        kind = "scenario" if choice.scenario_id else "node"
        payload: dict[str, Any] = {
            "kind": kind,
            "navigation_version": self.version,
            "node_id": node.id,
        }
        if choice.target_node_id:
            payload["target_node_id"] = choice.target_node_id
        return ChatAction(
            id=self._action_id(node.id, choice.id),
            type="guided_choice",
            label=choice.label,
            scenario_id=choice.scenario_id,
            payload=payload,
        )

    def _action_id(self, node_id: str, choice_id: str) -> str:
        return f"guided:{self.version}:{node_id}:{choice_id}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _project_root() / candidate


@lru_cache(maxsize=8)
def load_guided_navigation(path: str, max_depth: int = 2) -> GuidedNavigation:
    config_path = _resolve_path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    version = str(raw.get("version") or "").strip()
    root_node_id = str(raw.get("root_node_id") or "").strip()
    raw_nodes = raw.get("nodes")
    if not version or not root_node_id or not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Guided navigation requires version, root_node_id and nodes")

    nodes: dict[str, NavigationNode] = {}
    choice_ids: set[str] = set()
    for item in raw_nodes:
        node_id = str(item.get("id") or "").strip()
        if not node_id or node_id in nodes:
            raise ValueError(f"Duplicate or empty guided navigation node: {node_id!r}")
        raw_choices = item.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError(f"Guided navigation node has no choices: {node_id}")
        if node_id != root_node_id and len(raw_choices) > 6:
            raise ValueError(f"Guided navigation submenu has more than 6 choices: {node_id}")
        choices: list[NavigationChoice] = []
        for raw_choice in raw_choices:
            choice_id = str(raw_choice.get("id") or "").strip()
            label = str(raw_choice.get("label") or "").strip()
            target_node_id = str(raw_choice.get("target_node_id") or "").strip() or None
            scenario_id = str(raw_choice.get("scenario_id") or "").strip() or None
            unique_choice_id = f"{node_id}:{choice_id}"
            if not choice_id or not label or unique_choice_id in choice_ids:
                raise ValueError(f"Duplicate or incomplete guided choice: {unique_choice_id}")
            if bool(target_node_id) == bool(scenario_id):
                raise ValueError(f"Guided choice must target exactly one node or scenario: {unique_choice_id}")
            if scenario_id:
                scenario = get_scenario(scenario_id)
                if not scenario or scenario.status != "active":
                    raise ValueError(f"Guided choice targets an unknown or inactive scenario: {scenario_id}")
            choice_ids.add(unique_choice_id)
            choices.append(NavigationChoice(choice_id, label, target_node_id, scenario_id))
        nodes[node_id] = NavigationNode(
            id=node_id,
            label=str(item.get("label") or "").strip() or node_id,
            prompt=str(item.get("prompt") or "").strip(),
            parent_id=str(item.get("parent_id") or "").strip() or None,
            choices=tuple(choices),
        )

    if root_node_id not in nodes:
        raise ValueError("Guided navigation root node does not exist")
    if nodes[root_node_id].parent_id:
        raise ValueError("Guided navigation root node cannot have a parent")

    reached: set[str] = set()
    active: set[str] = set()

    def visit(node_id: str, depth: int) -> None:
        if depth > max_depth:
            raise ValueError(f"Guided navigation exceeds max depth at node: {node_id}")
        if node_id in active:
            raise ValueError(f"Guided navigation contains a cycle at node: {node_id}")
        if node_id in reached:
            return
        active.add(node_id)
        node = nodes[node_id]
        for choice in node.choices:
            if choice.target_node_id:
                if choice.target_node_id not in nodes:
                    raise ValueError(f"Guided choice targets an unknown node: {choice.target_node_id}")
                target = nodes[choice.target_node_id]
                if target.parent_id != node_id:
                    raise ValueError(f"Guided node parent mismatch: {target.id}")
                visit(target.id, depth + 1)
        active.remove(node_id)
        reached.add(node_id)

    visit(root_node_id, 0)
    unreachable = set(nodes) - reached
    if unreachable:
        raise ValueError(f"Unreachable guided navigation nodes: {sorted(unreachable)}")
    return GuidedNavigation(version=version, root_node_id=root_node_id, nodes=nodes)

