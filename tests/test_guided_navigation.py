from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.bot.dialog_logger import DialogLogger
from backend.app.bot.guided_navigation import load_guided_navigation


CONFIG_PATH = "configs/guided_navigation.v1.json"


@pytest.fixture()
def guided_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "guided_navigation_enabled", True)
    monkeypatch.setattr(main.settings, "guided_navigation_rollout_percentage", 100)
    monkeypatch.setattr(main.settings, "guided_navigation_config_path", CONFIG_PATH)
    monkeypatch.setattr(main.settings, "guided_navigation_max_depth", 2)
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", False)
    monkeypatch.setattr(main, "logger", DialogLogger(str(tmp_path / "guided.sqlite3")))
    load_guided_navigation.cache_clear()
    yield TestClient(main.app)
    load_guided_navigation.cache_clear()


def _choose(client, session_id, response, *, label):
    data = response.json()
    action = next(item for item in data["actions"] if item["label"] == label)
    return client.post(
        "/api/chat/message",
        json={
            "message": action["label"],
            "session_id": session_id,
            "selected_action_id": action["id"],
            "conversation_turn_id": data["message_id"],
        },
    )


def test_navigation_config_has_exactly_30_terminal_scenarios():
    navigation = load_guided_navigation(CONFIG_PATH, 2)
    scenario_ids = {
        choice.scenario_id
        for node in navigation.nodes.values()
        for choice in node.choices
        if choice.scenario_id
    }

    assert len(scenario_ids) == 30
    assert navigation.root_node_id == "root"
    assert all(len(node.choices) <= 6 for node in navigation.nodes.values() if node.id != "root")


def test_start_returns_server_owned_guided_root(guided_client):
    response = guided_client.post("/api/chat/start", json={"session_id": "guided-start"})

    assert response.status_code == 200
    data = response.json()
    assert data["experience_variant"] == "guided"
    assert data["navigation_version"] == "v1"
    assert data["navigation_node_id"] == "root"
    assert data["state_version"] == 1
    assert data["actions"][0]["label"] == "Начать пользоваться MIGTORG"
    assert all(action["type"] == "guided_choice" for action in data["actions"])


def test_terminal_guided_choice_bypasses_retrieval(guided_client, monkeypatch):
    session_id = "guided-terminal"
    root = guided_client.post("/api/chat/start", json={"session_id": session_id})
    submenu = _choose(guided_client, session_id, root, label="Ставки и торги")
    assert submenu.status_code == 200
    assert submenu.json()["navigation_node_id"] == "bidding"

    def fail_search(*args, **kwargs):
        raise AssertionError("guided terminal choice must not invoke retrieval")

    monkeypatch.setattr(main, "search_knowledge_match", fail_search)
    result = _choose(guided_client, session_id, submenu, label="Как сделать ставку")

    assert result.status_code == 200
    assert result.json()["scenario_id"] == "bid.place"
    assert result.json()["confidence_level"] == "high"


def test_all_30_guided_paths_open_the_configured_scenario(guided_client):
    navigation = load_guided_navigation(CONFIG_PATH, 2)
    root = navigation.node(navigation.root_node_id)
    terminals = [
        (node, choice)
        for node in navigation.nodes.values()
        for choice in node.choices
        if choice.scenario_id
    ]

    for index, (node, choice) in enumerate(terminals):
        session_id = f"guided-all-{index}"
        response = guided_client.post("/api/chat/start", json={"session_id": session_id})
        if node.id != root.id:
            root_choice = next(item for item in root.choices if item.target_node_id == node.id)
            response = _choose(guided_client, session_id, response, label=root_choice.label)
        response = _choose(guided_client, session_id, response, label=choice.label)

        assert response.status_code == 200
        assert response.json()["scenario_id"] == choice.scenario_id

    metrics = main.logger.get_guided_navigation_metrics(1)
    terminal_events = next(
        item for item in metrics["events"]
        if item["variant"] == "guided" and item["event_type"] == "terminal_selected"
    )
    assert terminal_events["count"] == 30


def test_action_not_issued_for_session_is_rejected(guided_client):
    issued = guided_client.post("/api/chat/start", json={"session_id": "owner"}).json()["actions"][0]

    response = guided_client.post(
        "/api/chat/message",
        json={
            "message": issued["label"],
            "session_id": "other-session",
            "selected_action_id": issued["id"],
        },
    )

    assert response.status_code == 200
    assert response.json()["confidence_level"] == "low"
    assert response.json()["scenario_id"] is None
    assert "больше не относится" in response.json()["answer"]


def test_repeated_guided_click_is_stale_after_first_transition(guided_client):
    session_id = "repeated-guided-click"
    root = guided_client.post("/api/chat/start", json={"session_id": session_id})
    data = root.json()
    action = next(item for item in data["actions"] if item["label"] == "Ставки и торги")
    payload = {
        "message": action["label"],
        "session_id": session_id,
        "selected_action_id": action["id"],
        "conversation_turn_id": data["message_id"],
    }

    first = guided_client.post("/api/chat/message", json=payload)
    repeated = guided_client.post("/api/chat/message", json=payload)

    assert first.json()["navigation_node_id"] == "bidding"
    assert repeated.json()["confidence_level"] == "low"
    assert "больше не относится" in repeated.json()["answer"]


def test_back_returns_to_root(guided_client):
    session_id = "guided-back"
    root = guided_client.post("/api/chat/start", json={"session_id": session_id})
    submenu = _choose(guided_client, session_id, root, label="Возврат или штраф")
    back = _choose(guided_client, session_id, submenu, label="Назад")

    assert back.status_code == 200
    assert back.json()["navigation_node_id"] == "root"
    assert any(action["label"] == "Ставки и торги" for action in back.json()["actions"])


def test_guided_terminal_updates_persistent_dialogue_state(guided_client, monkeypatch):
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", True)
    session_id = "guided-dialogue"
    root = guided_client.post("/api/chat/start", json={"session_id": session_id})
    submenu = _choose(guided_client, session_id, root, label="Осмотр и получение автомобиля")
    answer = _choose(guided_client, session_id, submenu, label="Как получить лот")

    state = main.logger.load_dialogue_state(session_id)
    assert answer.json()["scenario_id"] == "pickup.receive_lot"
    assert state.active_scenario_id == "pickup.receive_lot"
    assert state.version == answer.json()["state_version"]


def test_disabled_navigation_returns_control_start(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "guided_navigation_enabled", False)
    monkeypatch.setattr(main, "logger", DialogLogger(str(tmp_path / "control.sqlite3")))

    response = TestClient(main.app).post("/api/chat/start", json={"session_id": "control"})

    assert response.status_code == 200
    data = response.json()
    assert data["experience_variant"] == "control"
    assert [action["payload"]["message"] for action in data["actions"]] == [
        "Лот не передают",
        "Не могу сделать ставку",
        "Оплатил тариф, доступа нет",
        "Вопрос по штрафу или депозиту",
    ]


def test_validator_rejects_unknown_scenario(tmp_path):
    source = Path(CONFIG_PATH)
    payload = json.loads(source.read_text(encoding="utf-8"))
    broken = deepcopy(payload)
    broken["nodes"][0]["choices"][-1] = {
        "id": "broken",
        "label": "Broken",
        "scenario_id": "missing.scenario",
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    load_guided_navigation.cache_clear()

    with pytest.raises(ValueError, match="unknown or inactive scenario"):
        load_guided_navigation(str(path), 2)
