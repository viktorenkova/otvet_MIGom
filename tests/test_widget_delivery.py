from fastapi.testclient import TestClient

from backend.app.main import app


def test_widget_files_are_not_cached_after_ui_updates():
    client = TestClient(app)

    for path in ("/widget/", "/widget/widget.js", "/widget/style.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, max-age=0"


def test_served_widget_contains_ticket_description_clear_action():
    response = TestClient(app).get("/widget/")

    assert "20260905-guided-navigation" in response.text
    assert "Очистить и написать своё" in response.text
    assert '<select name="topic">' in response.text
    assert "Удобное время звонка" not in response.text
    assert "preferred_callback_time" not in response.text
    assert 'class="chat__home"' in response.text
    assert 'class="chat__navigation" aria-label="Навигация по чату" hidden' in response.text
    assert "Вернуться в начало и начать новый диалог" in response.text


def test_widget_script_submits_written_support_only():
    response = TestClient(app).get("/widget/widget.js")

    assert response.status_code == 200
    assert 'category: "support"' in response.text
    assert "preferred_callback_time" not in response.text
    assert "request_callback:" not in response.text
    assert 'postJson("/api/chat/start"' in response.text
    assert 'action.type === "guided_choice"' in response.text


def test_widget_home_action_starts_a_fresh_chat_session():
    response = TestClient(app).get("/widget/widget.js")

    assert response.status_code == 200
    assert 'homeButton.addEventListener("click", returnToStart)' in response.text
    assert 'localStorage.removeItem("migtorg_chat_session_id")' in response.text
    assert "messages.replaceChildren(...homeMessageNodes)" in response.text
    assert "chatGeneration += 1" in response.text
    assert "setHomeVisible(false)" in response.text
    assert "setHomeVisible(true)" in response.text
