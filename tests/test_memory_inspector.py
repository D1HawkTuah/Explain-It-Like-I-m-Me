import pytest

pytest.importorskip("flask")

import app as web_app


def test_memory_inspector_returns_structured_session_payload():
    client = web_app.app.test_client()

    with client.session_transaction() as sess:
        sess["chat_history"] = [
            {"role": "user", "text": "Explain photosynthesis in plain language."},
            {"role": "assistant", "text": "Plants convert sunlight into stored energy."},
        ]

    response = client.get("/memory/inspect.json")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["session_turn_count"] == 2
    assert isinstance(payload["chat_history"], list)
    assert payload["chat_history"][0]["role"] == "user"
    assert "semantic_context" in payload
    assert "summary" in payload["semantic_context"]
    assert payload["semantic_context"]["turn_count"] == 2


def test_memory_inspector_html_page_renders_collapsible_sections():
    client = web_app.app.test_client()

    with client.session_transaction() as sess:
        sess["chat_history"] = [
            {"role": "user", "text": "What is inertia?"},
            {"role": "assistant", "text": "Inertia is resistance to changes in motion."},
        ]

    response = client.get("/memory/inspect")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Memory Inspector" in body
    assert "<details" in body
    assert "Structured Semantic Fields" in body
    assert "roleFilter" in body
    assert "keywordFilter" in body
    assert "turnStart" in body
    assert "turnEnd" in body
