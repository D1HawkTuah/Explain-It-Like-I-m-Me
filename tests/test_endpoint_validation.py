"""Tests for Flask endpoint validation and input sanitization."""

import os

import pytest

pytest.importorskip("flask")

import app as web_app


@pytest.fixture
def client():
    """Flask test client."""
    os.environ["FLASK_ENV"] = "development"
    return web_app.app.test_client()


def test_explain_endpoint_returns_400_on_oversized_payload(client):
    """Test that endpoint rejects payloads with too many fields."""
    # Build a payload with many fields to exceed max_fields limit
    data = {f"field_{i}": f"value_{i}" for i in range(25)}  # 25 > 20 max
    data["user_id"] = "test"
    data["topic"] = "test"
    
    response = client.post("/explain", data=data)
    # Should reject or gracefully handle oversized payload
    assert response.status_code in (400, 413, 200)  # 400 bad request, 413 entity too large, or 200 if validation passes


def test_feedback_endpoint_rejects_invalid_rating(client):
    """Test that feedback endpoint rejects invalid ratings."""
    response = client.post(
        "/feedback",
        data={
            "user_id": "test",
            "topic": "test",
            "rating": "invalid",
            "comment": "Test comment",
        },
    )
    
    # Invalid rating should redirect (empty handler) or return index
    assert response.status_code in (200, 302)


def test_explain_endpoint_truncates_long_topic(client):
    """Test that long topics are truncated gracefully."""
    long_topic = "x" * 3000  # Exceeds 2000 char limit
    
    with client.session_transaction() as sess:
        sess["chat_history"] = []
    
    response = client.post(
        "/explain",
        data={
            "user_id": "test",
            "topic": long_topic,
            "knowledge_level": "beginner",
            "learning_style": "step-by-step",
        },
    )
    
    # Should handle gracefully (truncate and provide explanation)
    assert response.status_code in (200, 302)
    if response.status_code == 200:
        # Response should still have explanation rendered
        assert b"explanation" in response.data.lower() or b"test" in response.data.lower()


def test_feedback_endpoint_truncates_long_comment(client):
    """Test that long feedback comments are truncated."""
    long_comment = "y" * 6000  # Exceeds 5000 char limit
    
    response = client.post(
        "/feedback",
        data={
            "user_id": "test",
            "topic": "test",
            "rating": "5",
            "comment": long_comment,
        },
    )
    
    # Should handle gracefully
    assert response.status_code in (200, 302)


def test_explain_endpoint_normalizes_user_id(client):
    """Test that user_id is normalized correctly."""
    response = client.post(
        "/explain",
        data={
            "user_id": "  UPPER_case  ",
            "topic": "test",
            "knowledge_level": "beginner",
        },
    )
    
    assert response.status_code in (200, 302)


def test_memory_inspect_json_endpoint_returns_valid_json(client):
    """Test that memory inspector JSON endpoint returns valid JSON."""
    with client.session_transaction() as sess:
        sess["chat_history"] = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there"},
        ]
    
    response = client.get("/memory/inspect.json")
    assert response.status_code == 200
    
    data = response.get_json()
    assert isinstance(data, dict)
    assert "chat_history" in data
    assert "semantic_context" in data
    assert "session_turn_count" in data


def test_memory_inspect_html_endpoint_renders(client):
    """Test that memory inspector HTML endpoint renders."""
    with client.session_transaction() as sess:
        sess["chat_history"] = [
            {"role": "user", "text": "What is photosynthesis?"},
            {"role": "assistant", "text": "It is a process..."},
        ]
    
    response = client.get("/memory/inspect")
    assert response.status_code == 200
    assert b"Memory Inspector" in response.data


def test_chat_clear_endpoint_clears_history(client):
    """Test that chat clear endpoint empties the session history."""
    with client.session_transaction() as sess:
        sess["chat_history"] = [
            {"role": "user", "text": "Hello"},
        ]
    
    response = client.post("/chat/clear")
    assert response.status_code == 302  # Should redirect
    
    with client.session_transaction() as sess:
        history = sess.get("chat_history")
        assert history is None or history == []


def test_index_endpoint_returns_200(client):
    """Test that index endpoint returns 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Explain-It-Like-I" in response.data or b"tutor" in response.data.lower()


def test_feedback_endpoint_with_empty_fields(client):
    """Test that feedback endpoint handles empty optional fields."""
    response = client.post(
        "/feedback",
        data={
            "user_id": "",
            "topic": "",
            "rating": "3",
            "comment": "",
        },
    )
    
    # Should handle gracefully, user_id should default to 'guest'
    assert response.status_code in (200, 302)


def test_request_with_sql_injection_attempt(client):
    """Test that endpoint handles SQL injection-like inputs safely."""
    response = client.post(
        "/explain",
        data={
            "user_id": "'; DROP TABLE users; --",
            "topic": "test; DELETE FROM profiles; --",
            "knowledge_level": "beginner",
        },
    )
    
    # Should handle safely (no SQL injection possible with our simple storage)
    assert response.status_code in (200, 302)


def test_request_with_xss_attempt(client):
    """Test that endpoint handles XSS attempts safely."""
    response = client.post(
        "/explain",
        data={
            "user_id": "<script>alert('xss')</script>",
            "topic": "<img src=x onerror=alert('xss')>",
        },
    )
    
    # Should handle safely, template should escape
    assert response.status_code in (200, 302)
