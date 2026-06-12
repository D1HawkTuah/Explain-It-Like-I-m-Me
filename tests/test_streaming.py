import app as web_app


class FakeStreamLLM:
    enabled = True

    def stream_explain(self, topic, profile, recent_topics, domain_hint, semantic_context=None):
        yield "Hello"
        yield " world"


def test_explain_stream_endpoint_returns_sse_chunks(monkeypatch):
    monkeypatch.setattr(web_app, "llm", FakeStreamLLM())

    client = web_app.app.test_client()
    response = client.post(
        "/explain/stream",
        json={"user_id": "stream_user", "topic": "photosynthesis"},
    )

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    body = response.data.decode("utf-8")
    assert "data: Hello" in body
    assert "data:  world" in body
