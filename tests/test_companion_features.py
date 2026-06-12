import app as web_app


def test_dashboard_endpoint_returns_private_companion_summary():
    client = web_app.app.test_client()
    response = client.get("/dashboard", query_string={"user_id": "companion_user"})

    assert response.status_code == 200
    data = response.get_json()
    assert "profile" in data
    assert "mastery_summary" in data
    assert "memory_summary" in data
    assert "due_topics" in data


def test_explanations_include_reflection_prompt():
    engine = web_app.engine
    profile = web_app.UserProfile(
        user_id="u8",
        display_name="Reflector",
        knowledge_level="beginner",
        learning_style="step-by-step",
        interests=["science"],
        domains_of_focus=["learning"],
    )

    text = engine.explain(topic="photosynthesis", profile=profile, recent_topics=[])

    assert "reflection prompt" in text.lower()
