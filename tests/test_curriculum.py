import app as web_app


def test_curriculum_endpoint_returns_goal_plan():
    client = web_app.app.test_client()
    response = client.post(
        "/curriculum/plan",
        json={"user_id": "curriculum_user", "goal": "Learn calculus"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["goal"] == "Learn calculus"
    assert data["steps"]
    assert any("calculus" in step.lower() for step in data["steps"])
