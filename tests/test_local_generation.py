import app as web_app


def test_local_generation_endpoint_uses_local_backend():
    client = web_app.app.test_client()

    response = client.post(
        "/local/generate",
        json={"user_id": "local_user", "topic": "fractions"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["topic"] == "fractions"
    assert data["source"] == "local"
    assert "fractions" in data["explanation"]
    assert data["domain"]
