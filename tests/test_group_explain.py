import app as web_app


def test_group_room_can_store_and_return_explanations():
    client = web_app.app.test_client()

    create_response = client.post("/group/create", json={"user_id": "group_user"})
    assert create_response.status_code == 200
    create_data = create_response.get_json()
    assert create_data["room_id"]

    explain_response = client.post(
        f"/group/{create_data['room_id']}/explain",
        json={"user_id": "group_user", "topic": "photosynthesis"},
    )

    assert explain_response.status_code == 200
    data = explain_response.get_json()
    assert data["room_id"] == create_data["room_id"]
    assert data["topic"] == "photosynthesis"
    assert data["explanation"]
    assert len(data["messages"]) >= 1
