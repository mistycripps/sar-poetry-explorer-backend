from fastapi.testclient import TestClient
from main import app, detect_crisis_type

client = TestClient(app)


def test_detect_crisis_type_self_harm():
    assert detect_crisis_type("I want to kill myself") == "self"
    assert detect_crisis_type("I feel suicidal") == "self"
    assert detect_crisis_type("sometimes I think about hurting myself") == "self"


def test_detect_crisis_type_harm_others():
    assert detect_crisis_type("I feel like killing my sister") == "others"
    assert detect_crisis_type("I want to hurt someone") == "others"
    assert detect_crisis_type("I want to murder them") == "others"


def test_chat_self_harm_returns_no_poems():
    payload = {"message": "I want to kill myself", "top_k": 3}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["poems"] == []
    assert "self-harm" in data["reply"] or "suicide" in data["reply"]


def test_chat_harm_others_returns_no_poems():
    payload = {"message": "I feel like killing my sister", "top_k": 3}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["poems"] == []
    assert "harming someone else" in data["reply"] or "emergency services" in data["reply"]

