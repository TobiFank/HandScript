# tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app

def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "HandScript API is running"}