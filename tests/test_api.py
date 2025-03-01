# test_api.py
import pytest
import requests

def test_api():
    response = requests.post("http://127.0.0.1:5000/predict", json={"features": [80, 120, 37.5, 95]})
    assert response.status_code == 200
    assert "sepsis_risk" in response.json()