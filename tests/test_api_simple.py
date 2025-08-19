"""
Basic API tests for the Billboard backend
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from src.app_simple import app

client = TestClient(app)

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "mode" in data
    assert data["mode"] == "simple"

def test_health():
    """Test the health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_create_report():
    """Test creating a report"""
    report_data = {
        "storage_key": "test_image.jpg",
        "lat": 23.1765,
        "lon": 79.9006
    }
    response = client.post("/reports", json=report_data)
    assert response.status_code == 202
    data = response.json()
    assert "report_id" in data

def test_get_reports():
    """Test getting all reports"""
    response = client.get("/reports")
    assert response.status_code == 200
    data = response.json()
    assert "reports" in data
    assert "count" in data
