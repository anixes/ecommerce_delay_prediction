import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import datetime

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_loaded" in data

def test_predict_endpoint():
    # Sample data that matches the OrderInput schema
    sample_data = {
        "distance_km": 150.0,
        "lead_time_days_estimated": 10.0,
        "total_weight_g": 350.0,
        "total_price": 49.90,
        "total_freight": 12.00,
        "customer_state": "SP",
        "seller_state": "SP",
        "product_category": "health_beauty",
        "primary_payment_type": "credit_card",
        "order_purchase_timestamp": "2018-06-01 12:00:00",
        "seller_avg_review_score": 4.5,
        "seller_historical_delay_rate": 0.05,
        "total_items": 1,
        "avg_product_volume_cm3": 1000.0,
        "seller_state_backlog": 1.0,
        "seller_intensity_score": 1.0,
        "route_delay_rate": 0.02
    }
    
    response = client.post("/predict", json=sample_data)
    
    # If model is loaded, we expect 200. 
    # In CI, if we don't have the model file, we might get 503 or fail.
    # For now, let's assume we want to test the routing and schema validation.
    if response.status_code == 200:
        data = response.json()
        assert "delay_probability" in data
        assert "risk_level" in data
        assert 0 <= data["delay_probability"] <= 1
    elif response.status_code == 503:
        # Expected if model file is missing in CI environment
        assert "Model not loaded" in response.json()["detail"]
    else:
        # Should not get other errors if schema is correct
        assert response.status_code == 200, f"API failed with {response.text}"

def test_invalid_input():
    # Missing required field
    invalid_data = {
        "distance_km": 150.0
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422 # Unprocessable Entity
