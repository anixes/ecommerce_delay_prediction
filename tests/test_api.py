import pytest
import unittest.mock as mock
import src.api.main as api_main
from fastapi.testclient import TestClient

# 1. SETUP GLOBAL MOCKS PERMANENTLY
MOCK_COLUMNS = [
    'distance_km', 'lead_time_days_estimated', 'total_weight_g', 
    'total_price', 'total_freight', 'seller_avg_review_score',
    'seller_historical_delay_rate', 'seller_state_backlog',
    'required_velocity', 'is_black_friday', 'is_holiday',
    'purchase_month', 'purchase_day_of_week', 'purchase_hour'
]

mock_model = mock.MagicMock()
mock_model.predict_proba.return_value = [[0.8, 0.2]]
mock_model.get_feature_importance.return_value = [[0.1] * (len(MOCK_COLUMNS) + 1)]

# Injected BEFORE TestClient starts
api_main.model = mock_model
api_main.MODEL_COLUMNS = MOCK_COLUMNS

from src.api.main import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True

def test_predict_endpoint():
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
    
    assert response.status_code == 200, f"API failed with {response.text}"
    data = response.json()
    assert "delay_probability" in data
    assert data["delay_probability"] == 0.2
    assert "top_risk_factors" in data

def test_invalid_input():
    invalid_data = {"distance_km": 150.0}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
