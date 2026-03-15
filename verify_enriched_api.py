import requests
import json
import time
import subprocess
import os
import sys

def test_enriched_api():
    print("Restarting FastAPI server with enriched schema...")
    # Use sys.executable to ensure we use the same python interpreter
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=os.getcwd()
    )
    
    # Wait a few seconds for it to start
    time.sleep(8)
    
    try:
        # Test predict endpoint with enriched sample data
        print("Testing predict endpoint with enriched data...")
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
            # Enriched fields
            "seller_avg_review_score": 4.8,
            "seller_historical_delay_rate": 0.01,
            "total_items": 1,
            "avg_product_volume_cm3": 800.0
        }
        
        try:
            r_pred = requests.post("http://127.0.0.1:8000/predict", json=sample_data, timeout=10)
            print(f"Predict status: {r_pred.status_code}")
            if r_pred.status_code == 200:
                print(f"Prediction result: {json.dumps(r_pred.json(), indent=2)}")
            else:
                print(f"Error: {r_pred.text}")
        except Exception as e:
            print(f"Failed to connect to predict: {e}")
            
    finally:
        print("Stopping FastAPI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    test_enriched_api()
