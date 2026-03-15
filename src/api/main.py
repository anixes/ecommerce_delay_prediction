import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier
from delivery_delay_prediction.config import CATBOOST_TUNED_MODEL, PROCESSED_DATA_DIR, CAT_FEATURES
from delivery_delay_prediction.features import clean_and_prepare_data
import uvicorn
from loguru import logger
import json
import os

app = FastAPI(
    title="E-Commerce Delivery Delay Prediction API",
    description="Predicts if an order will be delayed based on logistics and seller data.",
    version="1.0.0"
)

# Initialize globally
model = None
MODEL_COLUMNS = [
    'distance_km', 'lead_time_days_estimated', 'total_weight_g', 
    'total_price', 'total_freight', 'seller_avg_review_score',
    'seller_historical_delay_rate', 'seller_state_backlog',
    'required_velocity', 'is_black_friday', 'is_holiday',
    'purchase_month', 'purchase_day_of_week', 'purchase_hour'
]

def load_resources():
    global model, MODEL_COLUMNS
    try:
        if CATBOOST_TUNED_MODEL.exists():
            model = CatBoostClassifier().load_model(str(CATBOOST_TUNED_MODEL))
            logger.info("Loaded real CatBoost model.")
            
            feat_csv = PROCESSED_DATA_DIR / "features.csv"
            if feat_csv.exists():
                df_cols = pd.read_csv(feat_csv, nrows=0)
                MODEL_COLUMNS = [c for c in df_cols.columns if c not in ['order_id', 'is_late']]
        else:
            logger.warning("Model file not found. API running in uninitialized state.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("startup")
def startup_event():
    # Only load if not already mocked/loaded
    if model is None:
        load_resources()

class OrderInput(BaseModel):
    distance_km: float = Field(..., example=150.0)
    lead_time_days_estimated: float = Field(..., example=12.0)
    total_weight_g: float = Field(..., example=800.0)
    total_price: float = Field(..., example=150.0)
    total_freight: float = Field(..., example=25.0)
    customer_state: str = Field(..., example="SP")
    seller_state: str = Field(..., example="SP")
    product_category: str = Field(default="UNKNOWN", example="health_beauty")
    primary_payment_type: str = Field(default="credit_card", example="credit_card")
    order_purchase_timestamp: str = Field(..., example="2018-05-15 10:00:00")
    seller_avg_review_score: float = Field(default=4.0, example=4.5)
    seller_historical_delay_rate: float = Field(default=0.05, example=0.02)
    total_items: int = Field(default=1, example=1)
    avg_product_volume_cm3: float = Field(default=1000.0, example=5000.0)
    seller_state_backlog: float = Field(default=1.0)
    seller_intensity_score: float = Field(default=1.0)
    route_delay_rate: float = Field(default=0.02)

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "feature_count": len(MODEL_COLUMNS)
    }

@app.post("/predict")
def predict(order: OrderInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = order.model_dump()
        df = pd.DataFrame([data])
        
        ts = pd.to_datetime(df['order_purchase_timestamp'])
        df['purchase_month'] = ts.dt.month.astype(str)
        df['purchase_day_of_week'] = ts.dt.dayofweek.astype(str)
        df['purchase_hour'] = ts.dt.hour.astype(str)
        df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
        
        for col in MODEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        
        df['order_id'] = "API_REQUEST"
        df_processed = clean_and_prepare_data(df)
        X = df_processed[MODEL_COLUMNS].copy()
        
        for col in CAT_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna("UNKNOWN").astype(str)
                
        probs = model.predict_proba(X)
        if hasattr(probs, 'flatten'):
            # Real model might return a 2D array
            prob = probs[0, 1]
        else:
            # Mock might return a list
            prob = probs[0][1]
        
        from catboost import Pool
        pool = Pool(X, cat_features=[c for c in CAT_FEATURES if c in X.columns])
        shap_values = model.get_feature_importance(data=pool, type='ShapValues')[0]
        feature_shap = dict(zip(MODEL_COLUMNS, shap_values[:-1]))
        
        top_contributors = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        contributors = [
            {
                "feature": f, 
                "impact": round(v, 4),
                "direction": "increasing" if v >= 0 else "decreasing"
            } for f, v in top_contributors
        ]
        
        return {
            "delay_probability": round(float(prob), 4),
            "is_high_risk": bool(prob > 0.5),
            "risk_level": "High" if prob > 0.5 else "Moderate" if prob > 0.3 else "Low",
            "top_risk_factors": contributors
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
