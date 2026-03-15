from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from delivery_delay_prediction.config import CATBOOST_TUNED_MODEL, PROCESSED_DATA_DIR
from delivery_delay_prediction.features import clean_and_prepare_data
import uvicorn
from loguru import logger
import json

app = FastAPI(
    title="E-Commerce Delivery Delay Prediction API",
    description="Predicts if an order will be delayed based on logistics and seller data.",
    version="1.0.0"
)

# Load model globally
MODEL_PATH = CATBOOST_TUNED_MODEL
model = None
MODEL_COLUMNS = [] # We'll populate this on startup

@app.on_event("startup")
def startup_event():
    global model, MODEL_COLUMNS
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
        else:
            model = CatBoostClassifier().load_model(str(MODEL_PATH))
            logger.info(f"Loaded CatBoost model from {MODEL_PATH}")
            
            # Get the exact columns from the processed features file
            feat_csv = PROCESSED_DATA_DIR / "features.csv"
            if feat_csv.exists():
                df_cols = pd.read_csv(feat_csv, nrows=0)
                # Drop target and ID
                MODEL_COLUMNS = [c for c in df_cols.columns if c not in ['order_id', 'is_late']]
                logger.info(f"Loaded {len(MODEL_COLUMNS)} expected features for the model.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

class OrderInput(BaseModel):
    # Core inputs
    distance_km: float = Field(..., example=150.0)
    lead_time_days_estimated: float = Field(..., example=12.0)
    total_weight_g: float = Field(..., example=800.0)
    total_price: float = Field(..., example=150.0)
    total_freight: float = Field(..., example=25.0)
    
    # Categoricals
    customer_state: str = Field(..., example="SP")
    seller_state: str = Field(..., example="SP")
    product_category: str = Field(default="UNKNOWN", example="health_beauty")
    primary_payment_type: str = Field(default="credit_card", example="credit_card")
    
    # Temporal
    order_purchase_timestamp: str = Field(..., example="2018-05-15 10:00:00")
    
    # Enriched Seller & Product Signals
    seller_avg_review_score: float = Field(default=4.0, example=4.5)
    seller_historical_delay_rate: float = Field(default=0.05, example=0.02)
    total_items: int = Field(default=1, example=1)
    avg_product_volume_cm3: float = Field(default=1000.0, example=5000.0)
    
    # Optional/Situational (Defaults provided for missing data)
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
        # 1. Convert input to DataFrame
        data = order.model_dump()
        df = pd.DataFrame([data])
        
        # 2. Add temporal features that CatBoost expects (if missing from raw input)
        ts = pd.to_datetime(df['order_purchase_timestamp'])
        df['purchase_month'] = ts.dt.month.astype(str)
        df['purchase_day_of_week'] = ts.dt.dayofweek.astype(str)
        df['purchase_hour'] = ts.dt.hour.astype(str)
        df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
        
        # 3. Handle default values for columns present in training but not in OrderInput
        # We'll fill with 0 or sensible defaults
        for col in MODEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 # Standard default for missing numeric signals
        
        # 4. Run the project's preprocessing pipeline
        df['order_id'] = "API_REQUEST" # Needed by pipeline
        df_processed = clean_and_prepare_data(df)
        
        # 5. Extract features in the EXACT ORDER the model was trained on
        X = df_processed[MODEL_COLUMNS].copy()
        
        # 6. CRITICAL: Final check for categorical types
        # Some categoricals might be represented as objects/strings, ensure they are strings
        from delivery_delay_prediction.config import CAT_FEATURES
        for col in CAT_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna("UNKNOWN").astype(str)
                
        # 7. Get prediction and SHAP values
        prob = model.predict_proba(X)[0, 1]
        
        # Calculate SHAP values for this specific prediction
        # CatBoost requires a Pool with cat_features defined for models with categoricals
        from catboost import Pool
        pool = Pool(X, cat_features=CAT_FEATURES)
        shap_values = model.get_feature_importance(data=pool, type='ShapValues')[0]
        
        # shap_values has len(MODEL_COLUMNS) + 1 (the last one is the base value)
        feature_shap = dict(zip(MODEL_COLUMNS, shap_values[:-1]))
        
        # Get Top 3 most impactful contributors (absolute value)
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
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
