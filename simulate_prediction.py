import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from delivery_delay_prediction.config import CATBOOST_TUNED_MODEL, PROCESSED_DATA_DIR, CAT_FEATURES
from delivery_delay_prediction.features import clean_and_prepare_data

# Load model
model = CatBoostClassifier().load_model(str(CATBOOST_TUNED_MODEL))

# Get expected columns
feat_csv = PROCESSED_DATA_DIR / "features.csv"
df_cols = pd.read_csv(feat_csv, nrows=0)
MODEL_COLUMNS = [c for c in df_cols.columns if c not in ['order_id', 'is_late']]

# Sample payload
data = {
    "distance_km": 150.0,
    "lead_time_days_estimated": 12.0,
    "total_weight_g": 500.0,
    "total_price": 89.0,
    "total_freight": 20.0,
    "customer_state": "SP",
    "seller_state": "SP",
    "product_category": "UNKNOWN",
    "primary_payment_type": "credit_card",
    "order_purchase_timestamp": "2026-03-16 10:00:00",
    "seller_avg_review_score": 4.5,
    "seller_historical_delay_rate": 0.05,
    "total_items": 1,
    "avg_product_volume_cm3": 1000.0,
    "seller_state_backlog": 1.0,
    "seller_intensity_score": 1.0,
    "route_delay_rate": 0.02
}

print("1. Converting to DF...")
df = pd.DataFrame([data])

print("2. Adding temporal features...")
ts = pd.to_datetime(df['order_purchase_timestamp'])
df['purchase_month'] = ts.dt.month.astype(str)
df['purchase_day_of_week'] = ts.dt.dayofweek.astype(str)
df['purchase_hour'] = ts.dt.hour.astype(str)
df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)

# Fill missing
for col in MODEL_COLUMNS:
    if col not in df.columns:
        df[col] = 0.0

print("3. Preprocessing...")
df['order_id'] = "DEBUG"
df_processed = clean_and_prepare_data(df)

print(f"Processed columns: {list(df_processed.columns)}")

print("4. Column alignment...")
try:
    X = df_processed[MODEL_COLUMNS].copy()
except KeyError as e:
    print(f"KeyError during alignment: {e}")
    # Show missing
    missing = [c for c in MODEL_COLUMNS if c not in df_processed.columns]
    print(f"Missing columns: {missing}")
    exit(1)

print("5. Categorical check...")
for col in CAT_FEATURES:
    if col in X.columns:
        X[col] = X[col].fillna("UNKNOWN").astype(str)

print("6. Prediction...")
prob = model.predict_proba(X)[0, 1]
print(f"Probability: {prob}")

print("7. SHAP values...")
from catboost import Pool
# Try without Pool first to see if it fails
try:
    shap_values = model.get_feature_importance(data=X, type='ShapValues')[0]
    print("SHAP successful without Pool")
except Exception as e:
    print(f"SHAP failed without Pool: {e}")
    print("Trying with Pool...")
    pool = Pool(X, cat_features=CAT_FEATURES)
    shap_values = model.get_feature_importance(data=pool, type='ShapValues')[0]
    print("SHAP successful with Pool")
