import pandas as pd
import numpy as np
import json
from pathlib import Path
from loguru import logger
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR

def evaluate():
    logger.info("Loading features...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    
    target_col = 'is_late'
    X = df.drop(columns=['order_id', target_col])
    y = df[target_col].astype(int)
    
    # Split for final evaluation (20% holdout)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Evaluate CatBoost Baseline
    cb_path = MODELS_DIR / "catboost_baseline.cbm"
    score_cb = 0
    if cb_path.exists():
        cb_model = CatBoostClassifier().load_model(str(cb_path))
        X_test_cb = X_test.copy()
        cat_features = ['customer_state', 'seller_state', 'product_category', 'primary_payment_type']
        for c in cat_features:
            X_test_cb[c] = X_test_cb[c].fillna("UNKNOWN").astype(str)
            
        y_prob_cb = cb_model.predict_proba(X_test_cb)[:, 1]
        score_cb = average_precision_score(y_test, y_prob_cb)
        logger.info(f"CatBoost Baseline PR-AUC: {score_cb:.4f}")

    # 2. Evaluate LightGBM Tuned
    # Since loading Booster directly from file is tricky with categoricals, 
    # we'll use the best params to reinstantiate a model and fit it quickly on 80% 
    # (since we are on CPU and it's fast now) to get a clean evaluation.
    lgb_params_path = MODELS_DIR / "best_lightgbm_params.json"
    score_lgb = 0
    if lgb_params_path.exists():
        with open(lgb_params_path, 'r') as f:
            params = json.load(f)
        
        # Prepare data for fresh fit (80/20)
        X_train_lgb = X_train.copy()
        X_test_lgb = X_test.copy()
        cat_cols = ['customer_state', 'seller_state', 'product_category', 'primary_payment_type']
        for col in cat_cols:
            X_train_lgb[col] = X_train_lgb[col].astype('category')
            X_test_lgb[col] = X_test_lgb[col].astype('category')
        
        logger.info("Fitting Tuned LightGBM for holdout evaluation...")
        model_lgb = lgb.LGBMClassifier(**params, is_unbalance=True, random_state=42, n_jobs=1)
        model_lgb.fit(X_train_lgb, y_train)
        
        y_prob_lgb = model_lgb.predict_proba(X_test_lgb)[:, 1]
        score_lgb = average_precision_score(y_test, y_prob_lgb)
        logger.info(f"LightGBM Tuned PR-AUC: {score_lgb:.4f}")

    print("\n" + "="*40)
    print("FINAL COMPARISON ON HOLDOUT SET")
    print("="*40)
    print(f"CatBoost (Baseline): {score_cb:.4f}")
    print(f"LightGBM (Tuned):    {score_lgb:.4f}")
    
    if score_lgb > score_cb:
        diff = (score_lgb - score_cb) / score_cb * 100
        print(f"\nWinner: LightGBM (Tuned) by {diff:.2f}% improvement")
    else:
        diff = (score_cb - score_lgb) / score_lgb * 100
        print(f"\nWinner: CatBoost by {diff:.2f}% improvement")
    print("="*40)

if __name__ == "__main__":
    evaluate()
