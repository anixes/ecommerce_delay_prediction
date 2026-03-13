import pandas as pd
import numpy as np
import typer
import json
from pathlib import Path
from loguru import logger
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from delivery_delay_prediction.config import (
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    CAT_FEATURES,
    CATBOOST_TUNED_MODEL,
    CATBOOST_BASELINE_MODEL,
    LIGHTGBM_TUNED_MODEL,
    LIGHTGBM_BASELINE_MODEL
)

app = typer.Typer()

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_splits: int = 5
):
    logger.info(f"Loading data for comparison from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'is_late'
    X_raw = df.drop(columns=['order_id', target_col])
    y = df[target_col].astype(int)

    # Setup CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    # Model Configurations to test
    model_configs = []

    # 1. CatBoost Baseline
    model_configs.append({
        "name": "CatBoost_Baseline",
        "type": "catboost",
        "params": {"iterations": 500, "depth": 6, "learning_rate": 0.05, "auto_class_weights": "Balanced", "verbose": False}
    })

    # 2. CatBoost Tuned (Check if exists)
    if (MODELS_DIR / "best_catboost_params.json").exists():
        with open(MODELS_DIR / "best_catboost_params.json", 'r') as f:
            params = json.load(f)
            params["auto_class_weights"] = "Balanced"
            params["verbose"] = False
            model_configs.append({"name": "CatBoost_Tuned", "type": "catboost", "params": params})

    # 3. LightGBM Baseline
    model_configs.append({
        "name": "LightGBM_Baseline",
        "type": "lightgbm",
        "params": {"n_estimators": 500, "learning_rate": 0.05, "class_weight": "balanced", "verbosity": -1, "n_jobs": 1}
    })

    # 4. LightGBM Tuned (Check if exists)
    if (MODELS_DIR / "best_lightgbm_params.json").exists():
        with open(MODELS_DIR / "best_lightgbm_params.json", 'r') as f:
            params = json.load(f)
            params["is_unbalance"] = True
            params["verbosity"] = -1
            params["n_jobs"] = 1
            model_configs.append({"name": "LightGBM_Tuned", "type": "lightgbm", "params": params})

    for config in model_configs:
        logger.info(f"Evaluating: {config['name']}...")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if config["type"] == "catboost":
                # Prep for CatBoost
                X_tr, X_v = X_train_raw.copy(), X_val_raw.copy()
                for col in CAT_FEATURES:
                    X_tr[col] = X_tr[col].fillna("UNKNOWN").astype(str)
                    X_v[col] = X_v[col].fillna("UNKNOWN").astype(str)
                
                model = CatBoostClassifier(**config['params'])
                model.fit(X_tr, y_train, cat_features=CAT_FEATURES)
                y_prob = model.predict_proba(X_v)[:, 1]
                y_pred = model.predict(X_v)
                
            else:
                # Prep for LightGBM
                X_tr, X_v = X_train_raw.copy(), X_val_raw.copy()
                for col in CAT_FEATURES:
                    X_tr[col] = X_tr[col].astype('category')
                    X_v[col] = X_v[col].astype('category')
                
                model = lgb.LGBMClassifier(**config['params'])
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_v)[:, 1]
                y_pred = model.predict(X_v)

            fold_scores.append({
                "PR-AUC": average_precision_score(y_val, y_prob),
                "ROC-AUC": roc_auc_score(y_val, y_prob),
                "F1": f1_score(y_val, y_pred),
                "Recall": recall_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred)
            })

        avg_metrics = pd.DataFrame(fold_scores).mean().to_dict()
        avg_metrics["Model"] = config['name']
        results.append(avg_metrics)
        logger.info(f"  Result: PR-AUC={avg_metrics['PR-AUC']:.4f}")

    comparison_df = pd.DataFrame(results).set_index("Model").sort_values("PR-AUC", ascending=False)
    report_path = MODELS_DIR / "model_comparison_report.csv"
    comparison_df.to_csv(report_path)
    
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON (CV AVERAGE)")
    print("="*60)
    print(comparison_df.to_string())
    logger.success(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    app()
