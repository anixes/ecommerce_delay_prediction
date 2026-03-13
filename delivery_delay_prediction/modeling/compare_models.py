import pandas as pd
import numpy as np
import typer
import json
from pathlib import Path
from loguru import logger
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_splits: int = 5
):
    logger.info(f"Loading data for comparison from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'is_late'
    X = df.drop(columns=['order_id', target_col])
    y = df[target_col].astype(int)

    # Prepare LightGBM data (category types)
    cat_cols = ['customer_state', 'seller_state', 'product_category', 'primary_payment_type']
    X_lgb = X.copy()
    for col in cat_cols:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype('category')

    # Setup CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []

    # Models to compare (Focusing on LightGBM as requested)
    model_configs = [
        {
            "name": "LightGBM_Baseline",
            "type": "lightgbm",
            "params": {"n_estimators": 500, "learning_rate": 0.05, "class_weight": "balanced", "verbosity": -1, "n_jobs": 1}
        }
    ]

    # Check for tuned lightgbm params
    lgb_tuned_path = MODELS_DIR / "best_lightgbm_params.json"
    if lgb_tuned_path.exists():
        with open(lgb_tuned_path, 'r') as f:
            tuned_params = json.load(f)
            tuned_params["is_unbalance"] = True
            tuned_params["verbosity"] = -1
            tuned_params["n_jobs"] = 1
            model_configs.append({"name": "LightGBM_Tuned", "type": "lightgbm", "params": tuned_params})

    for config in model_configs:
        logger.info(f"Evaluating Model: {config['name']} using {n_splits}-fold CV...")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgb, y)):
            X_train, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**config['params'])
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            fold_scores.append({
                "PR-AUC": average_precision_score(y_val, y_prob),
                "ROC-AUC": roc_auc_score(y_val, y_prob),
                "Accuracy": accuracy_score(y_val, y_pred),
                "F1": f1_score(y_val, y_pred),
                "Recall": recall_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred)
            })
            logger.info(f"  {config['name']} Fold {fold+1} finished.")

        avg_metrics = pd.DataFrame(fold_scores).mean().to_dict()
        avg_metrics["Model"] = config['name']
        results.append(avg_metrics)
        logger.info(f"Summary for {config['name']}: PR-AUC={avg_metrics['PR-AUC']:.4f}, Recall={avg_metrics['Recall']:.4f}")

    # Create Comparison Table
    comparison_df = pd.DataFrame(results).set_index("Model").sort_values("PR-AUC", ascending=False)
    
    # Save results
    report_path = MODELS_DIR / "model_comparison_report.csv"
    comparison_df.to_csv(report_path)
    
    logger.success("\n" + "="*50 + "\nLIGHTGBM COMPARISON RESULTS\n" + "="*50)
    print(comparison_df.to_string())
    logger.success(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    app()
