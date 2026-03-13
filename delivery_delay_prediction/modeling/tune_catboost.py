import pandas as pd
import numpy as np
import optuna
import typer
from pathlib import Path
from loguru import logger
import mlflow
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score

from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT, CAT_FEATURES, CATBOOST_TUNED_MODEL

app = typer.Typer()

def objective(trial, X, y, cat_features, n_splits):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "od_type": "Iter",
        "od_wait": 50,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0"
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Explicitly handle categoricals on slices
        for c in cat_features:
            X_train[c] = X_train[c].fillna("UNKNOWN").astype(str)
            X_val[c] = X_val[c].fillna("UNKNOWN").astype(str)

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        y_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_prob)
        scores.append(score)

    return np.mean(scores)

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_trials: int = 50,
    n_splits: int = 3,
    sample_fraction: float = 1.0  # Default to 100% since we use GPU
):
    logger.info(f"Loading features for tuning from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'is_late'
    X_full = df.drop(columns=['order_id', target_col])
    y_full = df[target_col].astype(int)
    cat_features = [c for c in CAT_FEATURES if c in df.columns]

    # Fast Sampling for Optuna
    if sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction*100:.0f}% of data for faster tuning...")
        df_sample = df.sample(frac=sample_fraction, random_state=42)
        X_tune = df_sample.drop(columns=['order_id', target_col])
        y_tune = df_sample[target_col].astype(int)
    else:
        X_tune, y_tune = X_full, y_full

    mlflow.set_experiment("E-Commerce Delay Hyperopt")
    
    db_path = PROJ_ROOT / "optuna_catboost.db"
    storage = f"sqlite:///{db_path}"
    
    with mlflow.start_run(run_name="CatBoost_Optuna"):
        study = optuna.create_study(
            direction="maximize", 
            study_name="catboost_tuning", 
            storage=storage, 
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, X_tune, y_tune, cat_features, n_splits), n_trials=n_trials)

        logger.success(f"Best Trial Score (PR-AUC): {study.best_value:.4f}")
        logger.info(f"Best Params: {study.best_params}")

        # Log best params and score
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_pr_auc", study.best_value)

        # Save best params
        params_path = MODELS_DIR / "best_catboost_params.json"
        import json
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        logger.info(f"Best params saved to {params_path}")

        # Train final model with best params on FULL data
        logger.info("Training final CatBoost model on FULL dataset...")
        X_final = X_full.copy()
        for c in cat_features:
            X_final[c] = X_final[c].fillna("UNKNOWN").astype(str)
        
        final_model = CatBoostClassifier(**study.best_params, auto_class_weights="Balanced", random_seed=42, verbose=False, task_type="GPU", devices="0")
        final_model.fit(X_final, y_full, cat_features=cat_features)
        
        # Save model
        model_path = CATBOOST_TUNED_MODEL
        final_model.save_model(str(model_path))
        logger.success(f"Tuned CatBoost model saved to {model_path}")

        # Save Feature Importance
        fi = final_model.get_feature_importance()
        fi_df = pd.DataFrame({'feature': X_full.columns, 'importance': fi}).sort_values('importance', ascending=False)
        fi_df.to_csv(MODELS_DIR / "catboost_feature_importance.csv", index=False)
        logger.info(f"Top 5 CatBoost Features: {fi_df.head(5)['feature'].tolist()}")

if __name__ == "__main__":
    app()
