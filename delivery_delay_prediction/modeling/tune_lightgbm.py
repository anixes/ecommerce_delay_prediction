import typer
from pathlib import Path
from loguru import logger
from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT, CAT_FEATURES, LIGHTGBM_TUNED_MODEL

app = typer.Typer()

def objective(trial, X, y, n_splits):
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "is_unbalance": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params, n_jobs=1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)]
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_prob)
        scores.append(score)

    return np.mean(scores)

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_trials: int = 50,
    n_splits: int = 3,
    sample_fraction: float = 1.0
):
    import pandas as pd
    import optuna
    import json
    import lightgbm as lgb
    
    logger.info(f"Loading features for LightGBM tuning from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'is_late'
    X_full = df.drop(columns=['order_id', target_col])
    y_full = df[target_col].astype(int)

    cat_cols = CAT_FEATURES
    for col in cat_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype('category')

    if sample_fraction < 1.0:
        logger.info(f"Sampling {sample_fraction*100:.0f}% of data for faster tuning...")
        X_tune = X_full.sample(frac=sample_fraction, random_state=42)
        y_tune = y_full.loc[X_tune.index]
    else:
        X_tune, y_tune = X_full, y_full

    db_path = PROJ_ROOT / "optuna_lightgbm.db"
    storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_tuning",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, X_tune, y_tune, n_splits), n_trials=n_trials)

    logger.success(f"Best Trial Score (PR-AUC): {study.best_value:.4f}")
    
    # Save best params
    params_path = MODELS_DIR / "best_lightgbm_params.json"
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Best params saved to {params_path}")

    # Train final model
    logger.info("Training final LightGBM model on FULL dataset...")
    final_model = lgb.LGBMClassifier(**study.best_params, is_unbalance=True, random_state=42, n_jobs=1)
    final_model.fit(X_full, y_full)
    
    # Save model artifact
    model_path = LIGHTGBM_TUNED_MODEL
    final_model.booster_.save_model(str(model_path))
    logger.success(f"Tuned LightGBM model saved to {model_path}")

    # Save Feature Importance
    fi = final_model.feature_importances_
    fi_df = pd.DataFrame({'feature': X_full.columns, 'importance': fi}).sort_values('importance', ascending=False)
    fi_df.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)
    logger.info(f"Top 5 LightGBM Features: {fi_df.head(5)['feature'].tolist()}")

if __name__ == "__main__":
    app()
