import typer
from pathlib import Path
from loguru import logger

from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, CAT_FEATURES, LIGHTGBM_BASELINE_MODEL

app = typer.Typer()

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_splits: int = 5
):
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

    logger.info(f"Loading features for LightGBM from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape}")
    
    target_col = 'is_late'
    logger.info("Separating target and features...")
    X = df.drop(columns=['order_id', target_col])
    y = df[target_col].astype(int)
    logger.info("Target and features separated.")

    # LightGBM requires categorical features to be 'category' type
    cat_cols = CAT_FEATURES
    logger.info(f"Converting {len(cat_cols)} columns to category...")
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    logger.info("Categorical conversion done.")

    logger.info("Initializing StratifiedKFold...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'pr_auc': [], 'f1': [], 'precision': [], 'recall': []}
    logger.info(f"Starting {n_splits}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=50,  # Reduced for testing
            learning_rate=0.05,
            class_weight='balanced',
            importance_type='gain',
            random_state=42,
            verbosity=-1,
            n_jobs=1  # Forced single thread to avoid OpenMP hangs
            # Removed device='gpu' for local reliability
        )

        logger.info(f"Fold {fold+1}: Starting fit...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)]
        )
        logger.info(f"Fold {fold+1}: Fit completed.")

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        fold_metrics['pr_auc'].append(average_precision_score(y_val, y_prob))
        fold_metrics['f1'].append(f1_score(y_val, y_pred))
        fold_metrics['precision'].append(precision_score(y_val, y_pred))
        fold_metrics['recall'].append(recall_score(y_val, y_pred))
        
        logger.info(f"Fold {fold+1} Finished: PR-AUC={fold_metrics['pr_auc'][-1]:.4f}")

    # Final average metrics
    for m, values in fold_metrics.items():
        logger.info(f"Avg {m}: {np.mean(values):.4f}")
    
    logger.success(f"LightGBM CV Average PR-AUC: {np.mean(fold_metrics['pr_auc']):.4f}")

    # Train final model
    logger.info("Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbosity=-1,
        n_jobs=1
    )
    final_model.fit(X, y)
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = LIGHTGBM_BASELINE_MODEL
    final_model.booster_.save_model(str(model_path))
    logger.success(f"LightGBM model saved to {model_path}")

if __name__ == "__main__":
    app()
