import pandas as pd
import numpy as np
from pathlib import Path
import typer
from loguru import logger
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

from delivery_delay_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR
from delivery_delay_prediction.features import get_catboost_cat_features

app = typer.Typer()

@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "features.csv",
    n_splits: int = 5,
    epochs: int = 500
):
    """
    Trains CatBoost on the prepared features using Stratified K-Fold CV.
    Logs metrics and feature importances to MLflow.
    """
    logger.info(f"Loading features from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Could not find features at {data_path}. Please run features.py first.")
        raise typer.Exit(code=1)

    # Separate target and features
    target_col = 'is_late'
    # Exclude order_id and any remaining target variants
    drop_cols = ['order_id', target_col]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df[target_col].astype(int)
    
    cat_features = get_catboost_cat_features(df)
    logger.info(f"Identified Categorical Features for CatBoost: {cat_features}")

    # Calculate class imbalance
    late_pct = y.mean() * 100
    logger.info(f"Total rows: {len(X)}")
    logger.info(f"Base delay rate: {late_pct:.2f}%")
    
    # ------------------
    # MLflow tracking
    # ------------------
    mlflow.set_experiment("E-Commerce Delivery Delay Baseline")
    with mlflow.start_run() as run:
        mlflow.log_param("model", "CatBoostClassifier")
        mlflow.log_param("cv_splits", n_splits)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("auto_class_weights", "Balanced")
        
        # Setup CV
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_metrics = {
            'f1': [],
            'precision': [],
            'recall': [],
            'pr_auc': []
        }
        
        logger.info(f"Starting {n_splits}-fold Stratified CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fill NaN for categorical inputs explicitly to string "UNKNOWN" on CV slices just to be safe
            for c in cat_features:
                X_train[c] = X_train[c].fillna("UNKNOWN").astype(str)
                X_val[c] = X_val[c].fillna("UNKNOWN").astype(str)
            
            # Create CatBoost Pools
            train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
            val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
            
            model = CatBoostClassifier(
                iterations=epochs,
                learning_rate=0.05,
                depth=6,
                auto_class_weights='Balanced', # Handle 8% late imbalance internally
                eval_metric='F1',
                random_seed=42,
                verbose=100
            )
            
            model.fit(
                train_pool, 
                eval_set=val_pool, 
                early_stopping_rounds=50,
                use_best_model=True
            )
            
            # Predictions
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            f1 = f1_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            pr_auc = average_precision_score(y_val, y_prob)
            
            fold_metrics['f1'].append(f1)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['pr_auc'].append(pr_auc)
            
            logger.info(f"Fold {fold+1} Finished: PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

        # Final average metrics
        mean_f1 = np.mean(fold_metrics['f1'])
        mean_pr_auc = np.mean(fold_metrics['pr_auc'])
        
        logger.success("-" * 40)
        logger.success(f"Final CV Average PR-AUC: {mean_pr_auc:.4f}")
        logger.success(f"Final CV Average F1-Score: {mean_f1:.4f}")
        logger.success("-" * 40)

        # Log average metrics to MLflow
        mlflow.log_metric("cv_mean_f1", mean_f1)
        mlflow.log_metric("cv_mean_pr_auc", mean_pr_auc)
        mlflow.log_metric("cv_mean_precision", np.mean(fold_metrics['precision']))
        mlflow.log_metric("cv_mean_recall", np.mean(fold_metrics['recall']))
        
        # Train final model on FULL dataset for artifact saving
        logger.info("Training final model on full feature set...")
        for c in cat_features:
            X[c] = X[c].fillna("UNKNOWN").astype(str)
        full_pool = Pool(data=X, label=y, cat_features=cat_features)
        
        final_model = CatBoostClassifier(
            iterations=epochs,
            learning_rate=0.05,
            depth=6,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
        final_model.fit(full_pool)
        
        # Save model to MLflow and local registry
        logger.info("Saving model architecture and artifact...")
        mlflow.catboost.log_model(final_model, artifact_path="model")
        
        # Assuming MODELS_DIR is set up
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "catboost_baseline.cbm"
        final_model.save_model(str(model_path))
        logger.success(f"Final model deployed to: {model_path}")

        # Grab and log feature importances
        feature_importances = final_model.get_feature_importance()
        fi_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
        fi_df = fi_df.sort_values('importance', ascending=False)
        fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        logger.info(f"Top 3 Features: {fi_df.head(3)['feature'].tolist()}")

if __name__ == "__main__":
    app()
