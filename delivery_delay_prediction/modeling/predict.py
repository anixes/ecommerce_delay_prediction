import pandas as pd
from pathlib import Path
from loguru import logger
import typer
from catboost import CatBoostClassifier
from delivery_delay_prediction.config import (
    MODELS_DIR, 
    PROCESSED_DATA_DIR, 
    CAT_FEATURES, 
    CATBOOST_TUNED_MODEL, 
    CATBOOST_BASELINE_MODEL
)

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """
    Inference script using the best identified model (CatBoost).
    """
    logger.info(f"Loading features for inference from {features_path}...")
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        raise typer.Exit(code=1)
        
    df = pd.read_csv(features_path)
    
    # Separate order_id for joining later
    order_ids = df['order_id']
    target_col = 'is_late'
    # Features (must match training features)
    X = df.drop(columns=['order_id', target_col], errors='ignore')
    
    # Select model (Prefer tuned over baseline)
    model_path = CATBOOST_TUNED_MODEL if CATBOOST_TUNED_MODEL.exists() else CATBOOST_BASELINE_MODEL
    
    logger.info(f"Loading CatBoost model from {model_path}...")
    if not model_path.exists():
        logger.error(f"No model found at {model_path}. Please run training or cloud tuning first.")
        raise typer.Exit(code=1)
        
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    
    # Handle categorical features (CatBoost needs them as strings or correctly typed)
    # The saved .cbm model remembers which indices were categorical, 
    # but we must ensure data types match.
    # Handle categorical features
    cat_cols = CAT_FEATURES
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN").astype(str)

    logger.info("Starting batch prediction...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    results = pd.DataFrame({
        'order_id': order_ids,
        'late_probability': probs,
        'is_late_prediction': preds
    })
    
    results.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")
    
    # stats
    logger.info(f"Predicted delay rate: {preds.mean()*100:.2f}%")

if __name__ == "__main__":
    app()
