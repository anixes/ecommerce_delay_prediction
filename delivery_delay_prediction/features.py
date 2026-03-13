import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import typer
from delivery_delay_prediction.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw analytical data for CatBoost ingestion.
    """
    logger.info("Starting feature preparation pipeline...")
    
    # Drop unnecessary IDs or timestamps we don't need for the model
    # Keep order_id for joining predictions back later
    cols_to_drop = [
        'customer_id', 
        'customer_city', 
        'seller_city',
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    
    # Fill any straggling NA values in categoricals
    cat_cols = ['customer_state', 'seller_state', 'product_category', 'primary_payment_type']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("UNKNOWN").astype(str)
            
    # Continuous Transformations (Log transform skewed distance/logistics)
    log_cols = ['distance_km', 'total_weight_g', 'avg_product_volume_cm3']
    for col in log_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col].fillna(0))
        
    # Cap outlier columns at 99.5th percentile
    cap_cols = ['total_price', 'total_freight', 'freight_ratio', 'total_payment']
    for col in cap_cols:
        if col in df_clean.columns:
            # Replace Inf/NaN with 0 before capping
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            p99 = df_clean[col].quantile(0.995)
            df_clean[col] = np.where(df_clean[col] > p99, p99, df_clean[col])
            
    # Ensure our target is int
    if 'is_late' in df_clean.columns:
        df_clean['is_late'] = df_clean['is_late'].astype(int)
        
    # History Features
    history_cols = ['seller_historical_delay_rate', 'seller_avg_review_score']
    for col in history_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(float)
    
    logger.info(f"Finished preparation. Output shape: {df_clean.shape}")
    return df_clean

def get_catboost_cat_features(df: pd.DataFrame) -> list:
    """Returns a list of categorical column names expected by CatBoost."""
    cat_cols = [
        'customer_state', 'seller_state', 'product_category', 
        'primary_payment_type', 'purchase_month', 'purchase_day_of_week', 'purchase_hour'
    ]
    return [col for col in cat_cols if col in df.columns]

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "analytical_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info(f"Generating features from dataset: {input_path}")
    df = pd.read_csv(input_path)
    df_prepared = clean_and_prepare_data(df)
    
    df_prepared.to_csv(output_path, index=False)
    logger.success(f"Features saved to {output_path}")

if __name__ == "__main__":
    app()
