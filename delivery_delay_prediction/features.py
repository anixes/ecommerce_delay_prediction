from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from delivery_delay_prediction.config import CAT_FEATURES, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw analytical data for CatBoost ingestion.
    """
    logger.info("Starting feature preparation pipeline...")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Temporal peak behavior (Black Friday, holidays)
    if 'order_purchase_timestamp' in df_clean.columns:
        ts = pd.to_datetime(df_clean['order_purchase_timestamp'])
        
        # Black Friday dates for the dataset timeframe
        # 2016-11-25, 2017-11-24
        df_clean['is_black_friday'] = ts.dt.date.astype(str).isin(['2016-11-25', '2017-11-24']).astype(int)
        
        # Major Holidays (simplified for Brazil)
        # Adding a few key ones: New Years, Christmas, etc.
        holidays_2016_2018 = [
            '2016-12-25', '2017-01-01', '2017-04-14', '2017-04-21', '2017-05-01', 
            '2017-09-07', '2017-10-12', '2017-11-02', '2017-11-15', '2017-12-25',
            '2018-01-01', '2018-03-30', '2018-04-21', '2018-05-01', '2018-09-07'
        ]
        df_clean['is_holiday'] = ts.dt.date.astype(str).isin(holidays_2016_2018).astype(int)
        
    # Drop unnecessary IDs or timestamps we don't need for the model
    # Keep order_id for joining predictions back later
    cols_to_drop = [
        'customer_id', 
        'customer_city', 
        'seller_city',
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'max_shipping_limit_date'
    ]
    
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    # Fill any straggling NA values in categoricals
    cat_cols = CAT_FEATURES
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("UNKNOWN").astype(str)
            
    # Continuous Transformations (Log transform skewed distance/logistics)
    log_cols = [
        'distance_km', 'total_weight_g', 'avg_product_volume_cm3',
        'seller_state_backlog', 'customer_state_backlog', 'customer_total_orders',
        'seconds_since_last_seller_order', 'seller_intensity_score',
        'avg_description_length'
    ]
    for col in log_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col].fillna(0))
    
    # NEW: Interaction Features
    # 1. Backlog density relative to distance (long distances + high backlog = high risk)
    df_clean['dist_backlog_ratio'] = df_clean['distance_km'] * df_clean['seller_state_backlog']
    
    # 2. Freight impact per gram (Efficiency)
    df_clean['freight_per_gram'] = df_clean['total_freight'] / (df_clean['total_weight_g'] + 1)
    
    # 3. Estimated daily velocity required
    df_clean['required_velocity'] = df_clean['distance_km'] / (df_clean['lead_time_days_estimated'] + 1)
    
    # 4. NEW: Shipping buffer ratio (how much of lead time is seller prep vs delivery)
    if 'seller_shipping_buffer_days' in df_clean.columns:
        df_clean['seller_shipping_buffer_days'] = df_clean['seller_shipping_buffer_days'].fillna(0)
        df_clean['buffer_to_lead_ratio'] = df_clean['seller_shipping_buffer_days'] / (df_clean['lead_time_days_estimated'] + 1)
    
    # 5. NEW: Route risk score (route delay * distance interaction)
    if 'route_delay_rate' in df_clean.columns:
        df_clean['route_delay_rate'] = df_clean['route_delay_rate'].fillna(0)
        df_clean['route_risk_score'] = df_clean['route_delay_rate'] * df_clean['distance_km']
    
    # 6. NEW: Hub Interaction (Targeting the São Paulo blindspot)
    if 'seller_state' in df_clean.columns and 'customer_state' in df_clean.columns:
        # Hub deliveries (SP to SP) - High volume, complex logistics
        df_clean['is_hub_delivery'] = ((df_clean['seller_state'] == 'SP') & (df_clean['customer_state'] == 'SP')).astype(int)
        # Hub exits (SP to elsewhere) - Terminal transit risk
        df_clean['is_hub_exit'] = ((df_clean['seller_state'] == 'SP') & (df_clean['customer_state'] != 'SP')).astype(int)

    # 7. NEW: Seller Stress Ratio (Seller intensity vs State capacity)
    # If intensity survives log-transform (line 65), we compute the ratio of the original values (or subtract the log values)
    if 'seller_intensity_score' in df_clean.columns and 'seller_state_backlog' in df_clean.columns:
        # Since they are log-transformed, subtraction equals division in log-space
        df_clean['seller_stress_ratio'] = df_clean['seller_intensity_score'] - df_clean['seller_state_backlog']

    # 8. NEW: Holiday Proximity (Days until/since holiday)
    if 'order_purchase_timestamp' in df: # Use original df for timestamp
        ts = pd.to_datetime(df['order_purchase_timestamp'])
        holiday_dates = pd.to_datetime(holidays_2016_2018)
        
        # Calculate days to nearest holiday for each order
        # (This is a bit expensive but very useful for process delays)
        def nearest_holiday_dist(d):
            return min([abs((d - h).days) for h in holiday_dates])
        
        df_clean['days_to_nearest_holiday'] = ts.apply(nearest_holiday_dist)
        # High volume risk: order placed within 7 days of a holiday
        df_clean['is_holiday_season'] = (df_clean['days_to_nearest_holiday'] <= 7).astype(int)

    # 9. NEW: Tight Schedule Risk Interaction
    df_clean['tight_schedule_holiday_risk'] = df_clean['required_velocity'] * df_clean['is_holiday']
        
    # Cap outlier columns at 99.5th percentile
    cap_cols = ['total_price', 'total_freight', 'freight_ratio', 'total_payment', 'dist_backlog_ratio', 'required_velocity']
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
    history_cols = ['seller_historical_delay_rate', 'seller_avg_review_score', 'route_delay_rate']
    for col in history_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(float)
    
    # Fill new numeric columns
    fill_zero_cols = ['avg_product_photos', 'shipping_limit_spread']
    for col in fill_zero_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    logger.info(f"Finished preparation. Output shape: {df_clean.shape}")
    return df_clean

def get_catboost_cat_features(df: pd.DataFrame) -> list:
    """Returns a list of categorical column names expected by CatBoost."""
    return [col for col in CAT_FEATURES if col in df.columns]

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
