import pandas as pd
import numpy as np
import pytest
from delivery_delay_prediction.features import clean_and_prepare_data

def get_base_mock_data():
    return {
        'order_id': ['test_1'],
        'customer_id': ['c1'],
        'customer_state': ['SP'],
        'seller_state': ['SP'],
        'distance_km': [100.0],
        'lead_time_days_estimated': [10.0],
        'total_weight_g': [500.0],
        'total_price': [100.0],
        'total_freight': [15.0],
        'product_category': ['health_beauty'],
        'primary_payment_type': ['credit_card'],
        'order_purchase_timestamp': ['2018-01-01 10:00:00'],
        'seller_state_backlog': [1.0],
        'customer_state_backlog': [1.0],
        'seller_intensity_score': [1.0],
        'route_delay_rate': [0.02],
        'avg_product_volume_cm3': [1000.0],
        'customer_total_orders': [1],
        'seconds_since_last_seller_order': [3600],
        'avg_description_length': [500],
        # Temporal features that main.py adds before calling cleaning
        'purchase_month': ['1'],
        'purchase_day_of_week': ['0'],
        'purchase_hour': ['10']
    }

def test_feature_pipeline_output_shape():
    df = pd.DataFrame(get_base_mock_data())
    
    # Run the pipeline
    df_processed = clean_and_prepare_data(df)
    
    # Check for expected feature presence added BY the pipeline
    expected_new_cols = ['is_holiday', 'is_hub_delivery', 'dist_backlog_ratio', 'required_velocity']
    for col in expected_new_cols:
        assert col in df_processed.columns, f"Missing expected feature: {col}"

def test_temporal_feature_logic():
    # Test Black Friday logic
    base_data = get_base_mock_data()
    base_data['order_purchase_timestamp'] = ['2017-11-24 10:00:00'] # Black Friday 2017
    df = pd.DataFrame(base_data)
    df_processed = clean_and_prepare_data(df)
    
    assert df_processed.loc[0, 'is_black_friday'] == 1
    
    # Test non-black friday
    base_data['order_purchase_timestamp'] = ['2018-05-01 10:00:00']
    df_reg = clean_and_prepare_data(pd.DataFrame(base_data))
    assert df_reg.loc[0, 'is_black_friday'] == 0
