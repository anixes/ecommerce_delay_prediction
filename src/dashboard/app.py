import streamlit as st
import pandas as pd
import requests
import datetime
from delivery_delay_prediction.config import CAT_FEATURES

st.set_page_config(
    page_title="Olist Delivery Risk Dashboard",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delivery Delay Prediction Dashboard")
st.markdown("""
Predict the likelihood of a late delivery for Olist orders. 
This dashboard uses a **CatBoost** model trained on historical logistics data.
""")

with st.sidebar:
    st.header("Logistics Configuration")
    
    # Core logistics inputs
    distance = st.number_input("Shipping Distance (km)", min_value=1.0, value=250.0, step=10.0)
    lead_time = st.number_input("Estimated Lead Time (days)", min_value=1.0, value=15.0, step=1.0)
    
    st.subheader("Product Details")
    weight = st.number_input("Weight (g)", min_value=0.0, value=500.0)
    price = st.number_input("Order Value ($)", min_value=0.0, value=89.90)
    freight = st.number_input("Freight Value ($)", min_value=0.0, value=15.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Route Info")
    customer_state = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "GO", "ES"], index=0)
    seller_state = st.selectbox("Seller State", ["SP", "RJ", "MG", "PR", "SC", "RS", "BA", "PE", "GO", "ES"], index=0)
    
    st.subheader("📅 Timing")
    purchase_date = st.date_input("Purchase Date", datetime.date.today())
    purchase_time = st.time_input("Purchase Time", datetime.time(10, 0))

with col2:
    st.subheader("📦 Order Complexity")
    cat_list = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "computers_accessories", "housewares", "cool_stuff", "UNKNOWN"]
    category = st.selectbox("Product Category", cat_list, index=cat_list.index("UNKNOWN"))
    payment = st.selectbox("Payment Type", ["credit_card", "boleto", "voucher", "debit_card"], index=0)
    
    st.subheader("📈 Situational Risk")
    backlog = st.slider("State Backlog Intensity", 0.0, 5.0, 1.0)
    route_risk = st.slider("Historical Route Delay Rate", 0.0, 1.0, 0.05)

# Prepare prediction button
if st.button("Predict Delivery Risk", type="primary"):
    # Construct timestamp string
    purchase_ts = f"{purchase_date} {purchase_time}"
    
    # Input data
    payload = {
        "distance_km": distance,
        "lead_time_days_estimated": lead_time,
        "total_weight_g": weight,
        "total_price": price,
        "total_freight": freight,
        "customer_state": customer_state,
        "seller_state": seller_state,
        "product_category": category,
        "primary_payment_type": payment,
        "seller_state_backlog": backlog,
        "seller_intensity_score": backlog, # Using same for simplicity in UI
        "route_delay_rate": route_risk,
        "order_purchase_timestamp": purchase_ts
    }
    
    # Show loading
    with st.spinner("Analyzing logistics data..."):
        try:
            # We hit the local FastAPI endpoint
            # In a real environment, this URL would be dynamic
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prob = result["delay_probability"]
                risk = result["risk_level"]
                
                # Metrics display
                m1, m2 = st.columns(2)
                m1.metric("Delay Probability", f"{prob*100:.1f}%")
                m2.metric("Risk Level", risk)
                
                # Visual feedback
                if prob > 0.5:
                    st.error("🚨 **High Risk**: This delivery is likely to be delayed due to logistics constraints.")
                elif prob > 0.3:
                    st.warning("⚠️ **Moderate Risk**: Some logistics stress detected. Monitor transit points.")
                else:
                    st.success("✅ **Low Risk**: Smooth delivery expected within the estimated timeframe.")
                    
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Connection Error: Ensure the FastAPI service is running on port 8000. \nError: {str(e)}")

st.divider()
st.info("💡 **Backend Tip**: Run `python src/api/main.py` before using this dashboard.")
