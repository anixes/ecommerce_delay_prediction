import streamlit as st
import pandas as pd
import requests
import datetime
import time
import os
from delivery_delay_prediction.config import CAT_FEATURES

# Page configuration
st.set_page_config(
    page_title="Delivery Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    .st-emotion-cache-12w0qpk {
        background-color: #1c2128;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-online { background-color: #238636; box-shadow: 0 0 10px #238636; }
    .status-offline { background-color: #da3633; box-shadow: 0 0 10px #da3633; }
    
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: None;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.4);
    }
    
    /* Mobile Responsiveness Tweaks */
    @media (max-width: 640px) {
        .st-emotion-cache-12w0qpk {
            padding: 12px;
        }
        h1 {
            font-size: 1.8rem !important;
        }
        .stMetric {
            background-color: #1c2128;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 12px;
            border: 1px solid #30363d;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        /* Force input columns to stack on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Default Advanced State Constants
ADV_DEFAULTS = {
    'seller_state_backlog': 1.0,
    'seller_avg_review_score': 4.5,
    'seller_historical_delay_rate': 0.05,
    'total_weight_g': 500.0,
    'total_price': 89.0,
    'product_category': "UNKNOWN"
}

# Human-readable labels for SHAP factors
FEATURE_LABELS = {
    "buffer_to_lead_ratio": "Delivery Time Buffer",
    "total_payment": "Order Commercial Value",
    "route_delay_rate": "Historical Route Reliability",
    "required_velocity": "Required Transit Speed",
    "distance_km": "Shipping Distance",
    "seller_state_backlog": "Regional Hub Congestion",
    "seller_avg_review_score": "Seller Performance Rating",
    "seller_historical_delay_rate": "Seller Reliability Index",
    "days_to_nearest_holiday": "Holiday Proximity",
    "purchase_month": "Seasonal Peak Factor",
    "purchase_day_of_week": "Weekly Transit Load",
    "purchase_hour": "Order Processing Window",
    "product_category": "Product Category Risk",
    "total_weight_g": "Package Physical Profile",
    "is_holiday": "Holiday Status",
    "is_black_friday": "Black Friday Peak",
    "is_hub_delivery": "Hub-specific Logistics",
    "required_velocity": "Transit Urgency"
}

def format_risk_description(feature, direction):
    label = FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())
    if direction == "increasing":
        return f":red[{label}]: This is contributing to a higher risk of delay."
    elif "buffer" in feature or "velocity" in feature:
        return f":blue[{label}]: This factor is in an optimal state for on-time delivery."
    else:
        return f":green[{label}]: This is significantly helping to keep the delivery on time."

# State Management Initialization
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['distance_km'] = 150.0
    st.session_state['lead_time_days_estimated'] = 12.0
    st.session_state['total_weight_g'] = 500.0
    st.session_state['total_price'] = 89.0
    st.session_state['customer_state'] = "SP"
    st.session_state['seller_state'] = "SP"
    st.session_state['seller_state_backlog'] = 1.0
    st.session_state['seller_avg_review_score'] = 4.5
    st.session_state['seller_historical_delay_rate'] = 0.05
    st.session_state['product_category'] = "UNKNOWN"
    st.session_state['purchase_date'] = datetime.date(2018, 5, 1) # Default to a valid trained date
    st.session_state['purchase_time'] = datetime.time(10, 0)

def set_preset(name):
    if name == "Standard Local":
        st.session_state['distance_km'] = 50.0
        st.session_state['lead_time_days_estimated'] = 5.0
        st.session_state['seller_state_backlog'] = 0.5
        st.session_state['customer_state'] = "SP"
        st.session_state['seller_state'] = "SP"
    elif name == "Long Distance":
        st.session_state['distance_km'] = 2500.0
        st.session_state['lead_time_days_estimated'] = 20.0
        st.session_state['customer_state'] = "RN"
        st.session_state['seller_state'] = "SP"
    elif name == "Holiday Rush":
        st.session_state['seller_state_backlog'] = 4.5
        st.session_state['seller_historical_delay_rate'] = 0.15

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Helper for API check
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=1)
        return r.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.title("Optimizer")
    
    online = check_api()
    if online:
        st.markdown('<p><span class="status-dot status-online"></span>API Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p><span class="status-dot status-offline"></span>API Disconnected</p>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Scenario Presets")
    if st.button("Reset to Default"):
        st.session_state['initialized'] = False
        st.rerun()
    if st.button("Standard Local"): set_preset("Standard Local"); st.rerun()
    if st.button("Long Distance"): set_preset("Long Distance"); st.rerun()
    if st.button("Holiday Rush"): set_preset("Holiday Rush"); st.rerun()

# Main UI
st.title("Delivery Risk Optimizer")
st.markdown("""
### Brazilian E-Commerce Logistics (Olist)
In a marketplace with **98k+ sellers** across Brazil, **delivery delays** are the primary driver of customer churn. 

This engine analyzes real-time signals to predict the risk of a late delivery before the package leaves the warehouse. Use this to identify high-risk orders and take proactive measures.
""")

st.divider()

# Core Inputs
st.markdown("### Customer and Delivery Context")
c1, c2 = st.columns(2)
with c1:
    st.session_state['distance_km'] = st.number_input("Shipping Distance (km)", 0.0, 5000.0, value=st.session_state['distance_km'], help="Physical distance between seller hub and customer location.")
    st.session_state['lead_time_days_estimated'] = st.number_input("Estimated Lead Time (days)", 1.0, 60.0, value=st.session_state['lead_time_days_estimated'], help="Timeframe promised to the customer.")
    st.session_state['customer_state'] = st.selectbox("Customer Territory (State)", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "GO", "ES", "RN"], index=["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "GO", "ES", "RN"].index(st.session_state['customer_state']))

with c2:
    st.session_state['seller_state'] = st.selectbox("Seller Territory (State)", ["SP", "RJ", "MG", "PR", "SC", "RS", "BA", "PE", "GO", "ES"], index=["SP", "RJ", "MG", "PR", "SC", "RS", "BA", "PE", "GO", "ES"].index(st.session_state['seller_state']))
    # Restrict to full years of Olist dataset for consistency
    st.session_state['purchase_date'] = st.date_input(
        "Order Date", 
        value=st.session_state['purchase_date'],
        min_value=datetime.date(2016, 1, 1),
        max_value=datetime.date(2018, 12, 31)
    )
    st.session_state['purchase_time'] = st.time_input("Order Time", value=st.session_state['purchase_time'])

# Advanced Inputs
with st.expander("Seller Performance and Product Risk Factors", expanded=False):
    st.markdown("*These signals capture the operational health of the seller at the moment of the order.*")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### Operational Health")
        st.session_state['seller_avg_review_score'] = st.slider("Seller Quality Score (1-5)", 1.0, 5.0, value=st.session_state['seller_avg_review_score'])
        st.session_state['seller_historical_delay_rate'] = st.slider("Historical Delay Rate", 0.0, 1.0, value=st.session_state['seller_historical_delay_rate'])
        st.session_state['seller_state_backlog'] = st.slider("Regional Hub Congestion Index", 0.0, 5.0, value=st.session_state['seller_state_backlog'])
    with a2:
        st.markdown("#### Package Profile")
        st.session_state['total_weight_g'] = st.number_input("Total Package Weight (g)", 0.0, 50000.0, value=st.session_state['total_weight_g'])
        st.session_state['total_price'] = st.number_input("Order Commercial Value ($)", 0.0, 10000.0, value=st.session_state['total_price'])
        categories = ["health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure", "UNKNOWN"]
        st.session_state['product_category'] = st.selectbox("Product Line", categories, index=categories.index(st.session_state['product_category']))

# Check if advanced features are tweaked
is_advanced_tweaked = any(
    st.session_state[k] != ADV_DEFAULTS[k] for k in ADV_DEFAULTS
)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button Label
button_label = "Generate Prediction"
if is_advanced_tweaked:
    button_label = "Generate Integrated Prediction"

if st.button(button_label, type="primary", use_container_width=True):
    if not online:
        st.error("Operation Failed: Connection to Prediction Engine is currently unavailable.")
    else:
        payload = {
            "distance_km": st.session_state['distance_km'],
            "lead_time_days_estimated": st.session_state['lead_time_days_estimated'],
            "total_weight_g": st.session_state['total_weight_g'],
            "total_price": st.session_state['total_price'],
            "total_freight": 20.0,
            "customer_state": st.session_state['customer_state'],
            "seller_state": st.session_state['seller_state'],
            "product_category": st.session_state['product_category'],
            "primary_payment_type": "credit_card",
            "order_purchase_timestamp": f"{st.session_state['purchase_date']} {st.session_state['purchase_time']}",
            "seller_avg_review_score": st.session_state['seller_avg_review_score'],
            "seller_historical_delay_rate": st.session_state['seller_historical_delay_rate'],
            "total_items": 1,
            "avg_product_volume_cm3": 1000.0,
            "seller_state_backlog": st.session_state['seller_state_backlog'],
            "seller_intensity_score": st.session_state['seller_state_backlog']
        }
        
        with st.spinner("Analyzing logistics signals..."):
            try:
                # API Call with error handling
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                
                if response.status_code == 200:
                    res = response.json()
                    prob = res.get("delay_probability")
                    level = res.get("risk_level", "Unknown")
                    factors = res.get("top_risk_factors", [])
                    
                    if prob is None:
                        st.error("API Response Error: Prediction result is missing from server response.")
                    else:
                        st.divider()
                        r1, r2 = st.columns(2)
                        prob_label = "Risk Probability"
                        if is_advanced_tweaked:
                            prob_label = "Combined Risk Probability"
                        
                        r1.metric(prob_label, f"{prob*100:.2f}%")
                        r2.metric("Assessment", level)
                        
                        # Risk Breakdown
                        if factors:
                            st.markdown("### Why this risk level?")
                            st.info("Factors explaining the assessment based on historical logistics data.")
                            for f in factors:
                                st.markdown(format_risk_description(f['feature'], f['direction']))
                        
                        if prob > 0.5:
                            st.error(f"High Risk Alert: Predicted delay probability is {prob*100:.1f}%. Immediate intervention required.")
                        elif prob > 0.3:
                            st.warning(f"Warning: Moderate risk detected ({prob*100:.1f}%). Monitor closely.")
                        else:
                            st.success(f"Success: Low risk delivery ({prob*100:.1f}%). On-time arrival expected.")
                else:
                    # Handle API Errors (e.g. 500, 422)
                    try:
                        err_detail = response.json().get("detail", response.text)
                    except:
                        err_detail = response.text
                    st.error(f"Prediction Engine Error ({response.status_code}): {err_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("Operation Timed Out: The prediction engine took too long to respond.")
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
