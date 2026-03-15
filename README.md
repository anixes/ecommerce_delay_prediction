# E-Commerce Delivery Delay Prediction

## Overview

This project builds a machine learning system that predicts whether an e-commerce order will be delivered late. Early detection allows a marketplace platform to proactively intervene by prioritizing shipments, notifying customers, or monitoring seller performance.

The project uses the **Olist Brazilian E-Commerce Dataset**, a relational dataset containing approximately 100,000 orders distributed across multiple tables. Instead of treating the dataset as separate CSV files, the tables are loaded into a relational database and transformed into a machine learning dataset using SQL queries.

---

## Problem Statement

Late deliveries negatively impact customer satisfaction and often lead to poor reviews, refunds, and customer churn. The goal of this project is to build a classification model that predicts whether an order will arrive later than its estimated delivery date.

### Target Variable

**Late Delivery**

- `1` = Order delivered after the estimated delivery date
- `0` = Order delivered on or before the estimated date

---

## Dataset

**Dataset**: Olist Brazilian E-Commerce Dataset (Kaggle)

The dataset represents a real-world e-commerce marketplace database with multiple relational tables:

- Customers
- Orders
- Order Items
- Order Payments
- Order Reviews
- Sellers
- Products
- Geolocation

Each table represents a different component of the marketplace system.

---

## Data Engineering Pipeline

### Database Construction

The nine dataset tables were first loaded into a **MySQL database**. This simulates a realistic production data environment where transactional data is stored in relational databases rather than flat files.

### SQL Data Wrangling

The modeling dataset was constructed using **SQL JOIN queries and aggregations** across multiple tables. Key operations included:

- Joining orders with customers, sellers, and product data
- Aggregating historical seller and customer statistics
- Combining logistics and payment information at the order level
- Generating derived features through SQL transformations

The resulting analytical dataset represents one row per order and serves as the input for machine learning models.

---

## Feature Engineering

### 🚀 Process Delay Sprint (New)

To push performance toward the 0.40 target, we added targeted features to address specific logistics blindspots:

- **Holiday Proximity**: `days_to_nearest_holiday` captures the "peak season" stress on the shipping network.
- **Seller Stress**: `seller_stress_ratio` identifies individual sellers falling behind their regional peers.
- **Hub Interaction**: Flags for São Paulo (`SP`) hub congestion (`is_hub_delivery`).

### Logistics & Core Features

- **Required Velocity**: `distance_km` / `lead_time_estimated` (High impact).
- **Route Risk**: Historical delay rates for specific origin-destination pairs.
- **Backlog**: Real-time state-level order volume.

---

## Modeling & Results

After extensive hyperparameter tuning using Optuna, **CatBoost** was selected as the final production model due to its superior handling of the categorical nature of the Olist dataset and its ability to capture complex non-linear interactions between logistics variables.

### Final Performance

- **Primary Metric (PR-AUC)**: **0.3540**
- **Model Type**: CatBoost Classifier

### Top Drivers of Delay

Interpretability through feature importance reveals the primary causes of lateness:

1. **`purchase_month`**: Seasonal volume is the #1 driver.
2. **`required_velocity`**: Extremely tight logistics schedules.
3. **`route_delay_rate`**: Structural inefficiencies in specific shipping routes.
4. **`days_to_nearest_holiday`**: Unexpected surges in process delays.

---

## Feature Generation Pipeline

### Python Feature Pipeline

The feature generation is handled in `delivery_delay_prediction/features.py`, which performs:

- Categorical encoding and missing value imputation.
- Log transformations of skewed distance/volume data.
- Interaction feature creation (e.g., `tight_schedule_risk`).

---

## Technologies Used

- **Data Engineering**: MySQL, Python (Pandas, SQLAlchemy)
- **Machine Learning**: CatBoost (Winner), LightGBM (Tuning Phase), Scikit-learn, SHAP
- **MLOps & Tracking**: Optuna (Tuning), MLflow (Experiment Tracking), DVC
- **Deployment**: FastAPI, Docker, Streamlit

---

## Key Skills Demonstrated

- Relational data engineering with SQL
- Large-scale feature engineering across multiple tables
- Classification modeling for logistics prediction
- Experiment tracking and model versioning
- API deployment and interactive ML applications
