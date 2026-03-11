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

### Customer Features
- Number of previous purchases
- Historical review behavior
- Purchase frequency

### Seller Features
- Historical delivery delay rate
- Average review score
- Seller reliability indicators

### Logistics Features
- Seller–customer geographic distance
- Shipping performance by region

### Order Features
- Number of items
- Total order value
- Product categories
- Payment type

### Temporal Features
- Order month
- Weekday vs weekend
- Seasonal trends

---

## Modeling Approach

The task is formulated as a **binary classification problem**.

**Models evaluated**:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (LightGBM / XGBoost)

Hyperparameter tuning is performed using cross-validation.

---

## Evaluation Metrics

Model performance is evaluated using:
- ROC-AUC
- Precision
- Recall
- F1 Score

Because delayed deliveries directly affect customer experience, the model prioritizes **recall**, ensuring the system detects as many potential delays as possible.

---

## Model Explainability

SHAP values are used to interpret model predictions and identify the primary drivers of delivery delays.

Common influential factors include:
- Long shipping distances
- Historically unreliable sellers
- Specific product categories
- Seasonal demand spikes

---

## System Output

The system produces a delivery risk score for each order.

**Example output**:
> Predicted Delay Probability: 0.74
> Risk Level: High

A REST API built with **FastAPI** serves the trained model and returns predictions for incoming order data. An interactive **Streamlit dashboard** allows users to input order features and visualize delivery risk predictions in real time.

---

## Technologies Used

- **Data Engineering**: MySQL, Python (Pandas, SQLAlchemy)
- **Machine Learning**: Scikit-learn, LightGBM / XGBoost, SHAP
- **MLOps & Experiment Tracking**: MLflow, DVC
- **Backend & Deployment**: FastAPI, Docker
- **Frontend Interface**: Streamlit

---

## Key Skills Demonstrated

- Relational data engineering with SQL
- Large-scale feature engineering across multiple tables
- Classification modeling for logistics prediction
- Experiment tracking and model versioning
- API deployment and interactive ML applications
