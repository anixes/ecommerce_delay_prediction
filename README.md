# 🚚 Ecommerce Delivery Delay Prediction

[![CI Pipeline](https://github.com/anixes/ecommerce_delay_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/anixes/ecommerce_delay_prediction/actions)
[![DVC Managed](https://img.shields.io/badge/Data-DVC-blue.svg)](https://dvc.org)
[![Live Dashboard](https://img.shields.io/badge/Deployment-Live-brightgreen.svg?style=flat-square)](http://13.204.212.148:8502)

**Public URL**: <http://13.204.212.148:8502>

> [!TIP]
> **Mobile Users**: If the link doesn't open from the GitHub app, try opening it in your mobile browser (Chrome/Safari) directly: **<http://13.204.212.148:8502>**
>

## 📖 Overview

This project builds a professional-grade MLOps system that predicts whether an e-commerce order from the **Olist Brazilian Dataset** will be delivered late. It demonstrates a full-lifecycle "Push-to-Deploy" pipeline, integrating advanced machine learning, data engineering, and automated cloud infrastructure.

---

## 🏗️ System Design (MLOps Architecture)

The system is designed for high reliability and reproducibility, utilizing a modern MLOps stack.

```mermaid
graph TD
    subgraph Development
        A[Git Push] --> B[GitHub Actions]
    end
    
    subgraph CI_Pipeline
        B --> C[Linting: Ruff]
        C --> D[Tests: Pytest]
        D --> E[DVC Pull from DAGsHub]
        E --> F[Docker Build & Model Baking]
    end
    
    subgraph Registry_Containerization
        F --> G[Push SHA-tagged Images to GHCR]
    end
    
    subgraph Production_EC2
        G --> H[SSH Deploy]
        H --> I[Docker Compose Pull]
        I --> J[Live Dashboard :8502]
        I --> K[FastAPI Service :8001]
    end
    
    subgraph MLOps_Backbone
        L[DAGsHub / DVC] -- Models/Data --> E
        M[MLflow / Optuna] -- Tracking --> L
    end
```

---

## 📂 Project Structure

```text
├── .github/workflows/     # CI/CD (Lint, Test, Bake, Deploy)
├── delivery_delay_prediction/
│   ├── modeling/          # Train, Tune, Evaluate (CatBoost, Optuna)
│   ├── features.py        # Logic for 20+ engineered features
│   └── config.py          # Environment & Path management
├── docker/                # Deployment-ready Dockerfiles (3.11-slim)
├── models/                # Binary artifacts (.cbm) pointers (DVC)
├── src/
│   ├── api/               # FastAPI prediction service
│   └── dashboard/         # Streamlit interactive application
├── tests/                 # Automated API & Feature verification
├── docker-compose.yml     # Multi-container orchestration
├── dvc.yaml               # DVC Data Pipeline definition
└── requirements.txt       # Project dependencies
```

---

## 🛠️ Key Technical Features

### **1. Professional MLOps Stack**

- **Data Version Control (DVC)**: Managed via **DAGsHub**, ensuring models and large analytical datasets are versioned alongside code without bloating the repository.
- **Model Baking Strategy**: The CI pipeline pulls real model artifacts from DAGsHub and "bakes" them into production Docker images, ensuring zero external dependencies during runtime.
- **Traceability**: Every deployment is deterministic, using **Git Commit SHAs** as container tags in the GitHub Container Registry (GHCR).

### **2. Automated CI/CD (GitHub Actions)**

- **Linting & Safety**: Automated code quality checks with `Ruff`.
- **"Fail-Fast" Guards**: CI validates model existence and checksums before the build phase.
- **Push-to-Deploy**: Successful builds trigger an automated SSH update to **AWS EC2**, using `docker compose` for zero-touch updates.

### **3. Production Hardening**

- **Disk Safety**: Service-level log rotation implemented to prevent EC2 disk exhaustion.
- **Reliability**: Configured Docker health checks to ensure the Dashboard and API self-heal if unresponsive.
- **Infrastructure**: Running on AWS EC2 with an Elastic IP for persistent availability.

---

## 🚀 Getting Started

### **Local Deployment**

```bash
# Clone the repo
git clone https://github.com/anixes/ecommerce_delay_prediction.git
cd ecommerce_delay_prediction

# Start the full stack (API + Dashboard)
docker-compose up --build
```

Access the local dashboard at `http://localhost:8502`.

---

## 📈 Model Performance

- **Model**: CatBoost Classifier
- **Metric (PR-AUC)**: **0.3540** (Significant improvement over baseline)
- **Top Features**: Route-specific delay rates, `days_to_nearest_holiday`, and seller-to-customer logistics velocity.

---
*Created for portfolio purposes. Demonstrates skills in: Data Engineering, ML Modeling, Docker, CI/CD, and MLOps.*
