# 🦈 SharkPredict AI: Venture Capital Forecasting Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](YOUR_STREAMLIT_LINK_HERE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**SharkPredict AI** is a data-driven framework designed to bridge the gap between human intuition and predictive analytics in venture capital. By analyzing **1,400+ pitches** spanning **17 seasons** of *Shark Tank US*, we developed a Machine Learning model that transforms raw pitch metrics into objective statistical probabilities.

### 🎯 Key Objectives
* Build a high-performance **Ensemble Model** to forecast investment outcomes.
* Achieve a high accuracy threshold (**80.4%**) using specialized feature engineering.
* Deploy an **Explainable AI (XAI)** advisor to provide real-time strategic coaching.

---

## 🚀 Live Demo
You can try the live, interactive dashboard here:  
**👉 [YOUR_STREAMLIT_LINK_HERE]**

---

## 🛠️ Technical Architecture & Methodology

### 1. Data Engineering
* **Data Acquisition:** Analyzed a massive historical dataset from Shark Tank US.
* **Preprocessing:** Handled missing values, standardized features with `StandardScaler`, and removed biometric bias.
* **Feature Engineering:** Developed two custom ratios:
    * **Valuation Ratio:** $Ask / (Equity + 1)$
    * **View-per-Ask:** $Viewership / (Ask + 1)$
* **Class Balancing:** Resolved historical "Rejection Bias" using **Manual Oversampling** for a balanced 50/50 split.

### 2. Machine Learning Pipeline
* **Algorithm:** Tuned **Random Forest Classifier** (500 estimators).
* **Inference Engine:** A custom-built advisor that translates model weights into actionable tips.
* **Serialization:** Model and assets preserved using **Pickle** for production deployment.

---

## 📊 Performance Results
| Metric | Value | Significance |
| :--- | :--- | :--- |
| **Model Accuracy** | **80.4%** | Highly reliable prediction foundation |
| **Ensemble Model** | Random Forest | Optimal for non-linear business patterns |
| **Key Predictor** | **US Viewership** | Proves marketing is a primary investment driver |

---

## 💻 Tech Stack
* **Language:** Python 3.12
* **Frameworks:** Streamlit, Scikit-Learn
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **Deployment:** Streamlit Community Cloud

---

## 📁 Repository Structure
```text
├── app.py                   # Main Streamlit dashboard code
├── shark_model.pkl          # Serialized Random Forest brain
├── scaler.pkl               # Standardized feature scaling logic
├── feature_names.pkl        # List of input columns for prediction
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation
