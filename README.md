# 📊 PlatefulNZ Customer Churn Prediction System (R | Tidymodels)

A full end-to-end machine-learning pipeline for predicting customer churn and identifying retention strategies for a subscription-based food service.

PlatefulNZ Churn Prediction is a full analytical workflow built in R to model, predict, and explain customer churn behaviour. The project integrates exploratory analysis, feature engineering, statistical modelling, and multiple machine-learning algorithms to evaluate churn risk and recommend retention strategies.

## 🤔 Project Overview
The objective is to identify customers at high risk of churning and design actionable retention strategies. The project integrates:
- Data cleaning & preprocessing
- Exploratory data analysis (EDA)
- Feature engineering & selection
- Machine learning model building using the **tidymodels** framework
- Cross-validation & hyperparameter tuning
- ROC–AUC-based model comparison
- Business interpretation

## 🚀 Key Features
### 1. Exploratory Data Analysis
Highlights include:
- Churn rate by location
- Churn rate by weeks since last purchase
- Impact of customer satisfaction on churn
- Purchase frequency patterns
- Average add-on value segments

### 2. Feature Engineering
- Derived variables for engagement & purchase behaviour
- Grouped categories (e.g., purchase bands, satisfaction groups)
- One-hot encoding for categorical predictors
- Scaling/centering for numerical predictors

### 3. Machine Learning Models
Evaluated multiple classification algorithms:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Neural Network (nnet)
- k-Nearest Neighbours

### 4. Model Evaluation & Selection
- Performed cross-validation and hyperparameter tuning using tidymodels.
- Compared models using ROC-AUC, accuracy, sensitivity, and F1 score.
- Selected XGBoost as the best model with ROC AUC ≈ 0.957.

### 5. Business Interpretation
- Profiles top predictors of churn.
- Explains which segments are at highest risk.
- Maps model findings to actionable interventions (e.g., targeted email campaigns, satisfaction follow-ups, discount strategies).
- Provides recommendations to improve customer lifetime value.

## 📊 Visuals
The `visuals/` directory contains key visualizations generated during the analysis, including:
- `churn_by_location.png`
- `churn_by_weeks_since_purchase.png`
- `churn_by_satisfaction.png`
- `churn_by_num_purchases.png`
- `variable_importance.png`
- `roc_curve_xgboost.png`
- `xgboost_top_configs.png`
- `confusion_matrix_xgboost.png`
- `model_performance_table.png`

## 🧰 Repository Structure
```
platefulnz-churn-prediction/
├── README.md                  – Project overview and documentation
├── churn_modelling.R          – Complete R modelling script
├── visuals/                   – Figures from the analysis
│   ├── churn_by_location.png
│   ├── churn_by_weeks_since_purchase.png
│   ├── churn_by_satisfaction.png
│   ├── churn_by_num_purchases.png
│   ├── variable_importance.png
│   ├── roc_curve_xgboost.png
│   ├── xgboost_top_configs.png
│   ├── confusion_matrix_xgboost.png
│   └── model_performance_table.png
└── docs/
    ├── analysis_summary.md    – Detailed analysis summary
    └── poster.pdf             – Poster summarizing the project (optional)
```

## 🛠 Tech Stack
- R (tidyverse, tidymodels, ggplot2, dplyr, yardstick, tune, parsnip, themis)
- Machine learning classification algorithms
- Data visualization

## 💼 Outcomes
This project produces a robust churn prediction framework that helps the business:
- Identify high-risk customers early.
- Understand key drivers of churn.
- Tailor retention initiatives (e.g., outreach, discounts, satisfaction follow-ups).
- Quantify the potential impact of targeted retention strategies.

Feel free to clone this repository and adapt the modelling framework to your own customer retention scenarios.
