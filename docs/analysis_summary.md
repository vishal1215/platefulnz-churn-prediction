# PlatefulNZ Customer Churn Prediction Analysis

## Overview
This report summarizes the methodology and key findings from a customer churn analysis for **PlatefulNZ**, a subscription‑based meal service.  The goal of the study was to predict which customers are at risk of churning and to identify the factors most strongly associated with retention.

## Dataset
- Approximately **200,000** customers with **26 behavioural and demographic variables**.
- Target variable `retained_binary`: `0` = churn, `1` = retained.
- Observed churn rate: **23.2 %** (retained‑to‑churn ratio of **3.3 : 1**).

## Feature Engineering & Selection
- Data cleaning: removed IDs, parsed numeric fields, converted target to a factor.
- Stratified sampling used to split data into training (80 %) and test sets (20 %).
- Top drivers of churn (variable importance on sampled data):
  - **discounted_rate_last_purchase**
  - **num_purchases**
  - **satisfaction_survey**
  - **last_browser**
  - **weeks_since_last_purchase**
  - **location**

## Models Evaluated
- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- Neural Network (nnet)  
- k‑Nearest Neighbours  

## Evaluation & Results
- Models were tuned using five‑fold cross‑validation.
- ROC AUC was the primary metric for selecting the best model.
- **XGBoost** achieved the highest ROC AUC of **0.957**, with:
  - Accuracy: **0.896**
  - Sensitivity (Recall): **0.858**
  - Specificity: **0.908**
  - Precision: **0.737**
  - F1 Score: **0.793**

## Key Insights
- **Satisfaction drives churn:** customers with low survey scores (1–2 stars) churned at **48 %**, whereas those with high scores (> 3 stars) churned at **16 %**.
- **Recency matters:** customers with ≥ 7 weeks since last purchase had an **89 %** churn rate, while those with a purchase in the last 4 weeks churned at **24 %**.
- **Regional concentration:** three regions – **Auckland**, **Canterbury**, and **Wellington** – account for over **56 %** of all churners, with Auckland alone representing **30 %**.

## Business Recommendations
- **Engage dissatisfied customers quickly:** follow up within 24 hours of a low satisfaction survey.
- **Incentivize recent activity:** send offers to customers who haven’t purchased in 5–7 weeks.
- **Segment by region:** tailor promotions to high‑churn locations.
- **Monitor discount usage:** reduce dependency on coupons while maintaining retention.

## Conclusion
The churn prediction system provides a robust framework for identifying high‑risk customers and developing targeted interventions.  Using a combination of advanced machine‑learning models and business insights, the analysis delivers actionable strategies that can improve retention and customer lifetime value for **PlatefulNZ**.
