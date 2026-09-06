# Knowledge-Guided Decision Intelligence for Customer Churn

Research code for:

**Knowledge-Guided Decision Intelligence for Cost-Aware Customer Churn Intervention Under Predictive Uncertainty**

## Main Code

All executable research logic is contained in `main.py`.

## Pipeline

IBM Telco Customer Data → Data Cleaning → AI Models → Probability Calibration → Predictive Uncertainty → CLTV + Business Costs → Knowledge-Guided Decision Intelligence → Business Utility Evaluation → Statistical Tests → Sensitivity Analysis

## Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Run

python main.py --data data/Telco_customer_churn.xlsx.zip --output results/paper --mode paper

## Methodological Note

Customer characteristics, churn outcomes and CLTV come from the IBM Telco Customer Churn dataset. Retention cost, intervention success probability, human review cost, management budget, and uncertainty penalties are simulated business assumptions evaluated through sensitivity analysis.
