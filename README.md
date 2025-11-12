# Bank Marketing Campaign Prediction

Predicting customer subscription to bank term deposits using machine learning

##  Project Overview

This project analyzes bank marketing campaign data from a Kaggle competition to predict whether customers will subscribe to a term deposit. The analysis includes comprehensive exploratory data analysis, statistical testing, custom preprocessing pipelines for different model types, and comparative evaluation of multiple machine learning models.

**Competition Result**: Achieved **0.96 ROC-AUC score** on Kaggle's held-out test set using the optimized XGBoost model. 

## Dataset

- **Source**: [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/overview)
- **Size**: [750,000 records 17 features]

##  Results

### Cross-Validation Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|------|----------|
| Logistic Regression |0.9446| 0.95 | X.XX | X.XX |
| XGBoost | **X.XX** | **X.XX** | **X.XX** | **X.XX** |
| Neural Network | X.XX | X.XX | X.XX | X.XX |


**Winner**: XGBoost demonstrated superior performance across all metrics, achieving a **0.96 ROC-AUC score** on Kaggle's held-out test set, demonstrating excellent generalization capability.

###  Model Diagnostics and Interpretability

####  Confusion Matrix (XGBoost)
Displays model performance in terms of correctly and incorrectly classified customers (Threshold = 0.75).

![Confusion Matrix](images/confusion_matrix_xgb.png)

#### Top 15 Feature Importances
Shows the most influential customer attributes driving subscription predictions.

![Feature Importances](images/feature_importances_xgb.png)

