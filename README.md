# Bank Marketing Campaign Prediction

Predicting customer subscription to bank term deposits using machine learning

##  Project Overview

This project analyzes bank marketing campaign data from a Kaggle competition to predict whether customers will subscribe to a term deposit. The analysis includes comprehensive exploratory data analysis, statistical testing, custom preprocessing pipelines for different model types, and comparative evaluation of multiple machine learning models.

**Competition Result**: Achieved **0.96 ROC-AUC score** on Kaggle's held-out test set using the optimized XGBoost model. 

## Dataset

- **Source**: [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/overview)
- **Size**: [750,000 records 17 features]

##  Results

### Performance Summary

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|------|----------|
| Logistic Regression |0.9446| 0.60 | 0.70 |0.65 |
| XGBoost | **0.9665** | **0.68** | **0.79** | **0.73** |
| Neural Network | 0.9611 | 0.67 | 0.77 | 0.71 |


**Winner**: XGBoost demonstrated superior performance across all metrics, achieving a **0.96 ROC-AUC score** on Kaggle's held-out test set, demonstrating excellent generalization capability.

###  Model Diagnostics and Interpretability

####  Confusion Matrix (XGBoost)
This plot illustrates the model's ability to distinguish between positive and negative classes.

![Confusion Matrix](Plots/confusion_matrix.png)
The model correctly identified most non-subscribers while maintaining solid performance on the subscriber class. At a 0.73 threshold, it achieves a strong balance between recall and precision, minimizing false positives and false negatives.

#### Feature Importances
These are the top 15 most influential features for predicting customer subscription.

![Feature Importance](Plots/feature_importance.png)

Call duration emerged as the strongest predictor of term deposit subscription, with longer conversations strongly correlating with higher conversion rates. Communication type and previous contact history also play significant roles, while housing loan status and past campaign outcomes provide additional demographic and behavioral context. Together, the top three features, duration, contact type, and previous contact history all account for roughly 60% of the model’s predictive power, underscoring the importance of effective and targeted customer engagement.

##  Project Structure
```
bank-marketing-prediction/
├── Models/                           # Trained model artifacts
│   ├── logistic_regression.pkl
│   ├── xgboost_model.pkl
│   └── neural_network.pkl
│
├── Plots/                            # Visualizations
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── encoders/                         # Encoders used in preprocessing
│   ├── nn_onehot_encoder.pkl
│   ├── xgb_ordinal_encoder.pkl
│   └── README.md                     # Documentation for encoders
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── scalers/                          # Scalers used in preprocessing
│   └── nn_standard_scaler.pkl
│
├── scripts/                          # Python scripts
│   ├── preprocess_data.py
│   └── inference.py
│
├── transformers/                     # Fitted preprocessing transformers
│   ├── nn_yeo_johnson_transformer.pkl
│   ├── xgb_yeo_johnson_transformer.pkl
│   └── README.md                     # Documentation for transformers
│
├── predictions/                      # Competition submissions
│   ├── preds_xgb.csv
│   ├── preds_nn.csv
│   ├── preds_logreg.csv
│   └── submission_log.md             # Track submission scores
│
└── README.md                         # Project overview and instructions

```

## Setup Instructions

1. **Clone the Repository**
```
git clone https://github.com/oluwadunni1/Bank-Marketing-Campaign-Analysis-and-Prediction.git
cd Bank-Marketing-Campaign-Analysis-and-Prediction

```
2. **Download the dataset**. The dataset is available on [Kaggle](https://www.kaggle.com/competitions/playground-series-s5e8/overview). Download the CSV files (train.csv and test.csv) manually.
3. **Place the dataset**. Ensure the dataset files are placed in the following folder structure
```
    bank-marketing-prediction/
└── data/
    └── raw/
        ├── train.csv
        └── test.csv
```
⚠️ Important: All scripts and notebooks expect the raw data to be located in data/raw/. Using a different location may cause preprocessing or model inference to fail.

4. **Run Notebooks**

After placing the dataset, the notebooks can be executed in the following order:

- Exploratory Data Analysis:
notebooks/exploratory_data_analysis.ipynb – Review data distributions, relationships, and summary statistics.

- Data Preprocessing:
notebooks/data_preprocessing.ipynb – Apply feature engineering, encoding, scaling, and split the data for modeling.

- Model Training & Evaluation:
notebooks/model_training_evaluation.ipynb – Train the models, evaluate performance, and save trained artifacts.
