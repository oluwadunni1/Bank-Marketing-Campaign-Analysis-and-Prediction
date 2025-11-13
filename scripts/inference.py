"""
Notebook-ready Inference Script for Bank Marketing Classification

This script:
- Loads the test dataset
- Applies preprocessing pipelines for XGBoost and NN/Logistic models
- Loads trained models
- Generates and saves probability predictions for each model

Designed to run inside a Jupyter Notebook cell.
"""

import logging
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIG
# ----------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:/Users/hp/Documents/DA projects/Bank Marketing Prediction")
TEST_CSV = PROJECT_ROOT / "data/raw/test.csv"

XGB_MODEL = PROJECT_ROOT / "Models/Xgboost.pkl"
LOGREG_MODEL = PROJECT_ROOT / "Models/logistic_regression_model.pkl"
NN_MODEL = PROJECT_ROOT / "Models/Final_NeuralNetwork.keras"

XGB_TRANSFORMER = PROJECT_ROOT / "transformers/xgb_yeo_johnson_transformer.pkl"
XGB_ENCODER = PROJECT_ROOT / "encoders/xgb_ordinal_encoder.pkl"

NN_TRANSFORMER = PROJECT_ROOT / "transformers/nn_yeo_johnson_transformer.pkl"
NN_ENCODER = PROJECT_ROOT / "encoders/nn_onehot_encoder.pkl"
NN_SCALER = PROJECT_ROOT / "scalers/nn_standard_scaler.pkl"

PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)


# ----------------------------
# Feature Engineering
# ----------------------------
def engineer_pdays(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying pdays feature engineering...")
    df = df.copy()
    
    df['was_contacted_before'] = np.where(df['pdays'] != -1, 1, 0)
    with np.errstate(divide='ignore'):
        df['log_pdays'] = np.where(df['pdays'] > 0, np.log(df['pdays'] + 1), 0)
    
    df = df.drop(columns='pdays', errors='ignore')
    logger.info("pdays feature engineering complete.")
    return df


# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_for_xgb(X: pd.DataFrame, transformer_path: Path, encoder_path: Path) -> pd.DataFrame:
    logger.info("Starting preprocessing for XGBoost...")
    pt = joblib.load(transformer_path)
    encoder = joblib.load(encoder_path)
    
    numeric_cols = ['balance', 'duration', 'age', 'campaign', 'previous', 'log_pdays', 'day']
    skewed_cols = ['balance', 'duration']
    binary_cols = ['was_contacted_before']
    categorical_cols = [col for col in X.columns if col not in numeric_cols + binary_cols]

    X[skewed_cols] = pt.transform(X[skewed_cols])
    X[categorical_cols] = encoder.transform(X[categorical_cols])
    
    logger.info(f"XGBoost preprocessing complete. Shape: {X.shape}")
    return X


def preprocess_for_nn_logreg(X: pd.DataFrame, transformer_path: Path, encoder_path: Path, scaler_path: Path, clip_value: float = 5.0) -> pd.DataFrame:
    logger.info("Starting preprocessing for Neural Network / Logistic Regression...")
    
    pt = joblib.load(transformer_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    numeric_cols = ['balance', 'duration', 'age', 'campaign', 'previous', 'log_pdays', 'day']
    skewed_cols = ['balance', 'duration']
    binary_cols = ['was_contacted_before']
    categorical_cols = [col for col in X.columns if col not in numeric_cols + binary_cols]

    # Power transform
    X[skewed_cols] = pt.transform(X[skewed_cols])

    # One-hot encode categorical
    X_cat = encoder.transform(X[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    X_cat = pd.DataFrame(X_cat, columns=feature_names, index=X.index)

    # Combine numeric + binary + encoded categorical
    X = pd.concat([X.drop(columns=categorical_cols), X_cat], axis=1)

    # Scale and clip numeric features
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    X[numeric_cols] = X[numeric_cols].clip(-clip_value, clip_value)
    
    logger.info(f"NN/Logistic preprocessing complete. Shape: {X.shape}")
    return X


# ----------------------------
# Prediction & Save
# ----------------------------
def predict_and_save(model, X: pd.DataFrame, output_path: Path, model_name: str, ids=None, use_keras=False):
    logger.info(f"Generating predictions for {model_name}...")
    
    if use_keras:
        y_pred_proba = model.predict(X).flatten()
    else:
        y_pred_proba = model.predict_proba(X)[:, 1]
    
    preds_df = pd.DataFrame({
        "id": ids if ids is not None else np.arange(1, len(y_pred_proba)+1),
        "probability": y_pred_proba
    })
    
    preds_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


# ----------------------------
# MAIN INFERENCE FLOW
# ----------------------------
# Load test data
df_test_full = pd.read_csv(TEST_CSV)

# Keep IDs for Kaggle submission
ids = df_test_full['id'].copy() if 'id' in df_test_full.columns else np.arange(1, len(df_test_full)+1)

# Drop ID before preprocessing
df_test = df_test_full.drop(columns=['id'], errors='ignore')

# Feature engineering
df_test = engineer_pdays(df_test)

# Preprocess test sets
X_test_xgb = preprocess_for_xgb(df_test.copy(), XGB_TRANSFORMER, XGB_ENCODER)
X_test_nn = preprocess_for_nn_logreg(df_test.copy(), NN_TRANSFORMER, NN_ENCODER, NN_SCALER)

# Load models
xgb_model = joblib.load(XGB_MODEL)
logreg_model = joblib.load(LOGREG_MODEL)
nn_model = load_model(NN_MODEL)

# Predict and save
predict_and_save(xgb_model, X_test_xgb, PREDICTIONS_DIR / "preds_xgb.csv", "XGBoost", ids=ids)
predict_and_save(logreg_model, X_test_nn, PREDICTIONS_DIR / "preds_logreg.csv", "Logistic Regression", ids=ids)
predict_and_save(nn_model, X_test_nn, PREDICTIONS_DIR / "preds_nn.csv", "Neural Network", ids=ids, use_keras=True)

logger.info("✅ All model predictions completed successfully!")
