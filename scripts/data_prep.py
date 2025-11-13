"""
Data Preprocessing Pipeline for Bank Marketing Campaign Classification

This script handles data preprocessing for both XGBoost and Neural Network models,
including feature engineering, encoding, scaling, and train-test splitting.

Usage:
    python preprocess_data.py --config config.yaml
    python preprocess_data.py --test-size 0.2 --random-state 42
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import warnings

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data preprocessing for machine learning models."""
    
    def __init__(
        self,
        project_root: Path,
        test_size: float = 0.2,
        random_state: int = 42,
        clip_value: float = 5.0
    ):
        """
        Initialize the preprocessor.
        
        Args:
            project_root: Root directory of the project
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            clip_value: Value to clip scaled features (prevents gradient explosion)
        """
        self.project_root = Path(project_root)
        self.test_size = test_size
        self.random_state = random_state
        self.clip_value = clip_value
        
        # Define feature groups
        self.numeric_cols = ['balance', 'duration', 'age', 'campaign', 
                            'previous', 'log_pdays', 'day']
        self.skewed_cols = ['balance', 'duration']
        self.binary_cols = ['was_contacted_before']
        
        # Setup directories
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        # Resolve to absolute path to handle relative paths correctly
        self.project_root = self.project_root.resolve()
        
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.transformers_dir = self.project_root / "transformers"
        self.encoders_dir = self.project_root / "encoders"
        self.scalers_dir = self.project_root / "scalers"
        
        # Create directories
        for directory in [self.processed_dir, self.transformers_dir, 
                         self.encoders_dir, self.scalers_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def load_data(self, filename: str = "train.csv") -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.raw_dir / filename
        logger.info(f"Loading dataset from: {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Applying feature engineering...")
        df = df.copy()
        
        # Drop ID column (no predictive value)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        # Engineer pdays feature
        df['was_contacted_before'] = np.where(df['pdays'] != -1, 1, 0)
        
        with np.errstate(divide='ignore'):
            df['log_pdays'] = np.where(
                df['pdays'] > 0,
                np.log(df['pdays'] + 1),
                0
            )
        
        df = df.drop(columns='pdays')
        
        logger.info("Feature engineering complete")
        return df
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with stratification.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data (test_size={self.test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        self._log_class_distribution(y_train, "Training")
        self._log_class_distribution(y_test, "Test")
        
        return X_train, X_test, y_train, y_test
    
    def _log_class_distribution(self, y: pd.Series, dataset_name: str) -> None:
        """Log class distribution for a dataset."""
        dist = y.value_counts(normalize=True)
        logger.info(f"{dataset_name} class distribution - "
                   f"Class 0: {dist[0]:.2%}, Class 1: {dist[1]:.2%}")
    
    def _get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify categorical columns."""
        return [col for col in X.columns 
                if col not in self.numeric_cols + self.binary_cols]
    
    def preprocess_for_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for XGBoost model.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (processed X_train, processed X_test)
        """
        logger.info("=" * 60)
        logger.info("Starting XGBoost Preprocessing")
        logger.info("=" * 60)
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        categorical_cols = self._get_categorical_columns(X_train)
        logger.info(f"Features - Numeric: {len(self.numeric_cols)}, "
                   f"Categorical: {len(categorical_cols)}, "
                   f"Binary: {len(self.binary_cols)}")
        
        # Power transform skewed features
        logger.info("Applying Yeo-Johnson power transformation...")
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_train[self.skewed_cols] = pt.fit_transform(X_train[self.skewed_cols])
        X_test[self.skewed_cols] = pt.transform(X_test[self.skewed_cols])
        
        joblib.dump(pt, self.transformers_dir / "xgb_yeo_johnson_transformer.pkl")
        logger.info(f"Transformer saved to {self.transformers_dir}")
        
        # Ordinal encode categorical features
        logger.info("Applying ordinal encoding...")
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        
        joblib.dump(encoder, self.encoders_dir / "xgb_ordinal_encoder.pkl")
        logger.info(f"Encoder saved to {self.encoders_dir}")
        
        # Validation
        self._validate_processed_data(X_train, X_test, "XGBoost")
        
        logger.info(f"XGBoost preprocessing complete. Final shape: {X_train.shape}")
        return X_train, X_test
    
    def preprocess_for_nn(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for Neural Network model.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (processed X_train, processed X_test)
        """
        logger.info("=" * 60)
        logger.info("Starting Neural Network Preprocessing")
        logger.info("=" * 60)
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        categorical_cols = self._get_categorical_columns(X_train)
        logger.info(f"Features - Numeric: {len(self.numeric_cols)}, "
                   f"Categorical: {len(categorical_cols)}, "
                   f"Binary: {len(self.binary_cols)}")
        
        # Power transform with standardization
        logger.info("Applying Yeo-Johnson power transformation with standardization...")
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train[self.skewed_cols] = pt.fit_transform(X_train[self.skewed_cols])
        X_test[self.skewed_cols] = pt.transform(X_test[self.skewed_cols])
        
        joblib.dump(pt, self.transformers_dir / "nn_yeo_johnson_transformer.pkl")
        logger.info(f"Transformer saved to {self.transformers_dir}")
        
        # One-hot encode categorical features
        logger.info("Applying one-hot encoding...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        
        # Convert to DataFrames
        feature_names = encoder.get_feature_names_out(categorical_cols)
        X_train_cat = pd.DataFrame(X_train_cat, columns=feature_names, 
                                   index=X_train.index)
        X_test_cat = pd.DataFrame(X_test_cat, columns=feature_names, 
                                 index=X_test.index)
        
        # Combine with original features
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_cat], 
                           axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_cat], 
                          axis=1)
        
        # Align columns
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        joblib.dump(encoder, self.encoders_dir / "nn_onehot_encoder.pkl")
        logger.info(f"Encoder saved. Created {len(feature_names)} one-hot features")
        
        # Standard scale numeric features
        logger.info("Applying standard scaling to numeric features...")
        scaler = StandardScaler()
        X_train[self.numeric_cols] = scaler.fit_transform(X_train[self.numeric_cols])
        X_test[self.numeric_cols] = scaler.transform(X_test[self.numeric_cols])
        
        # Clip extreme values
        X_train[self.numeric_cols] = X_train[self.numeric_cols].clip(
            -self.clip_value, self.clip_value
        )
        X_test[self.numeric_cols] = X_test[self.numeric_cols].clip(
            -self.clip_value, self.clip_value
        )
        
        joblib.dump(scaler, self.scalers_dir / "nn_standard_scaler.pkl")
        logger.info(f"Scaler saved. Values clipped to [{-self.clip_value}, "
                   f"{self.clip_value}]")
        
        # Validation
        self._validate_processed_data(X_train, X_test, "Neural Network")
        
        logger.info(f"NN preprocessing complete. Final shape: {X_train.shape}")
        return X_train, X_test
    
    def _validate_processed_data(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        model_type: str
    ) -> None:
        """Validate processed data for common issues."""
        assert X_train.isnull().sum().sum() == 0, f"{model_type}: NaN in train data!"
        assert X_test.isnull().sum().sum() == 0, f"{model_type}: NaN in test data!"
        assert X_train.shape[1] == X_test.shape[1], f"{model_type}: Feature mismatch!"
        logger.info(f"Validation passed for {model_type}")
    
    def save_processed_data(
        self,
        X_train_xgb: pd.DataFrame,
        X_test_xgb: pd.DataFrame,
        X_train_nn: pd.DataFrame,
        X_test_nn: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Save all processed datasets to disk.
        
        Args:
            X_train_xgb: XGBoost training features
            X_test_xgb: XGBoost test features
            X_train_nn: Neural network training features
            X_test_nn: Neural network test features
            y_train: Training labels
            y_test: Test labels
        """
        logger.info("Saving processed datasets...")
        
        datasets = {
            "X_train_xgb.pkl": X_train_xgb,
            "X_test_xgb.pkl": X_test_xgb,
            "X_train_nn.pkl": X_train_nn,
            "X_test_nn.pkl": X_test_nn,
            "y_train.pkl": y_train,
            "y_test.pkl": y_test
        }
        
        for filename, data in datasets.items():
            joblib.dump(data, self.processed_dir / filename)
        
        logger.info(f"All datasets saved to {self.processed_dir}")
    
    def run_full_pipeline(self, input_file: str = "train.csv") -> Dict:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            input_file: Name of input CSV file
            
        Returns:
            Dictionary containing all processed datasets
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Load and prepare data
        df = self.load_data(input_file)
        df = self.engineer_features(df)
        
        # Split features and target
        X = df.drop(columns=['y'])
        y = df['y']
        
        # Train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Preprocess for both model types
        X_train_xgb, X_test_xgb = self.preprocess_for_xgboost(X_train, X_test)
        X_train_nn, X_test_nn = self.preprocess_for_nn(X_train, X_test)
        
        # Save everything
        self.save_processed_data(
            X_train_xgb, X_test_xgb, X_train_nn, X_test_nn, y_train, y_test
        )
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETE!")
        logger.info("=" * 60)
        
        return {
            'X_train_xgb': X_train_xgb,
            'X_test_xgb': X_test_xgb,
            'X_train_nn': X_train_nn,
            'X_test_nn': X_test_nn,
            'y_train': y_train,
            'y_test': y_test
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess data for ML models'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default='..',
        help='Project root directory (default: parent directory of script)'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default='train.csv',
        help='Input CSV filename (default: train.csv)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--clip-value',
        type=float,
        default=5.0,
        help='Value to clip scaled features (default: 5.0)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        project_root=args.project_root,
        test_size=args.test_size,
        random_state=args.random_state,
        clip_value=args.clip_value
    )
    
    # Run pipeline
    try:
        results = preprocessor.run_full_pipeline(args.input_file)
        logger.info("Preprocessing completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()