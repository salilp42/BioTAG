"""Time series dataset loading and preprocessing module."""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification

class TimeSeriesDataset:
    """Handles loading and preprocessing of time series from aeon.
    
    Attributes:
        dataset_name: Name of the dataset to load
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        handle_missing: Whether to handle missing values
        imputation_method: Method to use for imputing missing values
    """
    
    RECOMMENDED_DATASETS = {
        'biomedical': ['NonInvasiveFetalECGThorax1'],
        'brain_activity': ['EOGHorizontalSignal', 'EOGVerticalSignal'],
        'other': ['LSST']
    }
    
    def __init__(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        handle_missing: bool = True,
        imputation_method: str = 'mean'
    ):
        """Initialize TimeSeriesDataset.
        
        Args:
            dataset_name: Name of the dataset to load
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            handle_missing: Whether to handle missing values
            imputation_method: Method to use for imputing missing values
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.handle_missing = handle_missing
        self.imputation_method = imputation_method
        
        self.scaler = StandardScaler()
        self.missing_stats = {
            'n_missing': 0,
            'missing_locations': [],
            'imputed_values': {}
        }
    
    def _detect_missing_values(self, X: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Detect missing values in the data.
        
        Args:
            X: Input data array
            
        Returns:
            Tuple of (has_missing, stats_dict)
        """
        missing_mask = np.isnan(X)
        n_missing = np.sum(missing_mask)
        stats = {
            'n_missing': n_missing,
            'missing_percentage': (n_missing / X.size * 100) if X.size > 0 else 0
        }
        if n_missing > 0:
            locs = np.where(missing_mask)
            stats['missing_locations'] = list(zip(*locs))
        return n_missing > 0, stats
    
    def _impute_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values using specified method.
        
        Args:
            X: Input data array with missing values
            
        Returns:
            Array with imputed values
        """
        X_imputed = X.copy()
        
        if self.imputation_method == 'mean':
            if len(X.shape) == 3:
                feat_means = np.nanmean(X, axis=(0,1), keepdims=True)
                X_imputed = np.where(np.isnan(X), feat_means, X)
            else:
                col_means = np.nanmean(X, axis=0)
                idxs = np.where(np.isnan(X))
                X_imputed[idxs] = np.take(col_means, idxs[1])
        
        elif self.imputation_method == 'forward':
            if len(X.shape) == 3:
                for i in range(X.shape[0]):
                    for j in range(X.shape[2]):
                        series = pd.Series(X[i,:,j])
                        X_imputed[i,:,j] = series.fillna(method='ffill').fillna(method='bfill')
            else:
                for j in range(X.shape[1]):
                    series = pd.Series(X[:,j])
                    X_imputed[:,j] = series.fillna(method='ffill').fillna(method='bfill')
        
        elif self.imputation_method in ['linear', 'cubic']:
            if len(X.shape) == 3:
                for i in range(X.shape[0]):
                    for j in range(X.shape[2]):
                        series = pd.Series(X[i,:,j])
                        X_imputed[i,:,j] = (series.interpolate(method=self.imputation_method)
                                          .fillna(method='bfill').fillna(method='ffill'))
            else:
                for j in range(X.shape[1]):
                    series = pd.Series(X[:,j])
                    X_imputed[:,j] = (series.interpolate(method=self.imputation_method)
                                    .fillna(method='bfill').fillna(method='ffill'))
        
        return X_imputed
    
    def _validate_imputation(self, X_imputed: np.ndarray) -> bool:
        """Validate that imputation was successful.
        
        Args:
            X_imputed: Imputed data array
            
        Returns:
            True if imputation was successful, False otherwise
        """
        if np.any(np.isnan(X_imputed)):
            warnings.warn("Imputation incomplete.")
            return False
        return True
    
    def inspect_data(self, X: np.ndarray, y: np.ndarray, split_name: str = "") -> None:
        """Print information about the data split.
        
        Args:
            X: Input features
            y: Target labels
            split_name: Name of the split (e.g., "Training", "Test")
        """
        print(f"\n{'-'*20} {split_name} Set Analysis {'-'*20}")
        print(f"Shape Information:\n  - X shape: {X.shape}\n  - y shape: {y.shape}")
        print("\nData Type / Range Info:")
        print(f"  - X dtype: {X.dtype}")
        print(f"  - X min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}, std: {X.std():.3f}")
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass Distribution:")
        for lbl, cnt in zip(unique, counts):
            print(f"  - Class {lbl}: {cnt} samples ({cnt/len(y)*100:.1f}%)")
    
    def load_ucr(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                               Optional[np.ndarray], Optional[np.ndarray]]:
        """Load dataset from UCR archive.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            print(f"\nLoading dataset: {self.dataset_name}")
            X, y, meta = load_classification(self.dataset_name, return_metadata=True)
            
            print("\nDataset Metadata:")
            for k,v in meta.items():
                print(f"  - {k}: {v}")
            
            # Aeon often stores shape as (n_samples, n_features, series_length)
            # We want (n_samples, series_length, n_features)
            if len(X.shape) == 3:
                X = np.transpose(X, (0,2,1))
            
            if meta.get('has_predefined_split', False):
                train_size = meta.get('train_size')
                X_train = X[:train_size]
                X_test = X[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, 
                    random_state=self.random_state, stratify=y
                )
            
            self.inspect_data(X_train, y_train, "Training")
            self.inspect_data(X_test, y_test, "Test")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {str(e)}")
            return None, None, None, None
    
    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data by handling missing values and scaling.
        
        Args:
            X_train: Training data
            X_test: Test data
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if self.handle_missing:
            has_missing, stats = self._detect_missing_values(X_train)
            self.missing_stats.update(stats)
            if has_missing:
                print(f"\nDetected {stats['n_missing']} missing values in training data "
                      f"({stats['missing_percentage']:.2f}%) -> Imputing")
                X_train = self._impute_missing_values(X_train)
                self._validate_imputation(X_train)
        
        print("Scaling data with StandardScaler...")
        if len(X_train.shape) == 3:
            n_train, t_train, f_train = X_train.shape
            n_test = X_test.shape[0]
            X_tr_2d = X_train.reshape(n_train, -1)
            X_te_2d = X_test.reshape(n_test, -1)
            X_tr_sc = self.scaler.fit_transform(X_tr_2d)
            X_te_sc = self.scaler.transform(X_te_2d)
            X_train_scaled = X_tr_sc.reshape(n_train, t_train, f_train)
            X_test_scaled = X_te_sc.reshape(n_test, t_train, f_train)
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def load_and_preprocess(self) -> Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]], 
                                         Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Load and preprocess the dataset.
        
        Returns:
            Tuple of ((X_train_scaled, y_train), (X_test_scaled, y_test))
        """
        X_train, X_test, y_train, y_test = self.load_ucr()
        if X_train is None or X_test is None:
            return (None, None), (None, None)
        
        # Map labels to [0..C-1]
        unique_labels = np.unique(y_train)
        label_map = {lbl:i for i,lbl in enumerate(unique_labels)}
        y_train = np.array([label_map[l] for l in y_train])
        y_test = np.array([label_map[l] for l in y_test])
        
        X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)
        return (X_train_scaled, y_train), (X_test_scaled, y_test)
