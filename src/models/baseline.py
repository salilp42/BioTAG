"""Baseline model implementations."""

import numpy as np
from typing import List
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

def train_svm_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = 42
) -> List[float]:
    """Cross-validation for SVM on TDA features.
    
    Args:
        X: Input features
        y: Target labels
        n_splits: Number of CV folds
        n_repeats: Number of CV repeats
        random_state: Random seed
        
    Returns:
        List of per-fold accuracies
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    accuracies = []
    
    fold_count = 0
    for train_idx, test_idx in tqdm(cv.split(X, y), total=n_splits*n_repeats,
                                   desc="SVM_CV", leave=False):
        fold_count += 1
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        model = SVC(kernel='rbf', probability=True, gamma='auto', random_state=42)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        
        acc = accuracy_score(y_test_fold, preds)
        accuracies.append(acc)
    
    return accuracies
