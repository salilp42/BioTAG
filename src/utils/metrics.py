"""Utility functions for model evaluation metrics."""

import numpy as np
from sklearn.metrics import calibration_curve
from typing import Dict, List, Tuple, Optional

def calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    n_classes: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve for binary or multi-class (OvR approach).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        n_classes: Number of classes
        
    Returns:
        Tuple of (fraction_positives, mean_predictions)
    """
    if n_classes == 2:
        # Binary case
        prob_pos = y_prob[:, 1]
        frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins)
    else:
        # Multi-class: average calibration curves across classes
        frac_pos_list = []
        mean_pred_list = []
        
        for c in range(n_classes):
            c_mask = (y_true == c).astype(int)
            prob_pos = y_prob[:, c]
            frac_pos, mean_pred = calibration_curve(c_mask, prob_pos, n_bins=n_bins)
            frac_pos_list.append(frac_pos)
            mean_pred_list.append(mean_pred)
        
        frac_pos = np.mean(frac_pos_list, axis=0)
        mean_pred = np.mean(mean_pred_list, axis=0)
    
    return frac_pos, mean_pred

def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score for multi-class prediction.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score
    """
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    
    # One-hot encode true labels
    one_hot_true = np.zeros_like(y_prob)
    one_hot_true[np.arange(n_samples), y_true] = 1.0
    
    # Compute MSE
    mse = (y_prob - one_hot_true)**2
    brier = mse.sum() / (n_samples * n_classes)
    return brier

def compute_confidence_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5
) -> Dict[str, float]:
    """Compute various confidence-based metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of confidence bins
        
    Returns:
        Dictionary of metrics including:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Average confidence
        - Average accuracy
        - Confidence-accuracy correlation
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_metrics = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_size = np.sum(in_bin) / len(confidences)
            
            # Update ECE and MCE
            ece += bin_size * np.abs(bin_acc - bin_conf)
            mce = max(mce, np.abs(bin_acc - bin_conf))
            
            bin_metrics.append({
                'bin_acc': bin_acc,
                'bin_conf': bin_conf,
                'bin_size': bin_size
            })
    
    # Compute overall metrics
    avg_confidence = np.mean(confidences)
    avg_accuracy = np.mean(accuracies)
    conf_acc_corr = np.corrcoef(confidences, accuracies)[0,1]
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'avg_confidence': float(avg_confidence),
        'avg_accuracy': float(avg_accuracy),
        'conf_acc_correlation': float(conf_acc_corr),
        'bin_metrics': bin_metrics
    }

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Compute metrics for each class separately.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary mapping class index to metrics dictionary
    """
    n_classes = y_prob.shape[1]
    metrics = {}
    
    for c in range(n_classes):
        # Convert to binary problem
        y_true_binary = (y_true == c).astype(int)
        y_prob_binary = y_prob[:, c]
        
        # Compute class-specific metrics
        class_metrics = {}
        
        # Accuracy for samples where this class has highest probability
        pred_this_class = (y_prob.argmax(axis=1) == c)
        if pred_this_class.sum() > 0:
            class_acc = np.mean(y_true[pred_this_class] == c)
            class_metrics['accuracy'] = float(class_acc)
        else:
            class_metrics['accuracy'] = 0.0
        
        # Mean confidence when predicting this class
        class_metrics['mean_confidence'] = float(y_prob_binary[pred_this_class].mean()) \
            if pred_this_class.sum() > 0 else 0.0
        
        # Class support
        class_metrics['support'] = int((y_true == c).sum())
        
        metrics[c] = class_metrics
    
    return metrics
