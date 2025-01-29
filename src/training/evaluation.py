"""Model evaluation utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int = 2
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to evaluate on
        n_classes: Number of classes
        
    Returns:
        Tuple of (metrics_dict, true_labels, predictions, probabilities)
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x_global = getattr(data, 'x_global', None)
            logits = model(data.x, data.edge_index, data.batch, x_global)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            y_true.append(data.y.view(-1).cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    if n_classes == 2:
        auc_val = roc_auc_score(y_true, y_prob[:,1])
    else:
        auc_val = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc_val
    }
    
    return metrics, y_true, y_pred, y_prob

def brier_score_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score for multi-class prediction.
    
    For multi-class, we sum MSE across one-hot columns, then divide by (N*C).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score
    """
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    
    # Convert to one-hot
    one_hot_true = np.zeros_like(y_prob)
    one_hot_true[np.arange(n_samples), y_true] = 1.0
    
    mse = (y_prob - one_hot_true)**2
    brier = mse.sum() / (n_samples * n_classes)
    return brier

def confidence_vs_correctness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analyze how confidence correlates with correctness.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of confidence bins
        
    Returns:
        Tuple of (bin_edges, empirical_accuracy, mean_confidence, bin_sizes)
    """
    max_prob = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correctness = (preds == y_true).astype(int)
    
    # Bin by confidence
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(max_prob, bins) - 1
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.sum(mask) == 0:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_sizes.append(0)
        else:
            bin_accs.append(correctness[mask].mean())
            bin_confs.append(max_prob[mask].mean())
            bin_sizes.append(np.sum(mask))
    
    return bins, bin_accs, bin_confs, bin_sizes

def statistical_significance_test(results_dict: Dict[str, list]) -> Dict[str, Dict]:
    """Perform Wilcoxon signed-rank test across different models.
    
    Args:
        results_dict: Dictionary of {model_name: [list_of_accuracy], ...}
        
    Returns:
        Dictionary of pairwise test results
    """
    from scipy.stats import wilcoxon
    
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    
    sig_test_results = {}
    
    if n_models < 2:
        print("Not enough models for significance test.")
        return {}
    
    print("\n=== Statistical Significance Tests (Wilcoxon) ===")
    for i in range(n_models):
        for j in range(i+1, n_models):
            name_i, name_j = model_names[i], model_names[j]
            acc_i = np.array(results_dict[name_i])
            acc_j = np.array(results_dict[name_j])
            
            if (len(acc_i) == len(acc_j)) and (len(acc_i) > 1):
                try:
                    stat, pval = wilcoxon(acc_i, acc_j)
                    print(f"{name_i} vs {name_j}: stat={stat:.3f}, p={pval:.4f}")
                    sig_test_results[f"{name_i}_vs_{name_j}"] = {
                        "stat": float(stat),
                        "pval": float(pval)
                    }
                except Exception as e:
                    print(f"Error in significance test {name_i} vs {name_j}: {str(e)}")
                    sig_test_results[f"{name_i}_vs_{name_j}"] = str(e)
            else:
                msg = f"Cannot compare {name_i} vs {name_j}, unequal or insufficient data."
                print(msg)
                sig_test_results[f"{name_i}_vs_{name_j}"] = msg
    
    return sig_test_results
