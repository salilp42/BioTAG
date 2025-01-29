"""Visualization utilities for model analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
import persim
import torch

def plot_averaged_timeseries_with_attention(
    X: np.ndarray,
    attention_weights: np.ndarray,
    dataset_name: str,
    save_path: str
) -> None:
    """Plot mean ± std of time-series plus color-shaded region by attention.
    
    For multi-feature data, we plot the average across features.
    
    Args:
        X: Time series data
        attention_weights: Attention weights
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    if X.shape[-1] != 1:
        X_plot = X.mean(axis=2)  # (N,T)
    else:
        X_plot = X[...,0]  # (N,T)
    
    mean_series = X_plot.mean(axis=0)
    std_series = X_plot.std(axis=0)
    
    avg_attention = attention_weights.mean(axis=0)  # shape (T,)
    
    t = np.arange(len(mean_series))
    
    fig, ax = plt.subplots(figsize=(8,4), dpi=600)
    # Plot mean ± std
    ax.plot(t, mean_series, label='Mean Time-Series', color='blue')
    ax.fill_between(t, mean_series-std_series, mean_series+std_series,
                   alpha=0.3, color='blue')
    
    # Color the baseline region by attention
    cmap = plt.get_cmap('Reds')
    norm_attn = (avg_attention - avg_attention.min()) / (avg_attention.ptp() + 1e-9)
    for i in range(len(t)-1):
        c = cmap(norm_attn[i])
        ax.fill_between([t[i], t[i+1]], [0,0], [mean_series[i], mean_series[i+1]],
                       color=c, alpha=0.4)
    
    ax.set_title(f"{dataset_name} - Mean ± Std Time-Series with Attention")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='best')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_layer_attention_flow(
    model: torch.nn.Module,
    data_sample: torch.Tensor,
    dataset_name: str,
    save_path: str
) -> None:
    """Visualize multi-layer attention.
    
    Extract alpha from each GAT layer and show them side-by-side as histograms.
    
    Args:
        model: GAT model
        data_sample: Sample data point
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    model.eval()
    with torch.no_grad():
        x, edge_index = data_sample.x, data_sample.edge_index
        
        x1, (ei1, alpha1) = model.gat1(x, edge_index, return_attention_weights=True)
        x1 = torch.nn.functional.elu(x1)
        x2, (ei2, alpha2) = model.gat2(x1, ei1, return_attention_weights=True)
        x2 = torch.nn.functional.elu(x2)
        x3, (ei3, alpha3) = model.gat3(x2, ei2, return_attention_weights=True)
        x3 = torch.nn.functional.elu(x3)
        
        # average over heads
        alpha_list = [
            alpha1.mean(dim=-1).cpu().numpy(),
            alpha2.mean(dim=-1).cpu().numpy(),
            alpha3.mean(dim=-1).cpu().numpy()
        ]
    
    fig, axes = plt.subplots(1,3, figsize=(9,3), dpi=600)
    layer_names = ["Layer1", "Layer2", "Layer3"]
    
    for i, (ax, a) in enumerate(zip(axes, alpha_list)):
        ax.hist(a, bins=50, color='gray')
        ax.set_title(f"{dataset_name} - {layer_names[i]}")
        ax.set_xlabel("Attention Weight")
        ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_persistence_analysis(diagrams: np.ndarray, dataset_name: str, save_path: str) -> None:
    """Plot a single persistence diagram.
    
    Args:
        diagrams: Persistence diagrams
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    plt.figure(figsize=(6, 6), dpi=600)
    persim.plot_diagrams(diagrams[0], show=False)
    plt.title(f'{dataset_name} - Persistence Diagram')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(
    y_true_clean: np.ndarray,
    y_prob_clean: np.ndarray,
    y_true_noisy: np.ndarray,
    y_prob_noisy: np.ndarray,
    n_classes: int,
    dataset_name: str,
    save_path: str
) -> None:
    """Plot ROC curves for clean vs. noisy data.
    
    If multi-class, do macro-average.
    
    Args:
        y_true_clean: True labels for clean data
        y_prob_clean: Predicted probabilities for clean data
        y_true_noisy: True labels for noisy data
        y_prob_noisy: Predicted probabilities for noisy data
        n_classes: Number of classes
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    def compute_roc(y_true, y_prob):
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
            return fpr, tpr
        else:
            # multi-class: compute macro-average
            fprs = []
            classes = np.unique(y_true)
            linspace = np.linspace(0,1,100)
            for c in classes:
                y_bin = (y_true==c).astype(int)
                fpr_c, tpr_c, _ = roc_curve(y_bin, y_prob[:,c])
                interp_tpr = np.interp(linspace, fpr_c, tpr_c)
                fprs.append(interp_tpr)
            mean_tpr = np.mean(fprs, axis=0)
            return linspace, mean_tpr
    
    fpr_c, tpr_c = compute_roc(y_true_clean, y_prob_clean)
    fpr_n, tpr_n = compute_roc(y_true_noisy, y_prob_noisy)
    
    plt.figure(figsize=(5,4), dpi=600)
    plt.plot(fpr_c, tpr_c, label='Clean')
    plt.plot(fpr_n, tpr_n, label='Noisy')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title(f"{dataset_name} - ROC Curve (Clean vs. Noisy)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_latent_space(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ds_name: str,
    save_path: str
) -> None:
    """Plot 2D PCA of penultimate layer embeddings.
    
    Args:
        model: Model to extract embeddings from
        loader: Data loader
        device: Device to run model on
        ds_name: Name of dataset
        save_path: Path to save figure
    """
    model.eval()
    all_embs = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x_global = getattr(data, 'x_global', None)
            embeddings = model.get_penultimate_embeddings(
                data.x, data.edge_index, data.batch, x_global
            )
            all_embs.append(embeddings.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if all_embs.shape[0] < 2:
        # Not enough samples to do PCA
        return
    
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(all_embs)
    
    plt.figure(figsize=(5,4), dpi=600)
    scatter = plt.scatter(embs_2d[:,0], embs_2d[:,1],
                         c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f'{ds_name} - PCA of Penultimate Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    save_path: str
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4), dpi=600)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{dataset_name} - Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(
    train_losses: list,
    val_losses: list,
    dataset_name: str,
    save_path: str
) -> None:
    """Plot training and validation learning curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    plt.figure(figsize=(6,4), dpi=600)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"{dataset_name} - Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
