"""Training utilities for deep learning models."""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeometricDataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import trange
from ..models.gnn import DeepGAT
from .evaluation import evaluate

def train_one_epoch(
    model: DeepGAT,
    loader: GeometricDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1
) -> float:
    """Train model for one epoch.
    
    Args:
        model: DeepGAT model
        loader: Data loader
        optimizer: Optimizer
        device: Device to train on
        label_smoothing: Label smoothing factor
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        x_global = getattr(batch_data, 'x_global', None)
        
        out = model(batch_data.x, batch_data.edge_index, batch_data.batch, x_global)
        loss = F.cross_entropy(out, batch_data.y.view(-1), label_smoothing=label_smoothing)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item() * batch_data.num_graphs
    
    return total_loss / len(loader.dataset)

def train_with_early_stopping(
    model: DeepGAT,
    train_loader: GeometricDataLoader,
    val_loader: GeometricDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    label_smoothing: float = 0.1,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """Train model with early stopping.
    
    Args:
        model: DeepGAT model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        epochs: Maximum number of epochs
        patience: Early stopping patience
        label_smoothing: Label smoothing factor
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (train_losses, val_losses)
    """
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    epoch_iter = trange(epochs, desc="Training") if verbose else range(epochs)
    
    for epoch in epoch_iter:
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=label_smoothing
        )
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_graphs = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_global = getattr(data, 'x_global', None)
                logits = model(data.x, data.edge_index, data.batch, x_global)
                loss = F.cross_entropy(logits, data.y.view(-1))
                total_val_loss += loss.item() * data.num_graphs
                total_graphs += data.num_graphs
        
        val_loss = total_val_loss / total_graphs
        val_losses.append(val_loss)
        
        # Update progress bar if verbose
        if verbose:
            epoch_iter.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}"
            })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_losses, val_losses

def cross_validate_gnn(
    data_list: List,
    n_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = 42,
    hidden_channels: int = 64,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cpu',
    dropout: float = 0.2,
    heads: int = 2,
    label_smoothing: float = 0.1,
    patience: int = 10,
    batch_size: int = 8,
    adjacency_name: str = ""
) -> List[float]:
    """Perform cross-validation for GNN model.
    
    Args:
        data_list: List of graph data objects
        n_splits: Number of CV folds
        n_repeats: Number of CV repeats
        random_state: Random seed
        hidden_channels: Hidden dimension size
        epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        dropout: Dropout probability
        heads: Number of attention heads
        label_smoothing: Label smoothing factor
        patience: Early stopping patience
        batch_size: Batch size
        adjacency_name: Name for progress bar
        
    Returns:
        List of per-fold accuracies
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    labels = np.array([d.y.item() for d in data_list])
    n_classes = len(np.unique(labels))
    
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    all_acc = []
    
    outer_pbar = tqdm(
        cv.split(data_list, labels),
        total=n_splits*n_repeats,
        desc=f"CV_GAT({adjacency_name})"
    )
    fold_idx = 0
    
    for train_idx, valid_idx in outer_pbar:
        fold_idx += 1
        
        train_data = [data_list[i] for i in train_idx]
        valid_data = [data_list[i] for i in valid_idx]
        
        train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = GeometricDataLoader(valid_data, batch_size=batch_size, shuffle=False)
        
        in_channels = train_data[0].num_node_features
        global_dim = getattr(train_data[0], 'x_global', torch.zeros(1,0)).shape[-1]
        
        model = DeepGAT(
            in_channels, hidden_channels, n_classes,
            global_features_dim=global_dim,
            dropout=dropout,
            heads=heads
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Train with early stopping
        train_losses, val_losses = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            patience=patience,
            label_smoothing=label_smoothing,
            verbose=False  # Disable inner progress bar
        )
        
        # Evaluate
        metrics, _, _, _ = evaluate(model, valid_loader, device, n_classes=n_classes)
        all_acc.append(metrics['accuracy'])
        
        # Update progress bar
        outer_pbar.set_postfix({'curr_acc': f"{metrics['accuracy']:.3f}"})
    
    return all_acc
