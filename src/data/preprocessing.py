"""Time series preprocessing and feature extraction module."""

import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy
from torch_geometric.data import Data
import torch

def downsample_time_series(X: np.ndarray, target_length: int = 200) -> Optional[np.ndarray]:
    """Simple downsampling to reduce sequence length if T > target_length.
    
    Args:
        X: Input time series array of shape (N, T, F)
        target_length: Target sequence length
        
    Returns:
        Downsampled array if T > target_length, otherwise original array
    """
    if X is None:
        return None
    
    N, T, F = X.shape
    if T <= target_length:
        return X
    
    idxs = np.linspace(0, T-1, target_length).astype(int)
    return X[:, idxs, :]

def multi_tda_features(X: np.ndarray, homology_dims: List[int] = [0,1,2]) -> Optional[np.ndarray]:
    """Compute multiple TDA representations for each sample.
    
    Features computed:
        - Bottleneck amplitude (H0, H1, H2)
        - Wasserstein amplitude (H0, H1, H2)
        - Persistence Entropy (H0, H1, H2)
    => total 9 features if homology_dims=[0,1,2]
    
    Args:
        X: Input time series array
        homology_dims: List of homology dimensions to compute
        
    Returns:
        Array of TDA features
    """
    if X is None:
        return None
    
    vr = VietorisRipsPersistence(
        homology_dimensions=homology_dims,
        metric='euclidean',
        n_jobs=-1
    )
    diagrams = vr.fit_transform(X)
    
    # Compute amplitude (bottleneck, wasserstein) + entropy
    amp_bot = Amplitude(metric='bottleneck').fit_transform(diagrams)
    amp_wass = Amplitude(metric='wasserstein').fit_transform(diagrams)
    pe = PersistenceEntropy().fit_transform(diagrams)
    
    feats = np.hstack([amp_bot, amp_wass, pe])
    
    # Scale TDA features
    scaler_tda = StandardScaler()
    feats_scaled = scaler_tda.fit_transform(feats)
    return feats_scaled

def create_time_series_graphs(
    X: np.ndarray,
    y: np.ndarray,
    tda_features: Optional[np.ndarray] = None,
    adjacency_window: int = 5
) -> List[Data]:
    """Create graph representation of time series data.
    
    Args:
        X: Input time series array
        y: Target labels
        tda_features: Optional TDA features to include
        adjacency_window: Window size for creating edges
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    data_list = []
    N, T, F = X.shape
    
    for i in range(N):
        # Node features
        node_x = torch.tensor(X[i], dtype=torch.float)
        
        # Create edges
        edges_src, edges_dst = [], []
        for t in range(T):
            for w in range(1, adjacency_window+1):
                if t+w < T:
                    edges_src.append(t)
                    edges_dst.append(t+w)
        edges_src = np.array(edges_src)
        edges_dst = np.array(edges_dst)
        
        # Bidirectional edges
        edges_bidir = np.hstack([
            np.vstack([edges_src, edges_dst]),
            np.vstack([edges_dst, edges_src])
        ])
        edge_index = torch.tensor(edges_bidir, dtype=torch.long)
        
        # Create graph
        label = torch.tensor([y[i]], dtype=torch.long)
        graph_data = Data(x=node_x, edge_index=edge_index, y=label)
        
        # Add TDA features if provided
        if tda_features is not None:
            feat = torch.tensor(tda_features[i], dtype=torch.float).unsqueeze(0)
            graph_data.x_global = feat
        
        data_list.append(graph_data)
    
    return data_list
