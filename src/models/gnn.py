"""Graph Neural Network model module."""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool

class DeepGAT(nn.Module):
    """3-layer GAT with optional global features.
    
    Features:
        - Hidden dimension customization
        - Multiple attention heads
        - Dropout regularization
        - Optional global (TDA) features appended before final FC layer
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        global_features_dim: int = 0,
        heads: int = 2,
        dropout: float = 0.1
    ):
        """Initialize DeepGAT.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (num classes)
            global_features_dim: Dimension of global features
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout)
        
        self.global_features_dim = global_features_dim
        
        # After the final GAT layer, output dimension = hidden_channels * heads
        in_lin1_dim = hidden_channels * heads
        
        # Combine local + global features
        self.lin1 = nn.Linear(in_lin1_dim + global_features_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        x_global: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
            x_global: Optional global features
            
        Returns:
            Output logits
        """
        # 1) GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat3(x, edge_index))
        
        # 2) Pool
        x = global_mean_pool(x, batch)  # => (batch_size, hidden_channels*heads)
        
        # 3) Append TDA features if available
        if x_global is not None:
            x = torch.cat([x, x_global], dim=-1)
        
        # 4) FC layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
    def get_penultimate_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        x_global: torch.Tensor = None
    ) -> torch.Tensor:
        """Get embeddings from the layer before output.
        
        Useful for PCA visualization and other downstream tasks.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
            x_global: Optional global features
            
        Returns:
            Penultimate layer embeddings
        """
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat3(x, edge_index))
        
        x = global_mean_pool(x, batch)
        if x_global is not None:
            x = torch.cat([x, x_global], dim=-1)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        return x  # shape: (batch_size, hidden_channels)
    
    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        x_global: torch.Tensor = None,
        n_passes: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monte Carlo dropout inference for uncertainty estimation.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
            x_global: Optional global features
            n_passes: Number of MC dropout passes
            
        Returns:
            Tuple of (mean_probabilities, std_probabilities)
        """
        self.train()  # Enable dropout
        outs = []
        for _ in range(n_passes):
            out = F.softmax(self.forward(x, edge_index, batch, x_global), dim=-1)
            outs.append(out.detach().cpu().numpy())
        self.eval()
        
        outs = np.array(outs)
        mean_probs = outs.mean(axis=0)
        std_probs = outs.std(axis=0)
        return mean_probs, std_probs
