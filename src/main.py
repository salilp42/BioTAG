"""Main script for running time series classification experiments."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

from data.dataset import TimeSeriesDataset
from data.preprocessing import (
    downsample_time_series,
    multi_tda_features,
    create_time_series_graphs
)
from models.baseline import train_svm_cv
from models.gnn import DeepGAT
from training.trainer import (
    train_with_early_stopping,
    cross_validate_gnn
)
from training.evaluation import (
    evaluate,
    brier_score_analysis,
    statistical_significance_test
)
from visualization.plots import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curves,
    plot_latent_space,
    plot_persistence_analysis
)
from utils.metrics import (
    calibration_analysis,
    compute_confidence_metrics,
    compute_per_class_metrics
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time Series Classification with GNNs')
    
    # Dataset parameters
    parser.add_argument('--datasets', nargs='+', default=[
        'NonInvasiveFetalECGThorax1',
        'EOGHorizontalSignal',
        'EOGVerticalSignal'
    ], help='List of dataset names to process')
    parser.add_argument('--downsample-size', type=int, default=200,
                       help='Target sequence length after downsampling')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension in GAT layers')
    parser.add_argument('--heads', type=int, default=2,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--adj-window', type=int, default=3,
                       help='Window size for creating graph edges')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Cross-validation parameters
    parser.add_argument('--n-splits', type=int, default=3,
                       help='Number of CV folds')
    parser.add_argument('--n-repeats', type=int, default=2,
                       help='Number of CV repeats')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def setup_output_dirs(base_dir: str) -> Dict[str, str]:
    """Create output directories for results and figures.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        'base': base_dir,
        'figures': os.path.join(base_dir, 'figures'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results')
    }
    
    # Create subdirectories for different plot types
    figure_subdirs = ['confusion', 'roc', 'learning_curves', 'latent', 'persistence']
    for subdir in figure_subdirs:
        dirs[f'figures_{subdir}'] = os.path.join(dirs['figures'], subdir)
    
    # Create all directories
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    return dirs

def run_experiment(args):
    """Run the full experimental pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup directories
    dirs = setup_output_dirs(args.output_dir)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Store all results here
    final_results = {
        "CrossVal_Results": {},
        "Holdout_Results": {},
        "Ablation_Overall": {},
        "SignificanceTests": {},
        "Parameters": vars(args)
    }
    
    # For significance testing
    ablation_results = {
        "SVM_TDA": [],
        "GAT_TDA": [],
        "GAT_noTDA": []
    }
    
    # Store CV means for "GAT+TDA"
    all_dataset_cv = {}
    
    for ds_name in args.datasets:
        print(f"\n\n========== DATASET: {ds_name} ==========")
        
        # 1. Load and preprocess data
        loader = TimeSeriesDataset(ds_name, test_size=0.2, random_state=args.seed)
        (X_train, y_train), (X_test, y_test) = loader.load_and_preprocess()
        
        if X_train is None:
            print(f"Skipping {ds_name}, load error.")
            continue
        
        # 2. Downsample if needed
        X_train = downsample_time_series(X_train, args.downsample_size)
        X_test = downsample_time_series(X_test, args.downsample_size)
        
        # Combine for cross-validation
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        # 3. Compute TDA features
        print("Computing TDA features...")
        tda_all = multi_tda_features(X_all, homology_dims=[0,1,2])
        
        # 4. SVM baseline with TDA features
        print(f"\n--- Cross-Val SVM baseline (TDA) on {ds_name} ---")
        svm_acc_list = train_svm_cv(
            tda_all, y_all,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed
        )
        ablation_results["SVM_TDA"].extend(svm_acc_list)
        
        # 5. GAT with TDA
        print(f"\n--- Cross-Val GAT with TDA on {ds_name} ---")
        data_list_tda = create_time_series_graphs(
            X_all, y_all,
            tda_features=tda_all,
            adjacency_window=args.adj_window
        )
        gat_tda_acc_list = cross_validate_gnn(
            data_list_tda,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed,
            hidden_channels=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            dropout=args.dropout,
            heads=args.heads,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            batch_size=args.batch_size,
            adjacency_name="TDA"
        )
        ablation_results["GAT_TDA"].extend(gat_tda_acc_list)
        
        # 6. GAT without TDA
        print(f"\n--- Cross-Val GAT no TDA on {ds_name} ---")
        data_list_no_tda = create_time_series_graphs(
            X_all, y_all,
            tda_features=None,
            adjacency_window=args.adj_window
        )
        gat_no_tda_acc_list = cross_validate_gnn(
            data_list_no_tda,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.seed,
            hidden_channels=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            dropout=args.dropout,
            heads=args.heads,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            batch_size=args.batch_size,
            adjacency_name="noTDA"
        )
        ablation_results["GAT_noTDA"].extend(gat_no_tda_acc_list)
        
        # Store cross-validation results
        final_results["CrossVal_Results"][ds_name] = {
            "SVM_TDA": {
                "acc_mean": float(np.mean(svm_acc_list)),
                "acc_std": float(np.std(svm_acc_list)),
                "all_folds": list(map(float, svm_acc_list))
            },
            "GAT_TDA": {
                "acc_mean": float(np.mean(gat_tda_acc_list)),
                "acc_std": float(np.std(gat_tda_acc_list)),
                "all_folds": list(map(float, gat_tda_acc_list))
            },
            "GAT_noTDA": {
                "acc_mean": float(np.mean(gat_no_tda_acc_list)),
                "acc_std": float(np.std(gat_no_tda_acc_list)),
                "all_folds": list(map(float, gat_no_tda_acc_list))
            }
        }
        
        # Store CV means for summary
        all_dataset_cv[ds_name] = {
            'acc_mean': float(np.mean(gat_tda_acc_list)),
            'acc_std': float(np.std(gat_tda_acc_list))
        }
        
        # 7. Final holdout demonstration with GAT+TDA
        print(f"\n--- Final holdout evaluation on {ds_name} ---")
        
        # Create data loaders
        from torch_geometric.loader import DataLoader as GeometricDataLoader
        train_list = create_time_series_graphs(X_train, y_train,
                                             multi_tda_features(X_train),
                                             args.adj_window)
        test_list = create_time_series_graphs(X_test, y_test,
                                            multi_tda_features(X_test),
                                            args.adj_window)
        
        train_loader = GeometricDataLoader(train_list, batch_size=args.batch_size,
                                         shuffle=True)
        test_loader = GeometricDataLoader(test_list, batch_size=args.batch_size,
                                        shuffle=False)
        
        # Create and train model
        n_classes = len(np.unique(y_all))
        in_channels = train_list[0].num_node_features
        global_dim = getattr(train_list[0], 'x_global',
                           torch.zeros(1,0)).shape[-1]
        
        model = DeepGAT(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=n_classes,
            global_features_dim=global_dim,
            heads=args.heads,
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # Train with early stopping
        train_losses, val_losses = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            label_smoothing=args.label_smoothing
        )
        
        # Evaluate
        metrics, y_true, y_pred, y_prob = evaluate(
            model, test_loader, device, n_classes=n_classes
        )
        
        # Additional metrics
        brier = brier_score_analysis(y_true, y_prob)
        conf_metrics = compute_confidence_metrics(y_true, y_prob)
        per_class_metrics = compute_per_class_metrics(y_true, y_prob)
        
        # Store results
        holdout_results = {
            "metrics": metrics,
            "brier_score": float(brier),
            "confidence_metrics": conf_metrics,
            "per_class_metrics": per_class_metrics
        }
        
        final_results["Holdout_Results"][ds_name] = holdout_results
        
        # 8. Generate plots
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            dataset_name=ds_name,
            save_path=os.path.join(dirs['figures_confusion'],
                                 f"{ds_name}_confusion.png")
        )
        
        # Learning curves
        plot_learning_curves(
            train_losses, val_losses,
            dataset_name=ds_name,
            save_path=os.path.join(dirs['figures_learning_curves'],
                                 f"{ds_name}_learning.png")
        )
        
        # Latent space
        plot_latent_space(
            model, test_loader, device,
            ds_name=ds_name,
            save_path=os.path.join(dirs['figures_latent'],
                                 f"{ds_name}_latent.png")
        )
    
    # 9. Compute significance tests
    sig_tests = statistical_significance_test(ablation_results)
    final_results["SignificanceTests"] = sig_tests
    
    # 10. Save final results
    results_path = os.path.join(dirs['results'], 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nExperiment complete! Results saved to:", results_path)
    
    # Print summary
    print("\nSummary of results across datasets:")
    for ds_name, results in all_dataset_cv.items():
        print(f"{ds_name}: {results['acc_mean']:.3f} ± {results['acc_std']:.3f}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
