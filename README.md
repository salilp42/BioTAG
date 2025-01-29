# Time Series Classification with Graph Neural Networks

This repository contains an implementation of time series classification using Graph Neural Networks (GNNs) enhanced with topological features. The approach combines Graph Attention Networks (GAT) with Topological Data Analysis (TDA) to achieve robust classification performance.

## Features

- Time series to graph conversion with flexible adjacency window
- Topological feature extraction using persistent homology
- Graph Attention Network (GAT) with configurable architecture
- Support for both binary and multi-class classification
- Comprehensive evaluation metrics and visualization tools
- Cross-validation and statistical significance testing
- Baseline SVM model for comparison

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd time_series_gnn
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── data/
│   │   ├── dataset.py        # Time series dataset loading
│   │   └── preprocessing.py  # Data preprocessing and TDA
│   ├── models/
│   │   ├── gnn.py           # GAT model implementation
│   │   └── baseline.py      # SVM baseline
│   ├── training/
│   │   ├── trainer.py       # Training utilities
│   │   └── evaluation.py    # Evaluation metrics
│   ├── visualization/
│   │   └── plots.py         # Visualization tools
│   └── utils/
│       └── metrics.py       # Additional metrics
└── tests/                   # Unit tests
```

## Usage

### Basic Usage

Run an experiment with default parameters:
```bash
python src/main.py
```

### Custom Configuration

Specify custom parameters:
```bash
python src/main.py \
    --datasets NonInvasiveFetalECGThorax1 EOGHorizontalSignal \
    --downsample-size 200 \
    --hidden-dim 128 \
    --heads 2 \
    --dropout 0.1 \
    --adj-window 3 \
    --epochs 200 \
    --batch-size 8 \
    --lr 1e-4 \
    --label-smoothing 0.1 \
    --patience 10 \
    --n-splits 3 \
    --n-repeats 2 \
    --output-dir results \
    --seed 42
```

### Available Datasets

The following datasets are recommended for use:
- Biomedical: NonInvasiveFetalECGThorax1
- Brain Activity: EOGHorizontalSignal, EOGVerticalSignal
- Other: LSST

## Model Architecture

The implemented Graph Attention Network (GAT) consists of:
- 3 GAT layers with configurable number of attention heads
- Global pooling for graph-level representations
- Optional global features from topological analysis
- Dropout regularization
- Label smoothing during training

## Evaluation Metrics

The framework provides comprehensive evaluation including:
- Accuracy, Precision, Recall, F1-score
- ROC curves and AUC
- Confusion matrices
- Calibration analysis
- Confidence metrics
- Per-class performance analysis
- Statistical significance tests

## Visualization

Various visualization tools are provided:
- Learning curves
- ROC curves
- Confusion matrices
- Latent space visualization (PCA)
- Persistence diagrams
- Attention flow analysis

## Results

Results are saved in the specified output directory with the following structure:
```
results/
├── figures/
│   ├── confusion/      # Confusion matrices
│   ├── roc/            # ROC curves
│   ├── learning_curves/# Training progress
│   ├── latent/        # Latent space visualizations
│   └── persistence/   # Persistence diagrams
├── models/            # Saved model checkpoints
└── results/          # Detailed results in JSON format
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{time_series_gnn,
  title = {Time Series Classification using Graph Neural Networks with Topological Features},
  author = {Salil Patel},
```
