"""Setup configuration for the time_series_gnn package."""

from setuptools import setup, find_packages

setup(
    name="time_series_gnn",
    version="0.1.0",
    description="Time Series Classification using Graph Neural Networks with Topological Features",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "aeon>=0.5.0",
        "giotto-tda>=0.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.65.0",
        "pandas>=1.5.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "persim>=0.3.1",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "typing-extensions>=4.5.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
