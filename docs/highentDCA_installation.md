# Installation Guide

This guide provides detailed instructions for installing highentDCA on your system.

## Prerequisites

Before installing highentDCA, ensure you have the following prerequisites:

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: Version 3.10 or higher
- **GPU** (recommended): NVIDIA GPU with CUDA support for optimal performance
- **Memory**: At least 8GB RAM (16GB+ recommended for large datasets)

### Python Environment

We strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
# Using conda (recommended)
conda create -n highentdca python=3.10
conda activate highentdca

# Or using venv
python -m venv highentdca_env
source highentdca_env/bin/activate  # On Windows: highentdca_env\Scripts\activate
```

## Installation Methods

### Method 1: Install from Source (Recommended)

This is currently the only installation method since the package is not yet on PyPI.

1. **Clone the repository**:

```bash
git clone https://github.com/robertonetti/highentropyDCA.git
cd highentropyDCA
```

2. **Install the package**:

```bash
pip install .
```

This will automatically install all required dependencies.

3. **Verify installation**:

```bash
highentDCA --help
```

You should see the help message with available commands.

### Method 2: Development Installation

If you plan to modify the code or contribute to the project:

```bash
git clone https://github.com/robertonetti/highentropyDCA.git
cd highentropyDCA
pip install -e .
```

The `-e` flag installs the package in "editable" mode, so changes to the source code are immediately reflected without reinstalling.

## Dependencies

highentDCA requires the following packages, which are installed automatically:

### Core Dependencies

- **adabmDCA** (== 0.5.0): Base DCA framework with core algorithms
- **PyTorch** (>= 2.1.0): Deep learning framework for GPU acceleration
- **NumPy** (>= 1.26.4): Numerical computing library
- **Pandas** (>= 2.2.2): Data manipulation and analysis

### Additional Dependencies

- **Matplotlib** (>= 3.8.0): Plotting and visualization
- **tqdm** (>= 4.66.6): Progress bars for long-running operations
- **BioPython** (>= 1.85): Biological sequence file handling
- **wandb** (>= 0.12.0): Experiment tracking (optional, for `--wandb` flag)




## Next Steps

Now that you have highentDCA installed:

1. **[Quick Start](usage.md#quick-start)**: Train your first model
2. **[Usage Guide](usage.md)**: Explore all features and options
3. **[API Reference](api/README.md)**: Use highentDCA in Python scripts
4. **[Examples](usage.md#examples)**: Learn from practical examples

## Getting Help

If you encounter issues not covered here:

- Check the [GitHub Issues](https://github.com/robertonetti/highentropyDCA/issues)
- Review the [adabmDCA documentation](https://spqb.github.io/adabmDCApy/)
- Contact: robertonetti3@gmail.com
