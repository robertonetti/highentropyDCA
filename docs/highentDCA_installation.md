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

## GPU Support

### CUDA Installation

For optimal performance, install PyTorch with CUDA support:

1. **Check CUDA version**:

```bash
nvidia-smi
```

Look for the CUDA version in the output (e.g., CUDA 11.8, 12.1).

2. **Install PyTorch with CUDA**:

Visit [PyTorch's installation page](https://pytorch.org/get-started/locally/) and select your configuration. For example:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Verify GPU support**:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

### CPU-Only Installation

If you don't have a GPU or prefer CPU-only execution:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Note: Training will be significantly slower on CPU.

## Installing adabmDCA

highentDCA depends on adabmDCA version 0.5.0. This will be installed automatically, but you can also install it manually:

### From PyPI

```bash
pip install adabmDCA==0.5.0
```

### From Source

```bash
git clone https://github.com/spqb/adabmDCApy.git
cd adabmDCApy
git checkout v0.5.0  # Ensure correct version
pip install .
```

## Troubleshooting

### Common Issues

#### Issue: `command not found: highentDCA`

**Solution**: Ensure the installation directory is in your PATH:

```bash
# Check if pip bin directory is in PATH
pip show highentDCA | grep Location
```

Add to PATH if needed:

```bash
export PATH="$HOME/.local/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

#### Issue: CUDA out of memory errors

**Solutions**:
- Reduce the number of chains: `--nchains 5000` (instead of default 10000)
- Use smaller batch sizes for sampling
- Monitor GPU memory: `nvidia-smi -l 1`

#### Issue: ImportError for adabmDCA modules

**Solution**: Ensure adabmDCA is correctly installed:

```bash
python -c "import adabmDCA; print(adabmDCA.__version__)"
```

Should output: `0.5.0`

If not, reinstall:

```bash
pip uninstall adabmDCA
pip install adabmDCA==0.5.0
```

#### Issue: Weights & Biases login required

**Solution**: Initialize wandb (only needed if using `--wandb` flag):

```bash
wandb login
```

Enter your API key from [wandb.ai](https://wandb.ai).

### Platform-Specific Notes

#### macOS

On macOS with Apple Silicon (M1/M2/M3):

```bash
# Use conda for better compatibility
conda create -n highentdca python=3.10
conda activate highentdca
conda install pytorch::pytorch -c pytorch
pip install adabmDCA==0.5.0
pip install .
```

Note: MPS (Metal Performance Shaders) acceleration is supported but may not be as optimized as CUDA.

#### Windows

On Windows, use WSL2 (Windows Subsystem for Linux) for the best experience:

1. Install WSL2 with Ubuntu
2. Follow the Linux installation instructions inside WSL2
3. For GPU support, install [CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

#### Linux (Docker)

For a containerized environment:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/robertonetti/highentropyDCA.git && \
    cd highentropyDCA && \
    pip3 install .

CMD ["bash"]
```

Build and run:

```bash
docker build -t highentdca .
docker run --gpus all -it highentdca
```

## Verifying Installation

After installation, verify everything works:

### 1. Check CLI Access

```bash
highentDCA train --help
```

Should display the training command help.

### 2. Test Python Import

```python
import highentDCA
from highentDCA.models.edDCA import fit
from highentDCA.checkpoint import Checkpoint
print("highentDCA successfully imported!")
```

### 3. Run a Quick Test

```bash
# Download test data
cd highentropyDCA/example_data

# Train a small model
highentDCA train \
    --data TEST/chains.fasta \
    --output test_output \
    --model edDCA \
    --density 0.98 \
    --drate 0.01 \
    --nchains 1000 \
    --nepochs 100 \
    --seed 42
```

This should complete without errors and create output in `test_output/`.

## Updating highentDCA

To update to the latest version:

```bash
cd highentropyDCA
git pull origin main  # or develop
pip install --upgrade .
```

For development installations (`pip install -e .`), just pull the latest code:

```bash
git pull origin main
```

## Uninstalling

To remove highentDCA:

```bash
pip uninstall highentDCA
```

To also remove dependencies:

```bash
pip uninstall highentDCA adabmDCA torch numpy pandas matplotlib tqdm biopython wandb
```

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
