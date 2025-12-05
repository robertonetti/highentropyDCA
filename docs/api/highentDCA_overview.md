# API Reference

Welcome to the highentDCA Python API documentation. This section provides detailed information about the modules, classes, and functions available in the highentDCA package.

## Overview

highentDCA extends the `adabmDCA` framework with specialized functionality for entropy-decimated DCA models. The package is organized into several modules:

- **[Checkpoint](checkpoint.md)**: Checkpoint strategies for saving model state
- **[Training](training.md)**: Training functions for graph-based DCA models
- **[edDCA Model](models.edDCA.md)**: Entropy decimation algorithm implementation
- **[CLI](cli.md)**: Command-line interface entry point
- **[Parser](parser.md)**: Argument parsing utilities
- **[Entropy Computation](entropy.md)**: Thermodynamic integration for entropy calculation

## Quick Links

### Core Modules

| Module | Description |
|--------|-------------|
| `highentDCA.models.edDCA` | Entropy decimation fitting algorithm |
| `highentDCA.training` | Graph training utilities |
| `highentDCA.checkpoint` | Checkpoint management classes |
| `highentDCA.parser` | CLI argument parsers |
| `highentDCA.scripts.entropy` | Entropy computation via thermodynamic integration |

### Common Imports

```python
# Model training
from highentDCA.models.edDCA import fit

# Training utilities
from highentDCA.training import train_graph

# Checkpoint classes
from highentDCA.checkpoint import Checkpoint, DecCheckpoint

# Argument parsing
from highentDCA.parser import add_args_train, add_args_edDCA

# Entropy computation
from highentDCA.scripts.entropy import compute_entropy
```

## Usage Examples

### Example 1: Basic edDCA Training

```python
import torch
from pathlib import Path
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, init_parameters, get_device
from adabmDCA.sampling import get_sampler
from highentDCA.models.edDCA import fit
from highentDCA.checkpoint import DecCheckpoint

# Configuration
device = get_device("cuda")
dtype = torch.float32

# Load dataset
dataset = DatasetDCA(
    path_data="data/protein_family.fasta",
    alphabet="protein",
    device=device,
    dtype=dtype,
)

# Initialize parameters and chains
params = init_parameters(L=dataset.L, q=dataset.q, device=device, dtype=dtype)
chains = init_chains(
    nchains=10000,
    L=dataset.L,
    q=dataset.q,
    device=device,
    dtype=dtype,
)
log_weights = torch.zeros(chains.shape[0], device=device, dtype=dtype)

# Set up sampler
sampler = get_sampler("gibbs")

# Configure checkpoint
checkpoint = DecCheckpoint(
    file_paths={
        "log": Path("output/training.log"),
        "params": Path("output/params.dat"),
        "chains": Path("output/chains.fasta"),
    },
    tokens=dataset.tokens,
    args={
        "model": "edDCA",
        "data": "data/protein_family.fasta",
        "alphabet": "protein",
        "density": 0.02,
        "drate": 0.01,
        # ... other args
    },
    target_density=0.02,
)

# Train edDCA model
fit(
    sampler=sampler,
    chains=chains,
    log_weights=log_weights,
    fi_target=dataset.fi,
    fij_target=dataset.fij,
    params=params,
    mask=torch.ones_like(params["coupling_matrix"]),
    lr=0.01,
    nsweeps=10,
    target_pearson=0.95,
    target_density=0.02,
    drate=0.01,
    checkpoint=checkpoint,
)
```

### Example 2: Custom Checkpoint Strategy

```python
from highentDCA.checkpoint import DecCheckpoint

# Create custom density checkpoints
custom_densities = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02]

checkpoint = DecCheckpoint(
    file_paths={
        "log": Path("output/custom.log"),
        "params": Path("output/params.dat"),
        "chains": Path("output/chains.fasta"),
    },
    tokens="protein",
    args=training_args,
    checkpt_steps=custom_densities,
    target_density=0.02,
)
```

### Example 3: Computing Entropy

```python
from highentDCA.scripts.entropy import compute_entropy
from adabmDCA.io import load_params
from adabmDCA.sampling import get_sampler

# Load trained model
params, tokens, L, q = load_params("output/params.dat")

# Initialize chains
chains = init_chains(nchains=10000, L=L, q=q, device="cuda")

# Get sampler
sampler = get_sampler("gibbs")

# Compute entropy
entropy = compute_entropy(
    params=params,
    path_targetseq="data/target_sequence.fasta",
    sampler=sampler,
    chains=chains,
    output="output/entropy",
    label="density_0.020",
    tokens=tokens,
    theta_max=5.0,
    nsteps=100,
    nsweeps=100,
    device="cuda",
)

print(f"Model entropy: {entropy:.4f}")
```

### Example 4: Training on Specific Graph

```python
from highentDCA.training import train_graph
import torch

# Create sparse mask (e.g., contact map)
mask = torch.zeros(L, q, L, q, device=device, dtype=torch.bool)
# ... populate mask with desired interactions ...

# Train on this specific graph
chains, params, log_weights, history = train_graph(
    sampler=sampler,
    chains=chains,
    mask=mask,
    fi=dataset.fi,
    fij=dataset.fij,
    params=params,
    nsweeps=10,
    lr=0.01,
    max_epochs=10000,
    target_pearson=0.95,
    checkpoint=checkpoint,
)

# Access training history
import matplotlib.pyplot as plt
plt.plot(history["epochs"], history["pearson"])
plt.xlabel("Epochs")
plt.ylabel("Pearson Correlation")
plt.show()
```

## Integration with adabmDCA

highentDCA is built on top of `adabmDCA`, so you have access to all adabmDCA functionality:

```python
# Import adabmDCA utilities
from adabmDCA.fasta import import_from_fasta, write_fasta
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.io import load_params, save_params
from adabmDCA.sampling import gibbs_sampling, metropolis
from adabmDCA.statmech import compute_energy, compute_log_likelihood
from adabmDCA.graph import decimate_graph, compute_density

# Use with highentDCA
from highentDCA.models.edDCA import fit
from highentDCA.checkpoint import DecCheckpoint
```

## Type Hints

highentDCA uses Python type hints for better code documentation and IDE support:

```python
from typing import Dict, Callable
import torch

def fit(
    sampler: Callable,
    chains: torch.Tensor,
    log_weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    params: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    lr: float,
    nsweeps: int,
    target_pearson: float,
    target_density: float,
    drate: float,
    checkpoint: Checkpoint,
    fi_test: torch.Tensor | None = None,
    fij_test: torch.Tensor | None = None,
    args=None,
) -> None:
    ...
```

## Module Details

Click on the links below for detailed documentation of each module:

- **[Checkpoint](checkpoint.md)**: Learn about checkpoint strategies
- **[Training](training.md)**: Understand graph training functions
- **[edDCA Model](models.edDCA.md)**: Deep dive into entropy decimation
- **[CLI](cli.md)**: Command-line interface implementation
- **[Parser](parser.md)**: Argument parsing utilities
- **[Entropy Computation](entropy.md)**: Thermodynamic integration details

## Contributing

To contribute to the API:

1. Follow PEP 8 style guidelines
2. Add type hints to all function signatures
3. Write comprehensive docstrings (Google style)
4. Include examples in docstrings where appropriate
5. Update this documentation when adding new features

## See Also

- [adabmDCA API Documentation](https://spqb.github.io/adabmDCApy/api/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Usage Guide](../highentDCA_usage.md)
