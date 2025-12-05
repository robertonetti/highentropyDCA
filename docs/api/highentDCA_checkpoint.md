# Checkpoint Module

The `highentDCA.checkpoint` module provides checkpoint strategies for saving model state during training.

## Overview

Checkpoints are essential for:
- Saving model progress during long training runs
- Creating snapshots at specific densities during decimation
- Logging training metrics
- Enabling training resumption after interruptions
- Tracking experiments with Weights & Biases

## Classes

### `Checkpoint` (Abstract Base Class)

Base class for all checkpoint strategies. Defines the interface that all checkpoint implementations must follow.

```python
from highentDCA.checkpoint import Checkpoint
```

#### Constructor

```python
Checkpoint(
    file_paths: dict,
    tokens: str,
    args: dict,
    params: Dict[str, torch.Tensor] | None = None,
    chains: torch.Tensor | None = None,
    use_wandb: bool = False,
)
```

**Parameters:**

- **file_paths** (`dict`): Dictionary with file paths for saving outputs.
  - Required keys: `"log"`, `"params"`, `"chains"`
  - Values: `pathlib.Path` objects or strings
  
- **tokens** (`str`): Alphabet for sequence encoding (e.g., `"protein"`, `"rna"`, or custom)

- **args** (`dict`): Training configuration dictionary with keys:
  - `"model"`: Model type (`"bmDCA"`, `"eaDCA"`, `"edDCA"`)
  - `"data"`: Path to input MSA
  - `"alphabet"`: Sequence alphabet
  - `"lr"`: Learning rate
  - `"nsweeps"`: Number of sweeps
  - `"target"`: Target Pearson correlation
  - Additional model-specific args

- **params** (`Dict[str, torch.Tensor] | None`): Initial model parameters
  - `"bias"`: Fields, shape `(L, q)`
  - `"coupling_matrix"`: Couplings, shape `(L, q, L, q)`
  
- **chains** (`torch.Tensor | None`): Initial Markov chains, shape `(n_chains, L, q)`

- **use_wandb** (`bool`): Enable Weights & Biases logging (default: `False`)

#### Methods

##### `header_log()`

Write the header row to the log file.

```python
checkpoint.header_log()
```

**Example:**

```python
checkpoint = DecCheckpoint(...)
checkpoint.header_log()
# Writes: "Epochs     Pearson    Entropy    Density    Time      "
```

##### `log(record: Dict[str, Any])`

Log training metrics and append to log file.

```python
checkpoint.log(record: Dict[str, Any])
```

**Parameters:**

- **record** (`Dict[str, Any]`): Metrics to log. Valid keys:
  - `"Epochs"`: Epoch number (int)
  - `"Pearson"`: Pearson correlation (float)
  - `"Entropy"`: Model entropy (float)
  - `"Density"`: Coupling density (float)
  - `"Time"`: Elapsed time in seconds (float)

**Raises:**

- `ValueError`: If record contains unrecognized keys

**Example:**

```python
checkpoint.log({
    "Epochs": 100,
    "Pearson": 0.95,
    "Entropy": 120.5,
    "Density": 0.587,
    "Time": 125.3,
})
```

##### `check(*args, **kwargs)` (Abstract)

Check if a checkpoint condition is met. Must be implemented by subclasses.

```python
should_save = checkpoint.check(...)
```

**Returns:**

- `bool`: `True` if checkpoint should be saved, `False` otherwise

##### `save(params, mask, chains, log_weights)` (Abstract)

Save model state to disk. Must be implemented by subclasses.

```python
checkpoint.save(
    params=params,
    mask=mask,
    chains=chains,
    log_weights=log_weights,
)
```

**Parameters:**

- **params** (`Dict[str, torch.Tensor]`): Model parameters
- **mask** (`torch.Tensor`): Binary coupling mask, shape `(L, q, L, q)`
- **chains** (`torch.Tensor`): Markov chains, shape `(n_chains, L, q)`
- **log_weights** (`torch.Tensor`): AIS log-weights, shape `(n_chains,)`

---

### `DecCheckpoint` (Density-Based Checkpoint)

Checkpoint strategy based on coupling matrix density thresholds. Saves model state when density crosses predefined values.

```python
from highentDCA.checkpoint import DecCheckpoint
```

#### Constructor

```python
DecCheckpoint(
    file_paths: dict,
    tokens: str,
    args: dict,
    params: Dict[str, torch.Tensor] | None = None,
    chains: torch.Tensor | None = None,
    checkpt_steps: list[float] | None = None,
    use_wandb: bool = False,
    target_density: float | None = None,
    n_steps: int = 10,
    **kwargs,
)
```

**Additional Parameters:**

- **checkpt_steps** (`list[float] | None`): Custom density thresholds (e.g., `[0.9, 0.5, 0.2, 0.05]`)
  - If `None`, generates `n_steps` geometrically-spaced values from `0.99` to `target_density`
  - Values are automatically sorted in descending order

- **target_density** (`float | None`): Final target density (used to generate checkpt_steps if None provided)

- **n_steps** (`int`): Number of checkpoint steps to generate (default: 10)

**Automatic Checkpoint Generation:**

If `checkpt_steps=None`, generates geometric sequence:

$$\rho_i = \rho_{\text{start}} \cdot \left(\frac{\rho_{\text{target}}}{\rho_{\text{start}}}\right)^{\frac{i}{N-1}}$$

where $\rho_{\text{start}} = 0.99$, $\rho_{\text{target}}$ = `target_density`, and $N$ = `n_steps`.

**Example:**

```python
from pathlib import Path
from highentDCA.checkpoint import DecCheckpoint

checkpoint = DecCheckpoint(
    file_paths={
        "log": Path("output/training.log"),
        "params": Path("output/params.dat"),
        "chains": Path("output/chains.fasta"),
    },
    tokens="protein",
    args={
        "model": "edDCA",
        "data": "data/alignment.fasta",
        "alphabet": "protein",
        "lr": 0.01,
        "nsweeps": 10,
        "nsweeps_dec": 10,
        "target": 0.95,
        "density": 0.02,
        "drate": 0.01,
        "pseudocount": None,
        "dtype": "float32",
        "label": None,
        "nepochs": 50000,
        "sampler": "gibbs",
        "nchains": 10000,
        "seed": 42,
    },
    target_density=0.02,
    n_steps=10,
    use_wandb=False,
)

# Automatic checkpoints at:
# [0.99, 0.82, 0.68, 0.56, 0.46, 0.38, 0.31, 0.26, 0.21, 0.02]
```

#### Methods

##### `check(density: float)`

Check if current density crossed a checkpoint threshold.

```python
should_save = checkpoint.check(density=0.35)
```

**Parameters:**

- **density** (`float`): Current coupling matrix density

**Returns:**

- `bool`: `True` if density crossed a threshold, `False` otherwise

**Behavior:**

- Automatically removes all passed thresholds
- Allows skipping intermediate checkpoints if density drops rapidly
- Returns `False` when all checkpoints exhausted

**Example:**

```python
checkpoint = DecCheckpoint(
    ...,
    checkpt_steps=[0.9, 0.5, 0.2, 0.05],
)

checkpoint.check(0.85)  # False (0.85 > 0.9)
checkpoint.check(0.60)  # True (crossed 0.9)
checkpoint.check(0.45)  # True (crossed 0.5)
checkpoint.check(0.18)  # True (crossed 0.2, skipped)
checkpoint.check(0.03)  # True (crossed 0.05)
checkpoint.check(0.02)  # False (all exhausted)
```

##### `save(params, mask, chains, log_weights, density: float)`

Save model state with density-labeled filenames.

```python
checkpoint.save(
    params=params,
    mask=mask,
    chains=chains,
    log_weights=log_weights,
    density=0.35,
)
```

**Parameters:**

- **density** (`float`): Current density (appended to filenames)
- Other parameters: Same as base `Checkpoint.save()`

**Output Files:**

Creates two files with density labels:

- `{stem}_density_{density:.3f}.dat` - Parameters file
- `{stem}_density_{density:.3f}.fasta` - Chains file

**Example:**

```python
checkpoint.save(
    params=params,
    mask=mask,
    chains=chains,
    log_weights=log_weights,
    density=0.587,
)

# Creates:
# - output/params_density_0.587.dat
# - output/chains_density_0.587.fasta
```

## Complete Usage Example

```python
import torch
from pathlib import Path
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, init_parameters
from adabmDCA.sampling import get_sampler
from highentDCA.models.edDCA import fit
from highentDCA.checkpoint import DecCheckpoint

# Configuration
device = torch.device("cuda")
dtype = torch.float32

# Load dataset
dataset = DatasetDCA(
    path_data="data/PF00072.fasta",
    alphabet="protein",
    device=device,
    dtype=dtype,
)

# Initialize model
params = init_parameters(L=dataset.L, q=dataset.q, device=device, dtype=dtype)
chains = init_chains(10000, dataset.L, dataset.q, device=device, dtype=dtype)
log_weights = torch.zeros(10000, device=device, dtype=dtype)
sampler = get_sampler("gibbs")

# Create output directory
output_dir = Path("results/PF00072")
output_dir.mkdir(parents=True, exist_ok=True)

# Configure checkpoint with custom densities
checkpoint = DecCheckpoint(
    file_paths={
        "log": output_dir / "training.log",
        "params": output_dir / "params.dat",
        "chains": output_dir / "chains.fasta",
    },
    tokens=dataset.tokens,
    args={
        "model": "edDCA",
        "data": "data/PF00072.fasta",
        "alphabet": "protein",
        "lr": 0.01,
        "nsweeps": 10,
        "nsweeps_dec": 10,
        "target": 0.95,
        "density": 0.02,
        "drate": 0.01,
        "pseudocount": None,
        "dtype": "float32",
        "label": "PF00072_edDCA",
        "nepochs": 50000,
        "sampler": "gibbs",
        "nchains": 10000,
        "seed": 42,
    },
    checkpt_steps=[0.98, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02],
    target_density=0.02,
    use_wandb=False,
)

# Initialize log header
checkpoint.header_log()

# Train model with checkpoint
mask = torch.ones(
    dataset.L, dataset.q, dataset.L, dataset.q,
    device=device, dtype=torch.bool
)

fit(
    sampler=sampler,
    chains=chains,
    log_weights=log_weights,
    fi_target=dataset.fi,
    fij_target=dataset.fij,
    params=params,
    mask=mask,
    lr=0.01,
    nsweeps=10,
    target_pearson=0.95,
    target_density=0.02,
    drate=0.01,
    checkpoint=checkpoint,
    args=checkpoint.args,
)

print("Training complete!")
print(f"Results saved in: {output_dir}")
```

## Weights & Biases Integration

Enable experiment tracking with W&B:

```python
checkpoint = DecCheckpoint(
    ...,
    use_wandb=True,  # Enable W&B
)

# Metrics automatically logged to W&B:
# - Epochs
# - Pearson correlation
# - Entropy
# - Coupling density
# - Training time
```

**Prerequisites:**

```bash
pip install wandb
wandb login
```

## See Also

- [Training Module](training.md) - Training functions using checkpoints
- [edDCA Model](models.edDCA.md) - Model fitting with checkpoints
- [Usage Guide](../highentDCA_usage.md) - CLI checkpoint options
