# edDCA Model Module

The `highentDCA.models.edDCA` module implements the entropy-decimated Direct Coupling Analysis (edDCA) algorithm.

## Overview

The edDCA algorithm progressively decimates (sparsifies) a fully-connected DCA model while:
- Maintaining model accuracy (high Pearson correlation with data)
- Computing entropy at key density checkpoints via thermodynamic integration
- Tracking the relationship between model complexity and information content

## Functions

### `fit()`

Main function for training an entropy-decimated DCA model.

```python
from highentDCA.models.edDCA import fit

fit(
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
    *extra_args,
    **kwargs,
) -> None
```

#### Parameters

- **sampler** (`Callable`): Sampling function for updating Markov chains
  - Signature: `sampler(chains, params, nsweeps) -> chains`
  - Get via: `from adabmDCA.sampling import get_sampler`

- **chains** (`torch.Tensor`): Initial Markov chains
  - Shape: `(n_chains, L, q)` in one-hot encoding
  - Initialize with: `from adabmDCA.utils import init_chains`

- **log_weights** (`torch.Tensor`): Log-weights for Annealed Importance Sampling
  - Shape: `(n_chains,)`
  - Initialize: `torch.zeros(n_chains, device=device, dtype=dtype)`

- **fi_target** (`torch.Tensor`): Single-point frequencies from training data
  - Shape: `(L, q)`
  - Compute with: `from adabmDCA.stats import get_freq_single_point`

- **fij_target** (`torch.Tensor`): Two-point frequencies from training data
  - Shape: `(L, q, L, q)`
  - Compute with: `from adabmDCA.stats import get_freq_two_points`

- **params** (`Dict[str, torch.Tensor]`): Model parameters
  - Keys: `"bias"`, `"coupling_matrix"`
  - `"bias"`: Fields, shape `(L, q)`
  - `"coupling_matrix"`: Couplings, shape `(L, q, L, q)`
  - Initialize with: `from adabmDCA.utils import init_parameters`

- **mask** (`torch.Tensor`): Binary mask for coupling matrix
  - Shape: `(L, q, L, q)`
  - Initial: `torch.ones_like(params["coupling_matrix"])`
  - Updated during decimation

- **lr** (`float`): Learning rate for gradient descent
  - Typical values: 0.005 - 0.02
  - Default: 0.01

- **nsweeps** (`int`): Monte Carlo sweeps per gradient update (initial training)
  - Typical values: 5 - 50
  - Default: 10

- **target_pearson** (`float`): Target Pearson correlation for convergence
  - Range: 0.90 - 0.99
  - Default: 0.95
  - Higher = stricter convergence

- **target_density** (`float`): Target coupling density to reach
  - Range: 0.01 - 0.20
  - Default: 0.02 (2% of couplings)
  - Lower = sparser model

- **drate** (`float`): Decimation rate (fraction of couplings to prune per step)
  - Range: 0.005 - 0.05
  - Default: 0.01 (1%)
  - Smaller = more gradual decimation

- **checkpoint** (`Checkpoint`): Checkpoint object for saving progress
  - Type: `DecCheckpoint` (recommended for edDCA)
  - See: [Checkpoint Module](highentDCA_checkpoint.md)

- **fi_test** (`torch.Tensor | None`): Single-point frequencies from test data
  - Shape: `(L, q)`
  - Optional: For test set evaluation

- **fij_test** (`torch.Tensor | None`): Two-point frequencies from test data
  - Shape: `(L, q, L, q)`
  - Optional: For test set evaluation

- **args**: Training arguments (namespace or dict)
  - Must contain entropy computation parameters:
    - `theta_max`, `nsteps`, `nsweeps_step`
    - `nsweeps_zero`, `nsweeps_theta`
    - `data` (path to target sequence for entropy)

#### Returns

- `None`: Function modifies `params`, `chains`, `mask` in-place and saves via checkpoint

#### Raises

- `ValueError`: If input tensors have incorrect dimensions
  - `fi_target` must be 2D
  - `fij_target` must be 4D
  - `chains` must be 3D

## Algorithm

The edDCA algorithm consists of the following steps:

### 1. Initial Convergence

If the model isn't already converged (Pearson < target):

```python
chains, params, log_weights, _ = train_graph(
    sampler=sampler,
    chains=chains,
    log_weights=log_weights,
    mask=mask,
    fi=fi_target,
    fij=fij_target,
    params=params,
    nsweeps=nsweeps,
    lr=lr,
    max_epochs=MAX_EPOCHS,
    target_pearson=target_pearson,
    check_slope=False,
    checkpoint=checkpoint,
    progress_bar=False,
)
```

### 2. Compute Initial Entropy

Before decimation starts:

```python
S_initial = compute_entropy(
    params=params,
    path_targetseq=args.data,
    sampler=sampler,
    chains=chains,
    output=output_folder,
    label=f"{density:.3f}",
    tokens=checkpoint.tokens,
    ...
)
```

### 3. Decimation Loop

While `density > target_density`:

#### a. Decimate Graph

```python
from adabmDCA.graph import decimate_graph

params, mask = decimate_graph(
    pij=pij,
    params=params,
    mask=mask,
    drate=drate,
)
```

Removes `drate` fraction of weakest couplings based on empirical two-point statistics.

#### b. Update AIS Weights

```python
from adabmDCA.statmech import _update_weights_AIS

log_weights = _update_weights_AIS(
    prev_params=prev_params,
    curr_params=params,
    chains=chains,
    log_weights=log_weights,
)
```

Tracks partition function changes for log-likelihood estimation.

#### c. Equilibrate Chains

```python
chains = sampler(
    chains=chains,
    params=params,
    nsweeps=args.nsweeps_dec,
)
```

#### d. Re-converge Model

```python
chains, params, log_weights, _ = train_graph(
    sampler=sampler,
    chains=chains,
    log_weights=log_weights,
    mask=mask,
    fi=fi_target,
    fij=fij_target,
    params=params,
    nsweeps=args.nsweeps_dec,
    lr=lr,
    max_epochs=MAX_EPOCHS,
    target_pearson=target_pearson_dec,
    check_slope=False,
    progress_bar=False,
    checkpoint=None,
)
```

#### e. Checkpoint & Entropy

If density crossed a checkpoint threshold:

```python
if checkpoint_dec.check(density):
    S = compute_entropy(...)
    
    checkpoint_dec.log({
        "Epochs": count,
        "Pearson": pearson,
        "Entropy": S,
        "Density": density,
        "Time": elapsed_time,
    })
    
    checkpoint_dec.save(
        params=params,
        mask=mask,
        chains=chains,
        log_weights=log_weights,
        density=density,
    )
```

### 4. Final Save

After reaching target density:

```python
checkpoint_dec.save(
    params=params,
    mask=mask,
    chains=chains,
    log_weights=log_weights,
    density=density,
)
```

## Complete Example

```python
import torch
from pathlib import Path
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, init_parameters, get_device, get_dtype
from adabmDCA.sampling import get_sampler
from highentDCA.models.edDCA import fit
from highentDCA.checkpoint import DecCheckpoint

# Configuration
device = get_device("cuda")
dtype = get_dtype("float32")

# Load dataset
dataset = DatasetDCA(
    path_data="data/PF00072.fasta",
    alphabet="protein",
    clustering_seqid=0.8,
    device=device,
    dtype=dtype,
)

print(f"Dataset: {dataset.M} sequences, L={dataset.L}, q={dataset.q}")

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

# Configure sampler
sampler = get_sampler("gibbs")

# Set up output
output_dir = Path("results/PF00072_edDCA")
output_dir.mkdir(parents=True, exist_ok=True)

# Create checkpoint
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
        "label": None,
        "nepochs": 50000,
        "sampler": "gibbs",
        "nchains": 10000,
        "seed": 42,
        # Entropy parameters
        "theta_max": 5.0,
        "nsteps": 100,
        "nsweeps_step": 100,
        "nsweeps_theta": 100,
        "nsweeps_zero": 100,
    },
    target_density=0.02,
    n_steps=10,
)

# Create args namespace for entropy computation
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args = Args(
    data="data/PF00072.fasta",
    nsweeps_zero=100,
    nsweeps_theta=100,
    theta_max=5.0,
    nsteps=100,
    nsweeps_step=100,
    nsweeps_dec=10,
)

# Initialize full coupling mask
mask = torch.ones(
    dataset.L, dataset.q, dataset.L, dataset.q,
    device=device,
    dtype=torch.bool,
)

# Train edDCA model
print("\nStarting edDCA training...")
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
    args=args,
)

print(f"\nTraining complete! Results in: {output_dir}")
print(f"- Parameters: {output_dir}/params.dat")
print(f"- Chains: {output_dir}/chains.fasta")
print(f"- Log: {output_dir}/training.log")
print(f"- Entropy data: {output_dir}/entropy_decimation/entropy_values.txt")
```

## Analyzing Results

### Load Saved Models

```python
from adabmDCA.io import load_params, load_chains

# Load final model
params, tokens, L, q = load_params("results/PF00072_edDCA/params.dat")

# Load checkpoint at specific density
params_587, _, _, _ = load_params(
    "results/PF00072_edDCA/entropy_decimation/params_density_0.587.dat"
)
```

### Visualize Entropy Evolution

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load entropy data
entropy_df = pd.read_csv(
    "results/PF00072_edDCA/entropy_decimation/entropy_values.txt",
    sep='\t',
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(entropy_df['Density'], entropy_df['Entropy'], 'o-', linewidth=2)
plt.xlabel('Coupling Density', fontsize=14)
plt.ylabel('Model Entropy', fontsize=14)
plt.title('Entropy Evolution During Decimation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('entropy_evolution.png', dpi=300)
plt.show()
```

### Analyze Coupling Sparsity

```python
import torch
import numpy as np

# Load parameters
params, tokens, L, q = load_params("results/PF00072_edDCA/params.dat")

# Count non-zero couplings
couplings = params['coupling_matrix']
total_couplings = L * (L - 1) / 2 * q * q  # Upper triangle only
non_zero = (couplings.abs() > 1e-10).sum().item() / 2  # Upper triangle

density = non_zero / total_couplings
print(f"Final density: {density:.4f}")
print(f"Non-zero couplings: {int(non_zero):,} / {int(total_couplings):,}")

# Compute sparsity per position pair
coupling_strength = torch.linalg.norm(
    couplings.reshape(L, q, L, q),
    dim=(1, 3),
).cpu().numpy()

# Plot coupling matrix
plt.figure(figsize=(10, 10))
plt.imshow(coupling_strength, cmap='viridis')
plt.colorbar(label='Coupling Strength (Frobenius norm)')
plt.xlabel('Position j')
plt.ylabel('Position i')
plt.title(f'Sparse Coupling Matrix (density={density:.3f})')
plt.tight_layout()
plt.savefig('coupling_matrix.png', dpi=300)
plt.show()
```

## Performance Tips

1. **GPU Memory**: Reduce `nchains` if out of memory
2. **Speed**: Increase `drate` for faster (but less refined) decimation
3. **Accuracy**: Increase `nsweeps_dec` for better equilibration
4. **Entropy**: Reduce `nsteps` for faster (less accurate) entropy estimates

## See Also

- [Training Module](highentDCA_training.md) - Graph training function
- [Checkpoint Module](highentDCA_checkpoint.md) - Checkpoint strategies
- [Entropy Module](highentDCA_entropy.md) - Thermodynamic integration
- [Usage Guide](../highentDCA_usage.md) - CLI usage
