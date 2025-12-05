# Training Module

The `highentDCA.training` module provides functions for training DCA models on sparse graphs.

## Functions

### `train_graph()`

Trains a DCA model on a fixed sparse graph using gradient descent until convergence.

```python
from highentDCA.training import train_graph

chains, params, log_weights, history = train_graph(
    sampler: Callable,
    chains: torch.Tensor,
    mask: torch.Tensor,
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    nsweeps: int,
    lr: float,
    max_epochs: int,
    target_pearson: float,
    fi_test: torch.Tensor | None = None,
    fij_test: torch.Tensor | None = None,
    checkpoint: Checkpoint | None = None,
    check_slope: bool = False,
    log_weights: torch.Tensor | None = None,
    progress_bar: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]
```

## Parameters

- **sampler** (`Callable`): Sampling function
  - Signature: `sampler(chains, params, nsweeps) -> chains`
  
- **chains** (`torch.Tensor`): Markov chains, shape `(n_chains, L, q)`

- **mask** (`torch.Tensor`): Binary coupling mask, shape `(L, q, L, q)`
  - `mask[i,a,j,b] = 1`: coupling (i,a)-(j,b) is active
  - `mask[i,a,j,b] = 0`: coupling is fixed to zero

- **fi** (`torch.Tensor`): Single-point target frequencies, shape `(L, q)`

- **fij** (`torch.Tensor`): Two-point target frequencies, shape `(L, q, L, q)`

- **params** (`Dict[str, torch.Tensor]`): Model parameters
  - `"bias"`: shape `(L, q)`
  - `"coupling_matrix"`: shape `(L, q, L, q)`

- **nsweeps** (`int`): MC sweeps per gradient update

- **lr** (`float`): Learning rate

- **max_epochs** (`int`): Maximum training epochs

- **target_pearson** (`float`): Target Pearson correlation (0-1)

- **fi_test** (`torch.Tensor | None`): Test set single-point frequencies (optional)

- **fij_test** (`torch.Tensor | None`): Test set two-point frequencies (optional)

- **checkpoint** (`Checkpoint | None`): Checkpoint object (optional)

- **check_slope** (`bool`): Also check correlation slope ≈ 1.0 (default: `False`)

- **log_weights** (`torch.Tensor | None`): AIS log-weights, shape `(n_chains,)` (optional)

- **progress_bar** (`bool`): Show tqdm progress bar (default: `True`)

## Returns

`Tuple` containing:

1. **chains** (`torch.Tensor`): Updated chains, shape `(n_chains, L, q)`
2. **params** (`Dict[str, torch.Tensor]`): Trained parameters
3. **log_weights** (`torch.Tensor`): Updated AIS weights, shape `(n_chains,)`
4. **history** (`Dict[str, List[float]]`): Training history with keys:
   - `"epochs"`: List of epoch numbers
   - `"pearson"`: Pearson correlation at each epoch
   - `"slope"`: Correlation slope at each epoch

## Algorithm

The training process consists of:

### 1. Parameter Update

At each epoch, update parameters using gradient descent:

```python
from adabmDCA.training import update_params

params = update_params(
    fi=fi,
    fij=fij,
    pi=pi,  # Model marginals from chains
    pij=pij,  # Model two-point marginals
    params=params,
    mask=mask,
    lr=lr,
)
```

Gradient: $\Delta h_i^a = \eta (f_i^a - p_i^a)$, $\Delta J_{ij}^{ab} = \eta (f_{ij}^{ab} - p_{ij}^{ab})$

where $\eta$ is the learning rate, $f$ are data frequencies, $p$ are model marginals.

### 2. Chain Sampling

Update chains using MCMC:

```python
chains = sampler(chains=chains, params=params, nsweeps=nsweeps)
```

### 3. Convergence Check

Check if target Pearson correlation reached:

```python
from adabmDCA.stats import get_correlation_two_points

pearson, slope = get_correlation_two_points(
    fij=fij, pij=pij, fi=fi, pi=pi
)

converged = (pearson >= target_pearson)
if check_slope:
    converged &= (abs(slope - 1.0) < 0.1)
```

### 4. Checkpointing

Periodically save model state:

```python
if checkpoint is not None and checkpoint.check(epochs, params, chains):
    checkpoint.log({...})
    checkpoint.save(params, mask, chains, log_weights)
```

## Example: Training on Full Graph

```python
import torch
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import init_chains, init_parameters
from adabmDCA.sampling import get_sampler
from highentDCA.training import train_graph

# Load data
dataset = DatasetDCA("data/alignment.fasta", alphabet="protein")

# Initialize
params = init_parameters(dataset.L, dataset.q)
chains = init_chains(10000, dataset.L, dataset.q)
log_weights = torch.zeros(10000)
sampler = get_sampler("gibbs")

# Full coupling mask
mask = torch.ones(dataset.L, dataset.q, dataset.L, dataset.q, dtype=torch.bool)

# Train
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
)

print(f"Converged after {len(history['epochs'])} epochs")
print(f"Final Pearson: {history['pearson'][-1]:.4f}")
```

## Example: Training on Sparse Graph

```python
import torch
from highentDCA.training import train_graph

# Create sparse mask (e.g., from contact map or previous decimation)
mask = torch.zeros(L, q, L, q, dtype=torch.bool)

# Add specific interactions
for (i, j) in contact_pairs:
    mask[i, :, j, :] = True
    mask[j, :, i, :] = True  # Symmetric

print(f"Mask density: {mask.float().mean():.4f}")

# Train on this sparse graph
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
    log_weights=log_weights,
)
```

## Example: With Test Set Evaluation

```python
from adabmDCA.dataset import DatasetDCA
from highentDCA.training import train_graph

# Load train and test data
train_data = DatasetDCA("data/train.fasta", alphabet="protein")
test_data = DatasetDCA("data/test.fasta", alphabet="protein")

# Train with test set monitoring
chains, params, log_weights, history = train_graph(
    sampler=sampler,
    chains=chains,
    mask=mask,
    fi=train_data.fi,
    fij=train_data.fij,
    fi_test=test_data.fi,
    fij_test=test_data.fij,
    params=params,
    nsweeps=10,
    lr=0.01,
    max_epochs=10000,
    target_pearson=0.95,
)
```

## Example: Plotting Training History

```python
import matplotlib.pyplot as plt

# Plot Pearson evolution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history['epochs'], history['pearson'], linewidth=2)
ax[0].axhline(y=0.95, color='r', linestyle='--', label='Target')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Pearson Correlation')
ax[0].set_title('Convergence')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].plot(history['epochs'], history['slope'], linewidth=2)
ax[1].axhline(y=1.0, color='r', linestyle='--', label='Ideal')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Correlation Slope')
ax[1].set_title('Slope Evolution')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()
```

## Convergence Criteria

The `halt_condition` function determines when to stop training:

```python
def halt_condition(epochs, pearson, slope, check_slope):
    c1 = pearson < target_pearson
    c2 = epochs < max_epochs
    if check_slope:
        c3 = abs(slope - 1.0) > 0.1
    else:
        c3 = False
    return not c2 * ((not c1) * c3 + c1)
```

Training stops when:
- Pearson ≥ `target_pearson` (AND slope ≈ 1.0 if `check_slope=True`)
- OR `epochs` ≥ `max_epochs`

## Performance Optimization

### GPU Memory

Reduce memory usage:

```python
# Fewer chains
chains = init_chains(5000, L, q)  # Instead of 10000

# Lower precision
dtype = torch.float32  # Instead of float64
```

### Speed

Faster training:

```python
# Fewer sweeps (if convergence is stable)
nsweeps = 5

# Higher learning rate (if stable)
lr = 0.02

# Disable progress bar
progress_bar = False
```

### Accuracy

Better convergence:

```python
# More sweeps
nsweeps = 20

# Lower learning rate
lr = 0.005

# Stricter target
target_pearson = 0.98
check_slope = True
```

## See Also

- [edDCA Model](highentDCA_models.edDCA.md) - Uses `train_graph` for decimation
- [Checkpoint Module](highentDCA_checkpoint.md) - Checkpoint strategies
- [adabmDCA Training](https://spqb.github.io/adabmDCApy/api/training/) - Base training functions
