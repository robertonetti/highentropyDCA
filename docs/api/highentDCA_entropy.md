# Entropy Computation Module

The `highentDCA.scripts.entropy` module implements thermodynamic integration for computing model entropy.

## Overview

Model entropy is computed by integrating the average sequence identity with respect to a bias parameter θ that guides sampling towards a target sequence. This thermodynamic integration approach provides accurate entropy estimates for Boltzmann machines.

## Functions

### `compute_entropy()`

Compute the entropy of a DCA model using thermodynamic integration.

```python
from highentDCA.scripts.entropy import compute_entropy

entropy = compute_entropy(
    params: Dict[str, torch.Tensor],
    path_targetseq: str,
    sampler: Callable,
    chains: torch.Tensor,
    output: str,
    label: str,
    tokens: str,
    nsweeps_zero: int = 100,
    nsweeps_theta: int = 100,
    theta_max: float = 5.0,
    nsteps: int = 100,
    nsweeps: int = 100,
    device: str = "cuda",
    dtype: str = "float32",
) -> float
```

## Parameters

- **params** (`Dict[str, torch.Tensor]`): Model parameters
  - `"bias"`: Fields, shape `(L, q)`
  - `"coupling_matrix"`: Couplings, shape `(L, q, L, q)`

- **path_targetseq** (`str`): Path to FASTA file with target sequence
  - Must contain at least one sequence
  - If multiple sequences, first one is used
  - Typically the wild-type or reference sequence

- **sampler** (`Callable`): Sampling function
  - Signature: `sampler(chains, params, nsweeps) -> chains`

- **chains** (`torch.Tensor`): Initial Markov chains, shape `(n_chains, L, q)`

- **output** (`str`): Path to output directory for log files

- **label** (`str`): Label for output files (e.g., density value)
  - Creates: `density_{label}.log`

- **tokens** (`str`): Sequence alphabet (e.g., `"protein"`, `"rna"`)

- **nsweeps_zero** (`int`): Sweeps for equilibration at θ=0 (default: 100)

- **nsweeps_theta** (`int`): Sweeps for equilibration at θ=θ_max (default: 100)

- **theta_max** (`float`): Initial maximum bias strength (default: 5.0)
  - Automatically adjusted to achieve ~10% target sequences

- **nsteps** (`int`): Number of integration steps (default: 100)
  - More steps = more accurate (but slower)

- **nsweeps** (`int`): Sweeps per integration step (default: 100)

- **device** (`str`): Computation device (`"cuda"` or `"cpu"`)

- **dtype** (`str`): Data type (`"float32"` or `"float64"`)

## Returns

- `float`: Computed entropy value

## Raises

- `FileNotFoundError`: If `path_targetseq` doesn't exist
- `ValueError`: If `label` is `None`

## Algorithm

The thermodynamic integration method computes entropy as:

$$S = E_0 - F(\theta_{\max})$$

where:
- $E_0$ = average energy at θ=0 (unbiased model)
- $F(\theta_{\max})$ = free energy at θ=θ_max

### Step 1: Equilibrate at θ=0

Sample from unbiased model:

```python
params_0 = params.copy()
chains_0 = sampler(chains, params_0, nsweeps_zero)
E_0 = mean(energy(chains_0, params_0))
```

### Step 2: Equilibrate at θ=θ_max

Sample with bias towards target sequence:

```python
params_theta = params.copy()
params_theta["bias"] += theta_max * target_seq  # Add bias

chains_theta = sampler(chains_theta, params_theta, nsweeps_theta)
```

### Step 3: Adjust θ_max

Ensure ~10% of chains match target sequence:

```python
while p_target < 0.1:
    theta_max *= 1.01
    params_theta["bias"] = params["bias"] + theta_max * target_seq
    chains_theta = sampler(chains_theta, params_theta, 100)
    p_target = fraction_matching_target(chains_theta)
```

### Step 4: Thermodynamic Integration

Integrate average sequence identity from θ=0 to θ_max:

```python
F = log(p_target) + mean(energy(chains_at_target, params_theta))

for theta in linspace(0, theta_max, nsteps):
    params_theta["bias"] = params["bias"] + theta * target_seq
    chains = sampler(chains, params_theta, nsweeps)
    
    seqID = sequence_identity(chains, target_seq)
    F += integration_weight * mean(seqID)  # Trapezoidal rule

S = E_0 - F
```

## Complete Example

```python
import torch
from pathlib import Path
from adabmDCA.io import load_params
from adabmDCA.utils import init_chains
from adabmDCA.sampling import get_sampler
from highentDCA.scripts.entropy import compute_entropy

# Load trained model
params, tokens, L, q = load_params("results/params.dat")

# Initialize chains
device = torch.device("cuda")
dtype = torch.float32
chains = init_chains(
    nchains=10000,
    L=L,
    q=q,
    device=device,
    dtype=dtype,
)

# Get sampler
sampler = get_sampler("gibbs")

# Create output directory
output_dir = Path("results/entropy_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Compute entropy
print("Computing model entropy...")
S = compute_entropy(
    params=params,
    path_targetseq="data/wildtype.fasta",
    sampler=sampler,
    chains=chains,
    output=str(output_dir),
    label="final_model",
    tokens=tokens,
    nsweeps_zero=200,
    nsweeps_theta=200,
    theta_max=5.0,
    nsteps=200,
    nsweeps=100,
    device="cuda",
    dtype="float32",
)

print(f"Model entropy: {S:.4f}")
print(f"Log saved: {output_dir}/density_final_model.log")
```

## Output Files

The function creates a detailed log file: `density_{label}.log`

### Log File Format

```
density:              final_model                                       
nchains:              10000                                             
nsweeps:              100                                               
nsweeps  theta:       200                                               
nsweeps zero:         200                                               
nsteps:               200                                               
data type:            float32                                           

Epoch           Theta           Free Energy     Entropy         Time           
0               0.000           50.123          125.456         0.000          
1               0.025           50.145          125.434         1.234          
2               0.050           50.178          125.401         2.456          
...
199             4.975           30.234          145.345         245.678        
```

## Analyzing Entropy Results

### Reading Log Files

```python
import pandas as pd

# Parse log file (skip header lines)
df = pd.read_csv(
    "results/entropy_analysis/density_final_model.log",
    sep=r'\s+',
    skiprows=8,  # Skip configuration lines
    names=["Epoch", "Theta", "Free_Energy", "Entropy", "Time"],
)

print(f"Final entropy: {df['Entropy'].iloc[-1]:.4f}")
print(f"Computation time: {df['Time'].iloc[-1]:.1f} seconds")
```

### Plotting Integration Progress

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Free energy evolution
axes[0].plot(df['Theta'], df['Free_Energy'], linewidth=2)
axes[0].set_xlabel('θ')
axes[0].set_ylabel('Free Energy F(θ)')
axes[0].set_title('Free Energy Integration')
axes[0].grid(True, alpha=0.3)

# Entropy evolution
axes[1].plot(df['Theta'], df['Entropy'], linewidth=2)
axes[1].set_xlabel('θ')
axes[1].set_ylabel('Entropy S(θ)')
axes[1].set_title('Entropy Evolution')
axes[1].grid(True, alpha=0.3)

# Convergence check
axes[2].plot(df['Epoch'], df['Entropy'], linewidth=2)
axes[2].axhline(y=df['Entropy'].iloc[-1], color='r', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Integration Step')
axes[2].set_ylabel('Entropy')
axes[2].set_title('Integration Convergence')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('entropy_integration.png', dpi=300)
plt.show()
```

## Integration with edDCA

During edDCA training, entropy is computed at density checkpoints:

```python
from highentDCA.models.edDCA import fit

# fit() automatically calls compute_entropy at checkpoints
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
    args=args,  # Must contain entropy parameters
)

# Results saved in:
# - output/entropy_decimation/density_0.980.log
# - output/entropy_decimation/density_0.587.log
# - etc.
```

## Performance Tuning

### Fast (Less Accurate)

```python
S = compute_entropy(
    ...,
    nsweeps_zero=50,
    nsweeps_theta=50,
    nsteps=50,
    nsweeps=50,
)
# ~10x faster, ±5% accuracy
```

### Standard (Default)

```python
S = compute_entropy(
    ...,
    nsweeps_zero=100,
    nsweeps_theta=100,
    nsteps=100,
    nsweeps=100,
)
# Balanced speed/accuracy, ±2% accuracy
```

### High Accuracy

```python
S = compute_entropy(
    ...,
    nsweeps_zero=200,
    nsweeps_theta=200,
    nsteps=200,
    nsweeps=200,
)
# ~4x slower, ±1% accuracy
```

## Theoretical Background

### Thermodynamic Integration

The entropy of a Boltzmann machine is related to its partition function:

$$S = \langle E \rangle - F$$

where $F = -\log Z$ is the free energy.

By introducing a bias parameter θ:

$$H_\theta(\mathbf{s}) = H(\mathbf{s}) - \theta \sum_i s_i \cdot s_i^{\text{target}}$$

We can compute:

$$F(\theta) = F(0) + \int_0^\theta \left\langle \sum_i s_i \cdot s_i^{\text{target}} \right\rangle_{\theta'} d\theta'$$

### Sequence Identity

The integrand is the average sequence identity:

$$\text{seqID}(\mathbf{s}) = \frac{1}{L} \sum_{i=1}^L \delta_{s_i, s_i^{\text{target}}}$$

### Numerical Integration

Trapezoidal rule:

$$\int_0^{\theta_{\max}} f(\theta) d\theta \approx \frac{\Delta\theta}{2} \left[ f(0) + 2\sum_{i=1}^{N-1} f(i\Delta\theta) + f(\theta_{\max}) \right]$$

where $\Delta\theta = \theta_{\max} / N$.

## Troubleshooting

### Low target sequence percentage (<5%)

**Solution**: Increase `theta_max`:

```python
S = compute_entropy(..., theta_max=10.0)
```

### Entropy oscillates

**Solution**: Increase equilibration:

```python
S = compute_entropy(
    ...,
    nsweeps_zero=200,
    nsweeps_theta=200,
    nsweeps=200,
)
```

### Out of memory

**Solution**: Reduce number of chains:

```python
chains = init_chains(5000, L, q)  # Instead of 10000
```

## See Also

- [edDCA Model](highentDCA_models.edDCA.md) - Automatic entropy computation
- [Statistical Mechanics](https://spqb.github.io/adabmDCApy/api/statmech/) - Energy and partition function
- [Barrat-Charlaix et al., 2021](https://doi.org/10.1103/PhysRevE.104.024407) - Entropy decimation paper
