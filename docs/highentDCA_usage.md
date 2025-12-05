# Usage Guide

This guide covers how to use highentDCA for training entropy-decimated DCA models, including command-line interface usage, parameter tuning, and practical examples.

## Quick Start

The simplest way to train an edDCA model:

```bash
highentDCA train \
    --data your_alignment.fasta \
    --output results \
    --model edDCA \
    --density 0.02
```

This command trains a sparse edDCA model with 2% coupling density using default parameters.

## Command-Line Interface

highentDCA provides a command-line interface through the `highentDCA` command. Currently, the main command is `train`.

### Basic Syntax

```bash
highentDCA train [OPTIONS]
```

### Required Arguments

#### `--data` / `-d`
Path to the input multiple sequence alignment (MSA) in FASTA format.

```bash
--data example_data/PF00072.fasta
```

**Requirements**:
- FASTA format with aligned sequences
- Minimum ~1000 sequences recommended
- All sequences must have the same length
- Quality-controlled alignment (remove fragments, handle gaps)

### Model Selection

#### `--model` / `-m`
Choose the type of DCA model to train. For highentDCA, use `edDCA`.

```bash
--model edDCA
```

**Options**:
- `bmDCA`: Fully-connected Boltzmann Machine (standard DCA)
- `eaDCA`: Edge-adding DCA (progressive sparsity)
- `edDCA`: Entropy-decimated DCA (progressive decimation with entropy tracking)

### Output Options

#### `--output` / `-o`
Directory where model outputs will be saved. Default: `DCA_model`

```bash
--output my_results
```

**Output structure**:
```
my_results/
├── params.dat                  # Final model parameters
├── chains.fasta               # Final Markov chains
├── adabmDCA_highent.log      # Training log
├── entropy_decimation/        # Checkpoints at different densities
│   ├── density_0.980.fasta
│   ├── density_0.587.fasta
│   └── ...
└── entropy_values.txt         # Entropy vs density data
```

#### `--label` / `-l`
Add a custom label to output files. Optional.

```bash
--label PF00072_run1
```

Output files will be named: `PF00072_run1_params.dat`, `PF00072_run1_chains.fasta`, etc.

## edDCA-Specific Parameters

### Decimation Parameters

#### `--density`
Target coupling density to reach (fraction of couplings to keep). Default: `0.02` (2%)

```bash
--density 0.05  # Keep 5% of couplings
```

**Typical values**:
- `0.02` - Very sparse (2% of couplings)
- `0.05` - Sparse (5% of couplings)
- `0.10` - Moderately sparse (10% of couplings)

**Guidelines**:
- Lower density = sparser model, faster inference
- Too low may lose important information
- Protein families: 2-5% usually sufficient
- RNA families: 5-10% may be needed

#### `--drate`
Decimation rate: fraction of remaining couplings to prune at each step. Default: `0.01` (1%)

```bash
--drate 0.02  # Prune 2% of remaining couplings per step
```

**Trade-offs**:
- Smaller `drate` (e.g., 0.005): Slower but more gradual decimation
- Larger `drate` (e.g., 0.05): Faster but less refined
- Recommended: 0.01-0.02 for most applications

#### `--nsweeps_dec`
Number of Monte Carlo sweeps per gradient update during decimation. Default: `10`

```bash
--nsweeps_dec 20
```

**Guidelines**:
- Increase for better equilibration (slower training)
- Decrease for faster training (may reduce accuracy)
- Typical range: 5-50

### Entropy Computation Parameters

At pre-defined density checkpoints, highentDCA computes model entropy using thermodynamic integration.

#### `--theta_max`
Maximum integration strength for thermodynamic integration. Default: `5.0`

```bash
--theta_max 10.0
```

Higher values provide better integration range but require more sampling.

#### `--nsteps`
Number of integration steps for entropy computation. Default: `100`

```bash
--nsteps 200
```

More steps = more accurate entropy estimate (but slower).

#### `--nsweeps_step`
Number of MC sweeps per integration step. Default: `100`

```bash
--nsweeps_step 50
```

#### `--nsweeps_theta`
Number of sweeps to equilibrate at θ_max. Default: `100`

```bash
--nsweeps_theta 200
```

#### `--nsweeps_zero`
Number of sweeps to equilibrate at θ=0. Default: `100`

```bash
--nsweeps_zero 200
```

**Entropy computation tips**:
- Increase all nsweeps values for better accuracy
- Decrease for faster (but less accurate) entropy estimates
- Default values are usually sufficient

## General Training Parameters

These parameters control the overall training process.

### Convergence Criteria

#### `--target` / `-t`
Target Pearson correlation between model and data statistics. Default: `0.95`

```bash
--target 0.98  # Stricter convergence
```

**Guidelines**:
- `0.90-0.95`: Standard convergence
- `0.95-0.98`: High accuracy (slower training)
- `>0.98`: Very strict (may not converge for complex families)

#### `--nepochs`
Maximum number of training epochs. Default: `50000`

```bash
--nepochs 100000
```

Training stops when either `--target` or `--nepochs` is reached.

### Sampling Parameters

#### `--sampler`
MCMC sampling method. Default: `gibbs`

```bash
--sampler metropolis
```

**Options**:
- `gibbs`: Gibbs sampling (default, usually faster)
- `metropolis`: Metropolis-Hastings sampling

#### `--nsweeps`
Number of MC sweeps per gradient update (before decimation starts). Default: `10`

```bash
--nsweeps 20
```

More sweeps = better gradient estimates but slower training.

#### `--nchains`
Number of parallel Markov chains for sampling. Default: `10000`

```bash
--nchains 5000   # Use fewer chains (less memory)
--nchains 20000  # Use more chains (better statistics)
```

**Guidelines**:
- More chains = better statistics, more GPU memory
- Fewer chains = faster, less memory
- Typical range: 5000-20000

### Optimization Parameters

#### `--lr`
Learning rate for gradient descent. Default: `0.01`

```bash
--lr 0.005  # Slower, more stable
--lr 0.02   # Faster, may be less stable
```

**Guidelines**:
- Start with 0.01
- Decrease if training is unstable
- Increase for faster convergence (if stable)

### Regularization

#### `--pseudocount`
Pseudocount for smoothing empirical frequencies. Default: `None` (automatic: 1/Meff)

```bash
--pseudocount 0.5
```

Acts as regularization to prevent overfitting.

## Sequence Processing

### Alphabet

#### `--alphabet`
Sequence alphabet/encoding. Default: `protein`

```bash
--alphabet protein  # Standard 20 amino acids + gap
--alphabet rna      # RNA: ACGU + gap
--alphabet dna      # DNA: ACGT + gap
--alphabet "ACDEFG" # Custom alphabet
```

**Built-in alphabets**:
- `protein`: `ACDEFGHIKLMNPQRSTVWY-`
- `rna`: `ACGU-`
- `dna`: `ACGT-`

Custom alphabets must include all characters present in the alignment.

### Sequence Reweighting

Sequence reweighting reduces phylogenetic bias in the dataset.

#### `--weights` / `-w`
Path to pre-computed sequence weights file. Optional.

```bash
--weights sequence_weights.txt
```

File format: one weight per line, same order as sequences in FASTA.

#### `--clustering_seqid`
Sequence identity threshold for automatic reweighting. Default: `0.8` (80%)

```bash
--clustering_seqid 0.9  # Cluster at 90% identity
```

Sequences with identity ≥ threshold are clustered, and cluster members share weight.

#### `--no_reweighting`
Disable automatic sequence reweighting.

```bash
--no_reweighting
```

Use if your alignment is already diversity-corrected or unweighted analysis is desired.

## Checkpoint Options

### Checkpoint Strategy

#### `--checkpoints`
Checkpoint strategy for saving model state. Default: `linear`

```bash
--checkpoints linear      # Save every N epochs
--checkpoints acceptance  # Save when acceptance rate changes
```

For edDCA, checkpoints are also triggered at pre-defined density thresholds for entropy computation.

#### `--target_acc_rate`
Target acceptance rate for acceptance-based checkpoints. Default: `0.5`

```bash
--target_acc_rate 0.6
```

Only used when `--checkpoints acceptance`.

## Experiment Tracking

### Weights & Biases

#### `--wandb`
Enable Weights & Biases logging for experiment tracking.

```bash
--wandb
```

Requires W&B account and login (`wandb login`).

Logs:
- Training metrics (Pearson, entropy, density)
- System metrics (GPU usage, time)
- Model parameters and outputs

## Computational Settings

### Device Selection

#### `--device`
Computation device. Default: `cuda`

```bash
--device cuda  # Use GPU
--device cpu   # Use CPU
```

GPU is strongly recommended for large datasets.

### Data Type

#### `--dtype`
Numerical precision. Default: `float32`

```bash
--dtype float32  # Standard precision (faster)
--dtype float64  # Double precision (more accurate)
```

`float32` is usually sufficient and faster.

## Advanced Options

### Restoration and Continuation

#### `--path_params` / `-p`
Path to existing model parameters to restore training.

```bash
--path_params previous_run/params.dat
```

#### `--path_chains` / `-c`
Path to existing chains for restoration.

```bash
--path_chains previous_run/chains.fasta
```

**Use case**: Continue training from a checkpoint or use a pre-trained bmDCA as starting point.

### Test Set Evaluation

#### `--test`
Path to test set MSA for evaluation during training.

```bash
--test test_sequences.fasta
```

Test log-likelihood will be computed and logged (requires additional computation).

### Random Seed

#### `--seed`
Random seed for reproducibility. Default: `0`

```bash
--seed 42
```

Use different seeds for multiple independent runs.

## Complete Example Commands

### Example 1: Basic Training

Train a sparse edDCA model with default settings:

```bash
highentDCA train \
    --data protein_family.fasta \
    --output results/protein_edDCA \
    --model edDCA \
    --density 0.02 \
    --seed 42
```

### Example 2: High-Accuracy Training

Train with stricter convergence and more sampling:

```bash
highentDCA train \
    --data protein_family.fasta \
    --output results/high_accuracy \
    --model edDCA \
    --density 0.03 \
    --target 0.98 \
    --nchains 20000 \
    --nsweeps 20 \
    --nsweeps_dec 20 \
    --lr 0.005 \
    --seed 42
```

### Example 3: Fast Exploration

Quick training for exploratory analysis:

```bash
highentDCA train \
    --data protein_family.fasta \
    --output results/fast_run \
    --model edDCA \
    --density 0.05 \
    --drate 0.02 \
    --nchains 5000 \
    --nsweeps 5 \
    --nsweeps_dec 5 \
    --target 0.90 \
    --nsteps 50 \
    --nsweeps_step 50
```

### Example 4: RNA Family

Train on RNA alignment with custom parameters:

```bash
highentDCA train \
    --data rna_family.fasta \
    --output results/rna_edDCA \
    --model edDCA \
    --alphabet rna \
    --density 0.08 \
    --drate 0.01 \
    --clustering_seqid 0.85 \
    --seed 123
```

### Example 5: With Weights & Biases Tracking

Track experiments with W&B:

```bash
highentDCA train \
    --data protein_family.fasta \
    --output results/wandb_run \
    --model edDCA \
    --density 0.02 \
    --label experiment_001 \
    --wandb \
    --seed 42
```

### Example 6: Continue from bmDCA

Start from pre-trained bmDCA model:

```bash
# First train bmDCA (or use existing)
highentDCA train \
    --data protein_family.fasta \
    --output bmdca_model \
    --model bmDCA \
    --target 0.95

# Then decimate it
highentDCA train \
    --data protein_family.fasta \
    --output eddca_model \
    --model edDCA \
    --path_params bmdca_model/params.dat \
    --path_chains bmdca_model/chains.fasta \
    --density 0.02
```

## Analyzing Results

### Reading Training Logs

The log file (`adabmDCA_highent.log`) contains training progress:

```bash
cat results/adabmDCA_highent.log
```

Example output:
```
Epochs     Pearson    Entropy    Density    Time      
0          0.950      125.456    1.000      0.000
50         0.955      120.123    0.587      120.450
100        0.953      115.678    0.359      250.890
```

### Extracting Entropy Values

Entropy vs. density data is saved in `entropy_values.txt`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results/entropy_values.txt', sep='\t')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Density'], df['Entropy'], 'o-', linewidth=2, markersize=8)
plt.xlabel('Coupling Density', fontsize=14)
plt.ylabel('Model Entropy', fontsize=14)
plt.title('Entropy Evolution During Decimation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('entropy_evolution.png', dpi=300)
plt.show()
```

### Loading Model Parameters

Load trained parameters for analysis:

```python
from adabmDCA.io import load_params
import torch

# Load parameters
params, tokens, L, q = load_params('results/params.dat')

# Access fields and couplings
fields = params['bias']  # Shape: (L, q)
couplings = params['coupling_matrix']  # Shape: (L, q, L, q)

print(f"Sequence length: {L}")
print(f"Alphabet size: {q}")
print(f"Number of non-zero couplings: {(couplings != 0).sum().item()}")
```

### Visualizing Coupling Matrix

Plot the sparse coupling matrix:

```python
import numpy as np
import matplotlib.pyplot as plt
from adabmDCA.io import load_params

# Load parameters
params, tokens, L, q = load_params('results/params.dat')
couplings = params['coupling_matrix'].cpu().numpy()

# Compute Frobenius norm for each coupling
coupling_strength = np.linalg.norm(couplings.reshape(L, q, L, q), axis=(1, 3))

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(coupling_strength, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Coupling Strength')
plt.xlabel('Position j')
plt.ylabel('Position i')
plt.title('Sparse Coupling Matrix (edDCA)')
plt.tight_layout()
plt.savefig('coupling_matrix.png', dpi=300)
plt.show()
```

## Troubleshooting

### Training doesn't converge

**Solutions**:
- Decrease learning rate: `--lr 0.005`
- Increase sweeps: `--nsweeps 20 --nsweeps_dec 20`
- Relax target: `--target 0.93`
- Check data quality: remove fragments, check alignment

### Out of memory errors

**Solutions**:
- Reduce number of chains: `--nchains 5000`
- Use float32: `--dtype float32` (default)
- Use smaller batch size for GPU
- Monitor with: `nvidia-smi -l 1`

### Training is too slow

**Solutions**:
- Ensure GPU is being used: check `--device cuda`
- Reduce accuracy requirements: `--target 0.93`
- Use fewer chains: `--nchains 8000`
- Increase decimation rate: `--drate 0.02`
- Reduce entropy computation accuracy

### Entropy computation fails

**Solutions**:
- Increase equilibration: `--nsweeps_zero 100 --nsweeps_theta 100`
- Reduce theta_max: `--theta_max 3.0`
- Check data quality and convergence

## Best Practices

1. **Start with defaults**: Begin with default parameters and adjust based on results
2. **Monitor convergence**: Check Pearson correlation in logs
3. **Use appropriate density**: 2-5% for most protein families
4. **Save checkpoints**: Use `--label` to organize multiple runs
5. **Validate results**: Check entropy evolution makes sense
6. **Use test sets**: Evaluate generalization with `--test`
7. **Set seeds**: Use `--seed` for reproducibility
8. **Track experiments**: Use `--wandb` for complex parameter searches

## Next Steps

- **[API Reference](api/README.md)**: Use highentDCA in Python scripts
- **[Checkpoint Documentation](api/checkpoint.md)**: Understand checkpoint strategies
- **[edDCA Model Documentation](api/models.edDCA.md)**: Deep dive into the algorithm

## Getting Help

If you encounter issues:

- Check the log file for error messages
- Review [GitHub Issues](https://github.com/robertonetti/highentropyDCA/issues)
- Contact: robertonetti3@gmail.com
