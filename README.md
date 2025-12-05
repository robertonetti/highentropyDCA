# highentDCA - High Entropy Direct Coupling Analysis

**highentDCA** is a Python package for training entropy-decimated Direct Coupling Analysis (edDCA) models on biological sequence data. This package extends the `adabmDCA` framework to enable efficient training of sparse Potts models while tracking their entropy evolution during the decimation process.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://robertonetti.github.io/highentropyDCA/)

## Overview

highentDCA implements the **entropy-decimation DCA** algorithm, which systematically reduces a fully-connected Potts model to a sparse network by iteratively pruning weak couplings, identifying the optimal sparsity level that maximizes model entropy while preserving predictive accuracy. This approach is particularly useful for:

- **Reducing computational complexity** by identifying essential interactions
- **Maximizing model entropy** by tracking entropy changes throughout the decimation process
- **Building interpretable sparse models** for biological sequence data

### Key Features

- 🔬 **Parameter decimation DCA**: Progressively sparse models with entropy tracking
- 📊 **Thermodynamic Integration**: Compute model entropy at key decimation checkpoints
- 🚀 **GPU-accelerated**: Efficient training leveraging PyTorch and CUDA
- 💾 **Flexible checkpointing**: Save models at critical density thresholds
- 📈 **Integrated with adabmDCA**: Built on top of the robust adabmDCA framework
- 🔧 **Command-line interface**: Easy-to-use CLI for training and analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.1.0 or higher (with CUDA support recommended)
- adabmDCA 0.5.0

### Install from source

```bash
git clone https://github.com/robertonetti/highentropyDCA.git
cd highentropyDCA
pip install .
```

### Dependencies

The package will automatically install the required dependencies:

- `adabmDCA==0.5.0` - Core DCA functionality
- `torch>=2.1.0` - Deep learning framework
- `numpy>=1.26.4` - Numerical computing
- `pandas>=2.2.2` - Data manipulation
- `matplotlib>=3.8.0` - Plotting
- `tqdm>=4.66.6` - Progress bars
- `wandb>=0.12.0` - Experiment tracking (optional)
- `biopython>=1.85` - Biological sequence handling

## Quick Start

### Training an edDCA Model

The simplest way to train an entropy-decimated DCA model is through the command-line interface:

```bash
highentDCA train \
    --data example_data/PF00072.fasta \
    --output DCA_model \
    --model edDCA \
    --density 0.02 \
    --drate 0.01 \
    --target 0.95 \
    --nsweeps 10 \
    --lr 0.01
```

This command will:
1. Load the multiple sequence alignment from `example_data/PF00072.fasta`
2. Train an edDCA model targeting 2% coupling density
3. Decimate 1% of remaining couplings at each step
4. Compute entropy at key density checkpoints
5. Save results in the `DCA_model` folder

### Key Parameters

- `--data`: Path to input MSA in FASTA format (required)
- `--model`: Model type - use `edDCA` for entropy decimation
- `--density`: Target coupling density (default: 0.02)
- `--drate`: Decimation rate - fraction of couplings to prune per step (default: 0.01)
- `--target`: Target Pearson correlation for convergence (default: 0.95)
- `--nsweeps`: Number of Monte Carlo sweeps per gradient update (default: 10)
- `--lr`: Learning rate (default: 0.01)

### Output Files

After training, the output folder will contain:

```
DCA_model/
├── params.dat                  # Final model parameters
├── chains.fasta               # Final Markov chains
├── adabmDCA_highent.log      # Training log with metrics
├── entropy_decimation/        # Checkpoints at key densities
│   ├── density_<checkpoint1>.fasta
│   ├── density_<checkpoint2>.fasta
│   ├── density_<checkpoint3>.fasta
│   └── ...
└── entropy_values.txt         # Entropy vs. density data
```

## Documentation

📚 **[View the full documentation here](https://robertonetti.github.io/highentropyDCA/)** 📚

The documentation includes:

- **[Installation Guide](https://robertonetti.github.io/highentropyDCA/highentDCA_installation/)**: Detailed installation instructions
- **[Usage Guide](https://robertonetti.github.io/highentropyDCA/highentDCA_usage/)**: Complete CLI reference and examples
- **[API Reference](https://robertonetti.github.io/highentropyDCA/api/highentDCA_overview/)**: Python API documentation
  - [Checkpoint Management](https://robertonetti.github.io/highentropyDCA/api/highentDCA_checkpoint/)
  - [Training Functions](https://robertonetti.github.io/highentropyDCA/api/highentDCA_training/)
  - [edDCA Model](https://robertonetti.github.io/highentropyDCA/api/highentDCA_models.edDCA/)
  - [Entropy Computation](https://robertonetti.github.io/highentropyDCA/api/highentDCA_entropy/)

## Core Concepts

### Entropy-Maximizing Parameter Decimation

The edDCA algorithm works by:

1. **Starting from a converged bmDCA model** (or training one if needed)
2. **Iteratively decimating** the least important couplings based on two-point statistics
3. **Re-equilibrating** the model after each decimation step
4. **Computing entropy** at pre-defined density checkpoints using thermodynamic integration
5. **Identifying the optimal sparsity** where model entropy is maximized while preserving statistical accuracy

### Thermodynamic Integration

Entropy is computed using thermodynamic integration by:

- Introducing a bias parameter θ towards a target sequence
- Integrating the average sequence identity from θ=0 to θ=θ_max
- Using multiple equilibration steps to ensure accurate sampling

This provides insights into the model's statistical mechanics properties during decimation.

## Example Workflow

### 1. Prepare your data

```bash
# Your MSA should be in FASTA format
head -n 4 example_data/PF00072.fasta
>seq1
MVKFKYKGEEKEVDISKIKKVWRVGKMISFTYDEGGGKTGRGAVSEKDAPKELLQMLEKQKK
>seq2
MVKFKYKGQEKEVDTSKIKKVWRVGKMVSFTYDEGGGKTGRGAVSEKDAPKELLQMLEKQKK
```

### 2. Train the model

```bash
highentDCA train \
    --data example_data/PF00072.fasta \
    --output results/PF00072_edDCA \
    --model edDCA \
    --density 0.05 \
    --drate 0.01 \
    --alphabet protein \
    --nchains 10000 \
    --seed 42
```

### 3. Monitor training

The training progress will show:
- Current decimation step
- Coupling density
- Pearson correlation with data statistics
- Entropy values at checkpoints


## Advanced Usage

### Custom Alphabet

For non-standard sequences (e.g., custom amino acids):

```bash
highentDCA train --data mydata.fasta --alphabet "ACDEFGHIKLMNPQRSTVWYX"
```

### Sequence Reweighting

Control phylogenetic bias with sequence clustering:

```bash
highentDCA train \
    --data mydata.fasta \
    --clustering_seqid 0.8  # Cluster sequences at 80% identity
```

### Integration with Weights & Biases

Track experiments with W&B:

```bash
highentDCA train --data mydata.fasta --wandb
```

### Entropy Computation Parameters

Fine-tune thermodynamic integration:

```bash
highentDCA train \
    --data mydata.fasta \
    --theta_max 5.0 \
    --nsteps 100 \
    --nsweeps_step 100 \
    --nsweeps_theta 50 \
    --nsweeps_zero 50
```

## Citation

If you use highentDCA in your research, please cite:

```bibtex
@misc{highentDCA2025,
  author = {Netti, Roberto and Calvanese, Francesco and Hinds, Emily and Ranganathan, Rama and Zamponi, Francesco and Weigt, Martin},
  title = {highentDCA: Entropy-maximizing parameter decimation for Direct Coupling Analysis},
  year = {2025},
  note = {Manuscript in preparation},
  url = {https://github.com/robertonetti/highentropyDCA}
}
```

And the related adabmDCA package:

```bibtex
@Inbook{Rosset2026,
  author="Rosset, Lorenzo and Netti, Roberto and Muntoni, Anna Paola and Weigt, Martin and Zamponi, Francesco",
  editor="Khan, Shahid M. and Pazos, Florencio",
  title="adabmDCA 2.0---A Flexible but Easy-to-Use Package for Direct Coupling Analysis",
  bookTitle="Protein Evolution: Methods and Protocols",
  year="2026",
  publisher="Springer US",
  address="New York, NY",
  pages="83--104",
  isbn="978-1-0716-4828-5",
  doi="10.1007/978-1-0716-4828-5_6",
  url="https://doi.org/10.1007/978-1-0716-4828-5_6"
}

@article{muntoni2021adabmDCA,
  title={adabmDCA: adaptive Boltzmann machine learning for biological sequences},
  author={Muntoni, Anna Paola and Pagnani, Andrea and Weigt, Martin and Zamponi, Francesco},
  journal={BMC bioinformatics},
  volume={22},
  pages={1--29},
  year={2021},
  publisher={Springer}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [adabmDCA](https://github.com/spqb/adabmDCApy)
- Inspired by the entropy decimation algorithm from [Barrat-Charlaix et al., 2021](https://doi.org/10.1103/PhysRevE.104.024407)

## Contact

- **Author**: Roberto Netti
- **Email**: robertonetti3@gmail.com
- **GitHub**: [robertonetti/highentropyDCA](https://github.com/robertonetti/highentropyDCA)

## Related Projects

- [adabmDCApy](https://github.com/spqb/adabmDCApy) - Python implementation of adaptive bmDCA
- [adabmDCA.jl](https://github.com/spqb/adabmDCA.jl) - Julia implementation for multi-core CPUs
- [adabmDCAc](https://github.com/spqb/adabmDCAc) - C++ implementation for single-core CPUs
