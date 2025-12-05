# Welcome to highentDCA Documentation

**highentDCA** is a Python package for training decimation Direct Coupling Analysis (edDCA) models on biological sequence data. This package extends the `adabmDCA` framework to enable efficient training of sparse Potts models while tracking their entropy evolution during the decimation process.

!!! info "About this Documentation"
    This documentation provides a comprehensive guide to using highentDCA for training sparse DCA models with entropy tracking. It complements the [adabmDCA documentation](https://spqb.github.io/adabmDCApy/) and focuses specifically on the entropy decimation features.

## What is highentDCA?

highentDCA implements an algorithm, which systematically reduces a fully-connected Potts model to a sparse network by iteratively pruning weak couplings, identifying the optimal sparsity level that maximizes model entropy while preserving predictive accuracy. This approach is particularly useful for:

- **Reducing computational complexity** by identifying essential interactions
- **Maximizing model entropy** by tracking entropy changes throughout the decimation process

Additionally, this method is valuable for:

- Understanding which interactions are essential for capturing sequence statistics
- Building interpretable sparse models for protein families

## Key Features

### 🔬 Parameter Decimation DCA

The main feature of highentDCA is the ability to train edDCA models that:

- Start from a converged bmDCA model (or train one automatically)
- Iteratively remove the least important couplings based on empirical statistics
- Re-equilibrate after each decimation step to maintain accuracy
- Track coupling density, Pearson correlation, and model entropy throughout the process

### 📊 Thermodynamic Integration

At pre-defined density checkpoints, highentDCA computes model entropy using thermodynamic integration:

- Introduces a bias parameter θ towards a target sequence
- Integrates average sequence identity from θ=0 to θ_max
- Uses careful equilibration to ensure accurate entropy estimates
- Saves entropy values for downstream analysis

### 💾 Flexible Checkpointing

Multiple checkpoint strategies for saving model state:

- **Density-based checkpointing**: Save at specific coupling densities
- Automatic saving of model parameters, chains, and statistics
- Optional integration with Weights & Biases for experiment tracking

### 🚀 GPU Acceleration

Built on PyTorch for efficient computation:

- GPU-accelerated training and sampling
- Efficient parallel Markov chain Monte Carlo
- Automatic device management (CUDA/CPU)
- Support for mixed precision (float32/float64)

## How edDCA Works

The entropy decimation algorithm follows these steps:

1. **Initialization**: Start with a converged bmDCA model or train one
2. **Decimation**: Remove a fraction of couplings with smallest empirical two-point correlations
3. **Equilibration**: Run MCMC to equilibrate chains on the decimated graph
4. **Re-convergence**: Perform gradient descent to match data statistics
5. **Entropy Computation**: At checkpoints, compute entropy via thermodynamic integration
6. **Iteration**: Repeat steps 2-5 until target density is reached

![Decimation Process](images/decimation_workflow.png)

## Key Advantages

### Sparsity with Accuracy

edDCA achieves high sparsity while maintaining model accuracy:

- Typical models retain only 2-5% of couplings
- Pearson correlation with data statistics remains >0.95
- Essential interactions are preserved

## Use Cases

### Maximize model entropy

to be completed...


```

## Getting Started

New to highentDCA? Follow these steps:

1. **[Installation](installation.md)**: Set up the package and dependencies
2. **[Quick Start](usage.md#quick-start)**: Train your first edDCA model
3. **[CLI Reference](usage.md#command-line-interface)**: Explore all available options
4. **[API Documentation](api/README.md)**: Use highentDCA in Python scripts
5. **[Examples](usage.md#examples)**: Learn from practical examples



highentDCA is built on top of adabmDCA and shares:

- **Data formats**: Compatible FASTA input and parameter formats
- **Sampling methods**: Same Gibbs and Metropolis samplers
- **Statistics functions**: Identical frequency and correlation computations
- **Utilities**: Common helper functions for encoding, I/O, etc.

You can use adabmDCA models as starting points for edDCA training, and edDCA outputs are compatible with adabmDCA analysis tools.

## Technical Requirements

### Software Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.1.0 (with CUDA recommended)
- adabmDCA == 0.5.0
- NumPy, Pandas, Matplotlib, BioPython

### Hardware Recommendations

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum 4GB VRAM for small datasets
  - 8GB+ VRAM for large protein families
- **CPU**: Multi-core processor for data preprocessing
- **RAM**: 8GB+ depending on dataset size

### Dataset Requirements

- Multiple sequence alignment in FASTA format
- Minimum ~1000 sequences (more is better)
- Quality-controlled alignment (gaps, truncations handled)
- Compatible alphabets: protein, RNA, DNA, or custom

## Support and Community

### Getting Help

- **Documentation**: Read the full documentation in the `docs/` folder
- **Issues**: Report bugs on [GitHub Issues](https://github.com/robertonetti/highentropyDCA/issues)
- **Questions**: Contact robertonetti3@gmail.com

### Contributing

Contributions are welcome! Areas for improvement:

## Related Resources

### Papers

- **Entropy Decimation**: [Barrat-Charlaix et al., 2021](https://doi.org/10.1103/PhysRevE.104.024407)
- **Adaptive bmDCA**: [Muntoni et al., 2021](https://doi.org/10.1186/s12859-021-04441-9)
- **DCA for Contact Prediction**: [Ekeberg et al., 2013](https://doi.org/10.1103/PhysRevE.87.012707)

### Software

- **adabmDCApy**: [Python implementation](https://github.com/spqb/adabmDCApy)
- **adabmDCA.jl**: [Julia implementation](https://github.com/spqb/adabmDCA.jl)
- **adabmDCAc**: [C++ implementation](https://github.com/spqb/adabmDCAc)

### Tutorials

- [adabmDCA Colab Notebook](https://colab.research.google.com/drive/1l5e1W8pk4cB92JAlBElLzpkEk6Hdjk7B?usp=sharing)
- [DCA Tutorial](https://spqb.github.io/adabmDCApy/)

## License

highentDCA is released under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

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

---

**Ready to get started?** Head to the [Installation Guide](installation.md) →
