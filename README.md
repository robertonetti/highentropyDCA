# highentDCA

`highentDCA` is a Python implementation of the entropy-based decimation
procedure used to identify maximum-entropy Direct Coupling Analysis models
(meDCA) along a sparse Potts-model training trajectory. The command-line
interface exposes a single training workflow, `highentDCA train`, which always
runs the edDCA decimation procedure.

The package is intended for research workflows in which a multiple sequence
alignment (MSA) of homologous biological sequences is used to infer a
generative model of the corresponding sequence family. It builds on
[`adabmDCA`](https://github.com/spqb/adabmDCApy) for Boltzmann-machine
training, Monte Carlo sampling, and model I/O, and adds the decimation and
entropy-monitoring steps needed to locate high-entropy sparse models.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://robertonetti.github.io/highentropyDCA/)

## Scientific Context

Direct Coupling Analysis (DCA) models a family of aligned protein, RNA, or DNA
sequences with a maximum-entropy probability distribution constrained by
empirical one-site and two-site frequencies. In statistical-physics notation,
this distribution is a Potts model,

```text
P(A) = exp[-H(A)] / Z
H(A) = - sum_i h_i(a_i) - sum_{i<j} J_ij(a_i, a_j),
```

where `A = (a_1, ..., a_L)` is an aligned sequence, `h_i` are site fields,
`J_ij` are pairwise couplings, and `Z` is the partition function. The fields
and couplings are fitted so that samples drawn from the model reproduce the
statistics of the input MSA.

Fully connected bmDCA models include all pairwise couplings. This can reproduce
the target statistics accurately, but the number of parameters can be large
relative to the effective number of sequences in the MSA. The work motivating
this package studies how model sparsity affects entropy, generalization, and
the size of the sequence space sampled by the model.

## Method

`highentDCA train` implements an entropy-decimation workflow:

1. The preferred input is a previously trained fully connected bmDCA model,
   provided as a parameter file together with its persistent Monte Carlo chains.
   These files define the initial dense model from which decimation starts.
2. If no parameter and chain files are provided, `highentDCA` initializes a fully
   connected bmDCA model from the MSA and trains it until the Pearson correlation
   between empirical and model two-site connected correlations reaches the
   requested target value.
3. Couplings are progressively removed by decimation. At each decimation step,
   the least statistically significant active couplings are set to zero.
4. The remaining parameters are re-equilibrated and, when needed, retrained on
   the surviving graph to maintain the requested statistical fit.
5. At prescribed coupling-density checkpoints, the model entropy is estimated
   by thermodynamic integration.
6. The maximum-entropy model along this decimation trajectory is identified
   a posteriori from the entropy-versus-density curve.

The package does not expose a user-selectable model argument. The training
script is specialized for edDCA: it starts from a fully connected graph, removes
couplings by decimation, estimates entropy at selected densities, and stores the
trajectory. The term `meDCA` denotes the checkpoint with maximal entropy along
this edDCA trajectory. It should not be interpreted as a proof of a global
entropy maximum over all possible DCA parameterizations.

## Installation

The package requires Python 3.10 or newer.

```bash
git clone https://github.com/robertonetti/highentropyDCA.git
cd highentropyDCA
pip install .
```

Main dependencies are installed through `setup.py`:

- `adabmDCA==0.5.0`
- `torch>=2.1.0`
- `numpy>=1.26.4`
- `pandas>=2.2.2`
- `matplotlib>=3.8.0`
- `biopython>=1.85`
- `tqdm>=4.66.6`
- `wandb>=0.12.0`

CUDA is recommended for large alignments or long entropy-integration runs, but
CPU execution is available through `--device cpu`.

## Command-Line Usage

Training is run through the `train` command. The intended use is to provide an
MSA, a fully connected bmDCA parameter file, and the corresponding persistent
chains:

```bash
highentDCA train \
    --data example_data/PF00014_mgap6.fasta \
    --path_params example_data/PF00014/highentDCA/params.dat \
    --path_chains example_data/PF00014/highentDCA/chains.fasta \
    --output example_data/PF00014/highentDCA \
    --alphabet protein \
    --density 0.02 \
    --target 0.95
```

In this mode, `highentDCA` uses the supplied fully connected bmDCA model as the
starting point and begins the edDCA decimation trajectory from that state.

If a dense bmDCA model is not available, the only required input is the MSA:

```bash
highentDCA train --data example_data/PF00072.fasta
```

The code then initializes a fully connected model, trains it to the target
two-site Pearson correlation, and proceeds with decimation. In this case, one
usually specifies the output directory, alphabet, target density, and Monte
Carlo settings:

```bash
highentDCA train \
    --data example_data/PF00072.fasta \
    --output results/PF00072_edDCA \
    --alphabet protein \
    --target 0.95 \
    --density 0.02 \
    --drate 0.001 \
    --nchains 10000 \
    --nsweeps 10 \
    --nsweeps_dec 100 \
    --seed 42
```

The input MSA must be in FASTA format. Sequence weights are computed by default
using sequence-identity clustering; alternatively, precomputed weights can be
provided with `--weights`.

There is no `--model` option. The package currently trains edDCA models only.

### Important Arguments

| Argument | Meaning |
| --- | --- |
| `--data` | Input MSA in FASTA format. |
| `--output` | Directory where parameters, chains, logs, and entropy files are written. |
| `--alphabet` | `protein`, `rna`, `dna`, or a custom alphabet string. |
| `--target` | Pearson-correlation target for two-site statistics before decimation. |
| `--density` | Final target density of active coupling parameters. |
| `--drate` | Fraction of currently active couplings removed at each decimation step. |
| `--nchains` | Number of persistent Monte Carlo chains. |
| `--nsweeps` | Monte Carlo sweeps per update during initial training. |
| `--nsweeps_dec` | Monte Carlo sweeps used after each decimation step. |
| `--pseudocount` | Frequency regularization. If omitted, it is set to `1 / Meff`. |
| `--checkpt_steps` | Optional space-separated list of density checkpoints for entropy estimates. |
| `--theta_max`, `--nsteps` | Thermodynamic-integration range and discretization. |
| `--device` | Computation device, for example `cuda` or `cpu`. |
| `--path_params`, `--path_chains` | Fully connected bmDCA parameters and corresponding chains used as the decimation starting point. |

If `--checkpt_steps` is not provided, density checkpoints are generated
automatically between the dense model and the requested final density. For a
more controlled entropy curve, provide explicit checkpoints:

```bash
highentDCA train \
    --data data/aroq.fasta \
    --output results/aroq_decimation \
    --density 0.02 \
    --drate 0.001 \
    --checkpt_steps "0.50 0.30 0.20 0.15 0.125 0.10 0.07 0.05 0.03 0.02"
```

## Output

A typical run produces:

```text
results/PF00072_edDCA/
├── params.dat
├── chains.fasta
├── weights.dat
├── adabmDCA.log
├── adabmDCA_highent.log
├── params_density_<density>.dat
├── chains_density_<density>.fasta
└── entropy_decimation/
    ├── entropy_values.txt
    ├── density_<density>.log
    └── ...
```

`entropy_values.txt` contains the entropy estimates as a function of coupling
density. The meDCA checkpoint is selected as the density with the largest
reported entropy, provided that the corresponding model still satisfies the
chosen statistical-fit criterion.

The density-labelled parameter and chain files are the saved models along the
decimation path. They can be used for subsequent sampling, scoring, or
comparison with other DCA models using compatible `adabmDCA` utilities.

## Practical Notes

- The entropy computation is often the most expensive part of the workflow.
  Increase `--nsteps`, `--nsweeps_step`, `--nsweeps_theta`, and
  `--nsweeps_zero` for more accurate estimates.
- Smaller `--drate` values give a finer decimation trajectory at the cost of
  longer runs.
- The Pearson target monitors agreement between empirical and model two-site
  statistics. The final choice of meDCA should consider both this fit and the
  entropy curve.
- For reproducible runs, set `--seed` and record the full command line together
  with the output logs.

## Citation

If you use this package, please cite the associated work:

```bibtex
@article{netti2026sparse,
  title = {Sparse generative models of protein sequences sample larger spaces at comparable experimental success rates},
  author = {Netti, Roberto and Hinds, Emily and Calvanese, Francesco and Ranganathan, Rama and Weigt, Martin and Zamponi, Francesco},
  year = {2026},
  note = {Manuscript}
}
```

Please also cite the underlying `adabmDCA` software:

```bibtex
@incollection{rosset2024adabmdca,
  title = {adabmDCA 2.0---A Flexible but Easy-to-Use Package for Direct Coupling Analysis},
  author = {Rosset, Lorenzo and Netti, Roberto and Muntoni, Anna Paola and Weigt, Martin and Zamponi, Francesco},
  booktitle = {Protein Evolution: Methods and Protocols},
  pages = {83--104},
  publisher = {Springer US},
  year = {2024},
  doi = {10.1007/978-1-0716-4828-5_6}
}

@article{muntoni2021adabmdca,
  title = {adabmDCA: adaptive Boltzmann machine learning for biological sequences},
  author = {Muntoni, Anna Paola and Pagnani, Andrea and Weigt, Martin and Zamponi, Francesco},
  journal = {BMC Bioinformatics},
  volume = {22},
  pages = {1--29},
  year = {2021}
}
```

## License

This project is distributed under the Apache License 2.0. See [LICENSE](LICENSE)
for the full text.

Apache 2.0 is a permissive open-source license that allows use, modification,
redistribution, and commercial use, while also including an explicit patent
grant. This makes it suitable for a research software package when the intent is
to encourage reuse with attribution and license preservation.

## Contact

- Roberto Netti
- GitHub: [robertonetti/highentropyDCA](https://github.com/robertonetti/highentropyDCA)
