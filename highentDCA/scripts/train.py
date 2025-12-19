from pathlib import Path
import importlib
import argparse

import numpy as np
import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.fasta import get_tokens
from adabmDCA.io import load_chains, load_params
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from adabmDCA.utils import init_chains, init_parameters, get_device, get_dtype
from adabmDCA.sampling import get_sampler
from adabmDCA.functional import one_hot
from adabmDCA.checkpoint import get_checkpoint

from highentDCA.parser import add_args_train


def create_parser():
    """Create and configure the argument parser for DCA model training.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all training parameters.
    """
    parser = argparse.ArgumentParser(description='Train a DCA model.')
    parser = add_args_train(parser)
    
    return parser


def main():
    """Main function to train a DCA model.
    
    This function orchestrates the complete training pipeline:
    - Parses command-line arguments
    - Loads and processes training (and optionally test) datasets
    - Initializes or loads model parameters and chains
    - Configures the sampler and checkpoint strategy
    - Executes the training procedure
    """
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert checkpt_steps from string to list of floats
    if args.checkpt_steps is not None:
        args.checkpt_steps = [float(x) for x in args.checkpt_steps.split()]
    
    print("\n" + "="*70)
    print(f"  TRAINING {args.model.upper()} MODEL")
    print("="*70 + "\n")
    
    # Set the device and data type
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    # Display configuration
    template = "{0:<30} {1:<50}"
    print("Configuration:")
    print(template.format("  Input MSA:", str(args.data)))
    print(template.format("  Output folder:", str(args.output)))
    print(template.format("  Alphabet:", args.alphabet))
    print(template.format("  Learning rate:", args.lr))
    print(template.format("  Number of sweeps:", args.nsweeps))
    print(template.format("  Sampler:", args.sampler))
    print(template.format("  Target Pearson Cij:", args.target))
    if args.pseudocount is not None:
        print(template.format("  Pseudocount:", args.pseudocount))
    print(template.format("  Random seed:", args.seed))
    print(template.format("  Data type:", args.dtype))
    print("\n")
    
    # Validate input file paths
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    if args.test is not None:
        if not Path(args.test).exists():
            raise FileNotFoundError(f"Test file {args.test} not found.")
    
    # Create output directory structure
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    # Configure output file paths
    if args.label is not None:
        file_paths = {
            "log": folder / Path(f"{args.label}.log"),
            "params": folder / Path(f"{args.label}_params.dat"),
            "chains": folder / Path(f"{args.label}_chains.fasta")
        }
    else:
        file_paths = {
            "log": folder / Path(f"adabmDCA.log"),
            "params": folder / Path(f"params.dat"),
            "chains": folder / Path(f"chains.fasta")
        }
    
    # Import dataset
    print("→ Importing dataset...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
    )
    
    # Import the test dataset if provided
    if args.test is not None:
        print("→ Importing test dataset...")
        test_dataset = DatasetDCA(
            path_data=args.test,
            path_weights=None,
            alphabet=args.alphabet,
            clustering_th=args.clustering_seqid,
            no_reweighting=args.no_reweighting,
            device=device,
            dtype=dtype,
        )
        
        # Compute test statistics
        test_oh = one_hot(test_dataset.data, num_classes=dataset.get_num_states()).to(dtype)
        pseudocount_test = 1. / test_dataset.get_effective_size()
        fi_test = get_freq_single_point(data=test_oh, weights=test_dataset.weights, pseudo_count=pseudocount_test)
        fij_test = get_freq_two_points(data=test_oh, weights=test_dataset.weights, pseudo_count=pseudocount_test)
    else:
        fi_test = None
        fij_test = None
    
    # Load the DCA model module
    DCA_model = importlib.import_module(f"highentDCA.models.{args.model}")
    tokens = get_tokens(args.alphabet)
    
    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = folder / f"{args.label}_weights.dat"
        else:
            path_weights = folder / "weights.dat"
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"  ✓ Weights saved: {path_weights}")
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Shuffle the dataset
    dataset.shuffle()
    
    # Extract dataset dimensions and compute statistics
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"  ℹ Pseudocount automatically set to {args.pseudocount:.6f}")
    
    # Compute target frequencies from training data
    data_oh = one_hot(dataset.data, num_classes=q).to(dtype)
    fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    
    # Initialize or load model parameters and interaction mask
    if args.path_params:
        print("→ Loading parameters...")
        params = load_params(fname=args.path_params, tokens=tokens, device=device, dtype=dtype)
        mask = torch.zeros(size=(L, q, L, q), dtype=torch.bool, device=device)
        mask[torch.nonzero(params["coupling_matrix"])] = 1
    else:
        params = init_parameters(fi=fi_target)
        
        if args.model in ["bmDCA", "edDCA"]:
            # Fully connected mask (excluding self-interactions)
            mask = torch.ones(size=(L, q, L, q), dtype=torch.bool, device=device)
            mask[torch.arange(L), :, torch.arange(L), :] = 0
        else:
            # Empty mask (for sparse models)
            mask = torch.zeros(size=(L, q, L, q), device=device, dtype=torch.bool)
    
    # Initialize or load Markov chains
    if args.path_chains:
        print("→ Loading chains...")
        chains, log_weights = load_chains(fname=args.path_chains, tokens=dataset.tokens, load_weights=True)
        chains = one_hot(
            torch.tensor(chains, device=device),
            num_classes=q,
        ).to(dtype)
        log_weights = torch.tensor(log_weights, device=device, dtype=dtype)
        args.nchains = chains.shape[0]
        print(f"  ✓ Loaded {args.nchains} chains")
    else:
        print(f"  ℹ Initializing {args.nchains} chains")
        chains = init_chains(num_chains=args.nchains, L=L, q=q, fi=fi_target, device=device, dtype=dtype)
        log_weights = torch.zeros(size=(args.nchains,), device=device, dtype=dtype)
    
    # Configure sampler and checkpoint strategy
    sampler = get_sampler(args.sampler)
    print("\n")
    
    checkpoint = get_checkpoint(args.checkpoints)(
        file_paths=file_paths,
        tokens=tokens,
        args=args,
        params=params,
        chains=chains,
        max_epochs=args.nepochs,
        target_acc_rate=args.target_acc_rate,
        use_wandb=args.wandb,
    )
    
    # Execute training
    DCA_model.fit(
        sampler=sampler,
        fij_target=fij_target,
        fi_target=fi_target,
        fi_test=fi_test,
        fij_test=fij_test,
        params=params,
        mask=mask,
        chains=chains,
        log_weights=log_weights,
        tokens=tokens,
        target_pearson=args.target,
        pseudo_count=args.pseudocount,
        nsweeps=args.nsweeps,
        nepochs=args.nepochs,
        lr=args.lr,
        drate=args.drate,
        target_density=args.density,
        checkpoint=checkpoint,
        args=args,
    )


if __name__ == "__main__":
    main()