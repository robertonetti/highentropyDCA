from pathlib import Path
from typing import Callable, Dict
import time

import torch

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.graph import decimate_graph, compute_density
from adabmDCA.statmech import compute_log_likelihood, _update_weights_AIS, compute_entropy, _compute_ess
from adabmDCA.utils import get_mask_save

from highentDCA.checkpoint import Checkpoint, DecCheckpoint
from highentDCA.training import train_graph
from highentDCA.scripts.entropy import compute_entropy

MAX_EPOCHS = 100_000


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
    *extra_args,
    **kwargs,
):
    """Fit an entropy-decimated DCA (edDCA) model on training data.
    
    This function trains a DCA model while progressively decimating the coupling matrix
    to achieve a target sparsity level. At key density checkpoints, the entropy is
    computed to track the model's evolution during the decimation process.
    
    Args:
        sampler (Callable): Sampling function for updating Markov chains.
        chains (torch.Tensor): Initial state of the Markov chains (shape: [nchains, L, q]).
        log_weights (torch.Tensor): Log-weights of chains for log-likelihood estimation.
        fi_target (torch.Tensor): Single-point frequencies from training data (shape: [L, q]).
        fij_target (torch.Tensor): Two-point frequencies from training data (shape: [L, q, L, q]).
        params (Dict[str, torch.Tensor]): Initial model parameters (fields and couplings).
        mask (torch.Tensor): Binary mask for the coupling matrix (shape: [L, q, L, q]).
        lr (float): Learning rate for parameter updates.
        nsweeps (int): Number of Monte Carlo sweeps per update.
        target_pearson (float): Target Pearson correlation coefficient for convergence.
        target_density (float): Target sparsity level for the coupling matrix.
        drate (float): Decimation rate (fraction of couplings to prune per step).
        checkpoint (Checkpoint): Checkpoint object for saving model state.
        fi_test (torch.Tensor | None, optional): Single-point frequencies from test data.
            Defaults to None.
        fij_test (torch.Tensor | None, optional): Two-point frequencies from test data.
            Defaults to None.
        args: Additional arguments containing training configuration.
        *extra_args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).
    
    Raises:
        ValueError: If input tensors have incorrect dimensions.
    """
    time_start = time.time()
    
    # Check the input sizes
    if fi_target.dim() != 2:
        raise ValueError("fi_target must be a 2D tensor")
    if fij_target.dim() != 4:
        raise ValueError("fij_target must be a 4D tensor")
    if chains.dim() != 3:
        raise ValueError("chains must be a 3D tensor")
    
    L, q = params["bias"].shape
    device = fi_target.device
    dtype = fi_target.dtype
    entropy_values = []
    

    # Configure decimation checkpoint paths
    file_paths_dec = checkpoint.file_paths.copy()
    parent_folder = checkpoint.file_paths["params"].parent
    file_paths_dec["log"] = parent_folder / Path(f"adabmDCA_highent.log")
    output = parent_folder / "entropy_decimation"
    # create output directory if it does not exist
    output.mkdir(parents=True, exist_ok=True)
    
    # Initialize decimation checkpoint
    checkpoint_dec = DecCheckpoint(
        file_paths=file_paths_dec,
        tokens=checkpoint.tokens,
        args=args,
        params=params,
        chains=chains,
        max_epochs=args.nepochs,
        target_acc_rate=args.target_acc_rate,
        use_wandb=args.wandb,
        target_density=target_density,
    )
    
    # Compute initial statistics and check convergence
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    _, pearson = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
    if pearson < target_pearson:
        with open(checkpoint_dec.file_paths["log"], "a") as f:
            f.write("Bringing the model to the convergence threshold...\n")
        print("\n→ Bringing the model to the convergence threshold...")
        chains, params, log_weights, _ = train_graph(
            sampler=sampler,
            chains=chains,
            log_weights=log_weights,
            mask=mask,
            fi=fi_target,
            fij=fij_target,
            fi_test=fi_test,
            fij_test=fij_test,
            params=params,
            nsweeps=nsweeps,
            lr=lr,
            max_epochs=MAX_EPOCHS,
            target_pearson=target_pearson,
            check_slope=False,
            checkpoint=checkpoint,
            progress_bar=False,
        )
    
    # Recompute statistics and save initial model
    _, pearson = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
    
    # Create mask for saving only upper-diagonal matrix
    mask_save = get_mask_save(L, q, device=device)
    checkpoint.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
    )
    
    # Start decimation process
    print(f"\n{'='*70}")
    print(f"  DECIMATION PROCESS (Target density: {target_density:.3f})")
    print(f"{'='*70}\n")
   
    # Template for wrinting the results
    template = "{0:<15} | {1:<20} | {2:<20} | {3:<20}" # | {4:<15}"
    density = compute_density(mask)
    count = 0
    checkpoint_dec.checkpt_interval = 10

    target_pearson_dec = target_pearson - 0.01

    # Initialize entropy file
    entropy_file = output / "entropy_values.txt"
    with open(entropy_file, "w") as f:
        f.write("Density\tEntropy\n")

    with open(checkpoint_dec.file_paths["log"], "a") as f:
        f.write(f"Computing entropy of initial model...\n")
    print("→ Computing entropy of initial model...")
    S = compute_entropy(
                params,
                args.data,
                sampler,
                chains,
                output,
                # round denssity to 3 decimal places for the
                label=f"{density:.3f}",
                tokens=checkpoint.tokens,
                nsweeps_zero=args.nsweeps_zero,
                nsweeps_theta=args.nsweeps_theta,
                theta_max=args.theta_max,
                nsteps=args.nsteps,
                nsweeps=args.nsweeps_step,
                device=device,
                dtype=dtype,
    )
    
    # Log initial state
    checkpoint_dec.header_log()
    checkpoint_dec.log(
        {
            "Epochs": 0,
            "Pearson": pearson,
            "Entropy": S,
            "Density": compute_density(mask),
            "Time": time.time() - time_start,
        }
    )
    
    # Store initial entropy value
    entropy_values.append((density, S))
    with open(entropy_file, "a") as f:
        f.write(f"{density:.6f}\t{S:.6f}\n")
    print(f"  ✓ Entropy at density {density:.3f}: {S:.6f}")
    
    # Main decimation loop
    while density > target_density:
        count += 1
        
        # Store the previous parameters
        prev_params = {key: value.clone() for key, value in params.items()}
        
        # Decimate the model
        params, mask = decimate_graph(
            pij=pij,
            params=params,
            mask=mask,
            drate=drate
        )
        
        # Update importance weights via Annealed Importance Sampling (AIS)
        log_weights = _update_weights_AIS(
            prev_params=prev_params,
            curr_params=params,
            chains=chains,
            log_weights=log_weights,
        )
        
        # Equilibrate chains on the decimated graph
        chains = sampler(
            chains=chains,
            params=params,
            nsweeps=args.nsweeps_dec,
        )
        
        # Compute statistics on equilibrated chains
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        
        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        if pearson < target_pearson_dec:
            with open(checkpoint_dec.file_paths["log"], "a") as f:
                f.write(f"Bringing the decimated model to the convergence threshold (Pearson: {pearson:.3f})...\n")
            print(f"  → Re-converging model (Current Pearson: {pearson:.4f})...")
        # Bring the model at convergence on the graph
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
        
        # Recompute statistics after convergence
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        
        pearson, slope = get_correlation_two_points(fi=fi_target, pi=pi, fij=fij_target, pij=pij)
        density = compute_density(mask)
        
        # Display decimation step progress
        print(template.format(f"Step {count:>3}", f"Density: {density:.4f}", f"Pearson: {pearson:.4f}", f"Slope: {slope:.4f}"))
        
        # Check if checkpoint threshold reached and compute entropy
        if checkpoint_dec.check(density):
            print(f"\n  → Computing entropy at density {density:.3f}...")
            S = compute_entropy(
                params,
                args.data,
                sampler,
                chains,
                output,
                label=f"{density:.3f}",
                tokens=checkpoint.tokens,
                nsweeps_zero=args.nsweeps_zero,
                nsweeps_theta=args.nsweeps_theta,
                theta_max=args.theta_max,
                nsteps=args.nsteps,
                nsweeps=args.nsweeps_step,
                device=device,
                dtype=dtype,
            )
            
            # Log entropy checkpoint (if not already logged at regular interval)
            if count % checkpoint_dec.checkpt_interval != 0:
                checkpoint_dec.log(
                    {
                        "Epochs": count,
                        "Pearson": pearson,
                        "Entropy": S,
                        "Density": density,
                        "Time": time.time() - time_start,
                    }
                )
            
            # Save model state at checkpoint
            checkpoint_dec.save(
                params=params,
                mask=torch.logical_and(mask, mask_save),
                chains=chains,
                log_weights=log_weights,
                density=density,
            )
            
            # Store and save entropy value
            entropy_values.append((density, S))
            with open(entropy_file, "a") as f:
                f.write(f"{density:.6f}\t{S:.6f}\n")
            print(f"  ✓ Entropy at density {density:.3f}: {S:.6f}\n")
        
        # Log progress at regular intervals or at completion
        elif count % checkpoint_dec.checkpt_interval == 0 or count == MAX_EPOCHS or density <= target_density:
            checkpoint_dec.log(
                {
                    "Epochs": count,
                    "Pearson": pearson,
                    "Entropy": "nan",
                    "Density": density,
                    "Time": time.time() - time_start,
                }
            )
    
    # Save final model state
    checkpoint_dec.save(
        params=params,
        mask=torch.logical_and(mask, mask_save),
        chains=chains,
        log_weights=log_weights,
        density=density
    )
    
    # Display completion message
    print(f"\n{'='*70}")
    print(f"  DECIMATION COMPLETED")
    print(f"{'='*70}")
    print(f"\n  ✓ Entropy values saved: {entropy_file}")
    print(f"  ✓ Model parameters saved: {checkpoint.file_paths['params_dec']}\n")