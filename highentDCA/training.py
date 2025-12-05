from tqdm.autonotebook import tqdm
import time
from typing import Tuple, Callable, Dict, List
import torch

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.utils import get_mask_save
from adabmDCA.statmech import _update_weights_AIS, compute_log_likelihood, compute_entropy, _compute_ess
from adabmDCA.checkpoint import Checkpoint
from adabmDCA.training import update_params


def train_graph(
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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]:
    """Trains a Direct Coupling Analysis (DCA) model on a sparse graph using gradient descent.
    
    This function performs iterative training by updating model parameters to match the target
    single-point and two-point statistics of the data. Training continues until either the
    Pearson correlation between model and data two-point statistics reaches the target value,
    or the maximum number of epochs is exceeded.
    
    The training process uses Annealed Importance Sampling (AIS) to estimate the log-partition
    function and log-likelihood online during training. At each epoch:
    1. Parameters are updated based on the gradient of the log-likelihood
    2. AIS weights are updated to track the changing partition function
    3. Markov chains are evolved using the sampler
    4. Statistics are computed and convergence is checked

    Args:
        sampler (Callable): Sampling function that takes chains, params, and nsweeps as arguments
            and returns updated chains after Monte Carlo sampling.
        chains (torch.Tensor): Initial Markov chains of shape (n_chains, L) in integer encoding,
            representing configurations to be evolved during training.
        mask (torch.Tensor): Binary mask of shape (L, q, L, q) encoding the sparse graph structure.
            Only couplings where mask=1 will be learned.
        fi (torch.Tensor): Single-point frequencies (marginals) of the training data, shape (L, q).
        fij (torch.Tensor): Two-point frequencies (pairwise marginals) of the training data,
            shape (L, q, L, q).
        params (Dict[str, torch.Tensor]): Dictionary containing model parameters:
            - 'bias': fields of shape (L, q)
            - 'coupling_matrix': couplings of shape (L, q, L, q)
        nsweeps (int): Number of Gibbs sampling sweeps to perform at each epoch for updating chains.
        lr (float): Learning rate for gradient descent parameter updates.
        max_epochs (int): Maximum number of training epochs allowed.
        target_pearson (float): Target Pearson correlation coefficient between model and data
            two-point statistics. Training stops when this is reached.
        fi_test (torch.Tensor | None, optional): Single-point frequencies of test data for
            computing test log-likelihood. Shape (L, q). Defaults to None.
        fij_test (torch.Tensor | None, optional): Two-point frequencies of test data for
            computing test log-likelihood. Shape (L, q, L, q). Defaults to None.
        checkpoint (Checkpoint | None, optional): Checkpoint object for saving model state
            periodically during training. Defaults to None.
        check_slope (bool, optional): If True, also checks that the slope of the correlation
            between model and data statistics is close to 1.0 (within 0.1). Defaults to False.
        log_weights (torch.Tensor | None, optional): Initial log-weights for AIS, shape (n_chains,).
            If None, initialized to zeros. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar showing training progress.
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, List[float]]]: 
            A tuple containing:
            - chains (torch.Tensor): Updated Markov chains after training, shape (n_chains, L)
            - params (Dict[str, torch.Tensor]): Trained model parameters
            - log_weights (torch.Tensor): Final AIS log-weights, shape (n_chains,)
            - history (Dict[str, List[float]]): Training history with keys:
                - 'epochs': list of epoch numbers
                - 'pearson': Pearson correlation at each epoch
                - 'slope': slope of correlation at each epoch
    """
    # Initialize device, dtype and dimensions
    device = fi.device
    dtype = fi.dtype
    L, q = fi.shape
    time_start = time.time()
    
    # Compute initial log-partition function and log-likelihood
    n_chains = torch.tensor(len(chains), device=device, dtype=dtype)
    
    # Compute initial statistics from the Markov chains
    pi = get_freq_single_point(data=chains)
    pij = get_freq_two_points(data=chains)
    
    # Initialize training history
    history = {
        "epochs": [],
        "pearson": [],
        "slope": [],
    }
    
    # Define convergence criterion
    def halt_condition(epochs, pearson, slope, check_slope):
        c1 = pearson < target_pearson
        c2 = epochs < max_epochs
        if check_slope:
            c3 = abs(slope - 1.) > 0.1
        else:
            c3 = False
        return not c2 * ((not c1) * c3 + c1)
    
    # Mask for saving only the upper-diagonal coupling matrix
    mask_save = get_mask_save(L, q, device=device)

    # Compute initial Pearson correlation and slope
    pearson, slope = get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)
    epochs = 0
    
    # Initialize progress bar
    if progress_bar: 
        pbar = tqdm(
            initial=max(0, float(pearson)),
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]"
        )
        pbar.set_description(f"Epochs: {epochs} - LL: {log_likelihood:.2f}")
    
    # Training loop
    while not halt_condition(epochs, pearson, slope, check_slope):
        # Store current parameters for AIS weight update
        params_prev = {key: value.clone() for key, value in params.items()}
        
        # Update model parameters using gradient descent
        params = update_params(
            fi=fi,
            fij=fij,
            pi=pi,
            pij=pij,
            params=params,
            mask=mask,
            lr=lr,
        )
     
        # Evolve Markov chains with updated parameters
        chains = sampler(chains=chains, params=params, nsweeps=nsweeps)
        epochs += 1
        
        # Compute statistics from updated chains
        pi = get_freq_single_point(data=chains)
        pij = get_freq_two_points(data=chains)
        pearson, slope = get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)
        
        # Update progress bar
        if progress_bar:
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Epochs: {epochs}") # - LL: {log_likelihood:.2f}")
        
        # Record training history
        history["epochs"].append(epochs)
        history["pearson"].append(pearson)
        history["slope"].append(slope)
        
        # Handle checkpointing
        if checkpoint is not None:
           
            # Compute test log-likelihood if test data is provided
            if fi_test is not None and fij_test is not None:
                log_likelihood_test = compute_log_likelihood(
                    fi=fi_test, 
                    fij=fij_test, 
                    params=params, 
                    logZ=logZ
                )
            else:
                log_likelihood_test = float("nan")
            
            # Save checkpoint if required
            if checkpoint.check(epochs, params, chains):
                checkpoint.log({
                    "Epochs": epochs,
                    "Pearson": pearson,
                    "Slope": slope,
                    "LL_train": float("nan"),
                    "LL_test": float("nan"),
                    "ESS": float("nan"),
                    "Entropy": float("nan"),
                    "Density": 1.0,
                    "Time": time.time() - time_start,
                })
                checkpoint.save(
                    params=params,
                    mask=mask_save,
                    chains=chains,
                    log_weights=log_weights,
                )
    
    # Clean up progress bar
    if progress_bar:
        pbar.close()
    
    # Final checkpoint save
    if checkpoint is not None:            
        checkpoint.save(
            params=params,
            mask=mask_save,
            chains=chains,
            log_weights=log_weights,
        )
    
    return chains, params, log_weights, history
