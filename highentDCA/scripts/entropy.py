from pathlib import Path
import time

import torch
import numpy as np

from adabmDCA.io import import_from_fasta
from adabmDCA.functional import one_hot
from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.dca import get_seqid
from adabmDCA.utils import init_chains


def compute_entropy(
    params,
    path_targetseq,
    sampler,
    chains,
    output,
    label,
    tokens,
    nsweeps_zero=100,
    nsweeps_theta=100,
    theta_max=5.0,
    nsteps=100,
    nsweeps=100,
    device="cuda",
    dtype="float32"
):
    """Compute the entropy of a DCA model using Thermodynamic Integration.
    
    This function performs thermodynamic integration to compute the entropy of a DCA model
    by integrating the average sequence identity from theta=0 to theta=theta_max, where
    theta controls the bias towards a target sequence.
    
    Args:
        params (Dict[str, torch.Tensor]): Model parameters (fields and couplings).
        path_targetseq (str): Path to the fasta file containing the target sequence.
        sampler (callable): Sampling function to update Markov chains.
        chains (torch.Tensor): Initial Markov chains for sampling.
        output (str): Path to the output folder for saving results.
        label (str): Label for the output files. Must be provided.
        tokens (str): Alphabet tokens for sequence encoding.
        nsweeps_zero (int, optional): Number of sweeps to equilibrate at theta=0. Defaults to 100.
        nsweeps_theta (int, optional): Number of sweeps to equilibrate at theta_max. Defaults to 100.
        theta_max (float, optional): Initial maximum integration strength. Defaults to 5.0.
        nsteps (int, optional): Number of integration steps. Defaults to 100.
        nsweeps (int, optional): Number of sweeps per integration step. Defaults to 100.
        device (str, optional): Device for computations ('cuda' or 'cpu'). Defaults to 'cuda'.
        dtype (str, optional): Data type for tensors. Defaults to 'float32'.
        
    Returns:
        float: Computed entropy value.
        
    Raises:
        FileNotFoundError: If the target sequence file does not exist.
        ValueError: If label is not provided.
    """
    
    print("\n" + "="*70)
    print("  COMPUTING MODEL'S ENTROPY")
    print("="*70 + "\n")
    
    # Create output directory structure
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    
    # Validate target sequence file path
    if not Path(path_targetseq).exists():
        raise FileNotFoundError(f"Target Sequence file {path_targetseq} not found.")
    
    if label is not None:
        file_log = folder / Path(f"density_{label}.log")
    else:
        raise ValueError("Label must be provided for entropy computation.")
    
    # Load and process target sequence
    _, targetseq = import_from_fasta(path_targetseq, tokens=tokens, filter_sequences=True)
    nchains, L, q = chains.shape[0], chains.shape[1], chains.shape[2]
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    if len(targetseq) != 1:
        print(f"Target sequence file contains more than one sequence. Using the first sequence as target sequence.")
        targetseq = targetseq[0]
    
    # Initialize log file and write configuration
    template = "{0:<20} {1:<50}\n"
    with open(file_log, "w") as f:
        f.write(template.format("density:", label))
        f.write(template.format("nchains:", nchains))
        f.write(template.format("nsweeps:", nsweeps))
        if nsweeps_theta is not None:
            f.write(template.format("nsweeps  theta:", str(nsweeps_theta)))
        if nsweeps_zero is not None:
            f.write(template.format("nsweeps zero:", str(nsweeps_zero)))
        f.write(template.format("nsteps:", nsteps))
        f.write(template.format("data type:", str(dtype)))
        f.write("\n")
        
        # Write log header
        logs = {
            "Epoch": 0,
            "Theta": 0.0,
            "Free Energy": 0.0,
            "Entropy": 0.0,
            "Time": 0.0
        }
        header_string = " ".join([f"{key:<15}" for key in logs.keys()])
        f.write(header_string + "\n")
    
    # Thermalize chains at theta = 0 (unbiased model)
    print("→ Thermalizing at theta = 0...")
    chains_0 = sampler(chains, params, nsweeps_zero)
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))
    
    # Thermalize chains at theta = theta_max (biased towards target)
    print("→ Thermalizing at theta = theta_max...")
    params_theta = {k: v.clone() for k, v in params.items()}
    params_theta["bias"] += theta_max * targetseq
    
    chains_theta = init_chains(nchains, L, q, device=device, dtype=dtype)
    chains_theta = sampler(chains_theta, params_theta, nsweeps_theta)
    seqID = get_seqid(chains_theta, targetseq)
    
    # Adjust theta_max to achieve ~10% target sequences in sample
    print("→ Finding theta_max to generate 10% target sequences...")
    p_wt = (seqID == L).sum().item() / nchains
    nsweep_find_theta = 100
    while p_wt <= 0.1:
        theta_max += 0.01 * theta_max
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = get_seqid(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / nchains
    print(f"  ✓ Found theta_max = {theta_max:.4f} ({(p_wt * 100):.2f}% sequences at target)")
    
    # Initialize thermodynamic integration
    print(f"\n→ Running Thermodynamic Integration ({nsteps} steps)...\n")
    int_step = nsteps
    F_max = np.log(p_wt) + torch.mean(compute_energy(chains_theta[seqID == L], params_theta))
    thetas = torch.linspace(0, theta_max, int_step)
    factor = theta_max / (2 * int_step)
    F, S, integral = F_max, 0, 0
    torch.set_printoptions(precision=2)
    time_start = time.time()
    
    # Perform thermodynamic integration loop
    for i, theta in enumerate(thetas):
        # Sample chains at current theta value
        params_theta["bias"] = params["bias"] + theta * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweeps)
        seqID = get_seqid(chains_theta, targetseq)
        mean_seqID = seqID.mean()
        
        # Trapezoidal integration step to accumulate free energy
        if i == 0 or i == int_step - 1:
            F += factor * torch.mean(seqID)
            integral += factor * mean_seqID
        else:
            F += 2 * factor * mean_seqID
            integral += 2 * factor * mean_seqID
        S = ave_energy_0 - F
        
        # Log current step results
        logs["Epoch"] = i
        logs["Theta"] = float(theta)
        logs["Free Energy"] = F.item()
        logs["Entropy"] = S.item()
        logs["Time"] = time.time() - time_start
        with open(file_log, "a") as f:
            f.write(" ".join([f"{value:<15.3f}" if isinstance(value, float) else f"{value:<15}" for value in logs.values()]) + "\n")
    
    print(f"\n  ✓ Entropy computation completed")
    print(f"  ✓ Results saved: {file_log}\n")
    
    return S.item()