import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import time

from adabmDCA.fasta import get_tokens
from adabmDCA.utils import init_chains, get_device, get_dtype, resample_sequences
from adabmDCA.io import load_params, load_chains, import_from_fasta
from adabmDCA.functional import one_hot
from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy
from adabmDCA.parser import add_args_tdint
from adabmDCA.dca import get_seqid


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
    # seed=0,
    device="cuda",
    dtype="float32"
):
    """
    Computes the Entropy of the given model using a Thermodynamical Integration.
    
    Args:
        data: Path to the fasta file containing the data
        path_params: Path to the file containing the model's parameters
        path_targetseq: Path to the file containing the target sequence
        output: Path to the folder where to save the output
        path_chains: Path to the fasta file containing the model's chains (optional)
        label: Label to be used for the output files (optional)
        nchains: Number of chains to be used
        theta_max: Maximum integration strength
        nsteps: Number of integration steps
        nsweeps: Number of chain updates for each integration step
        nsweeps_theta: Number of chain updates to equilibrate chains at theta_max
        nsweeps_zero: Number of chain updates to equilibrate chains at theta=0
        sampler: Sampling method to be used
        device: Device to perform computations on
        dtype: Data type to be used
        
    Returns:
        dict: Dictionary containing the final entropy and free energy values
    """
    
    print("\n" + "".join(["*"] * 10) + f" Computing model's entropy " + "".join(["*"] * 10) + "\n")
    
    # Create output folder
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)


    # Check if the target-sequence file exists
    if not Path(path_targetseq).exists():
        raise FileNotFoundError(f"Target Sequence file {path_targetseq} not found.")
    
    if label is not None:
        file_log = folder / Path(f"density_{label}.log")      
    else:
        # ERROR HANDLING: label must be provided
        raise ValueError("Label must be provided for entropy computation.")
        
    # target sequence
    _, targetseq = import_from_fasta(path_targetseq, tokens=tokens, filter_sequences=True)
    nchains, L, q = chains.shape[0], chains.shape[1], chains.shape[2]
    targetseq = one_hot(torch.tensor(targetseq, device=device, dtype=torch.int32), num_classes=q).to(dtype)
    if len(targetseq) != 1:
        print(f"Target sequence file contains more than one sequence. Using the first sequence as target sequence.")
        targetseq = targetseq[0]

    # initialize checkpoint
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
        # write the header of the log file
        logs = {
            "Epoch": 0,
            "Theta": 0.0,
            "Free Energy": 0.0,
            "Entropy": 0.0,
            "Time": 0.0
        }
        header_string = " ".join([f"{key:<15}" for key in logs.keys()])
        f.write(header_string + "\n")
    
    # Sampling to thermalize at theta = 0
    print("Thermalizing at theta = 0...")
    chains_0 = sampler(chains, params, nsweeps_zero) 
    ave_energy_0 = torch.mean(compute_energy(chains_0, params))

    # Sampling to thermalize at theta = theta_max
    print("Thermalizing at theta = theta_max...")
    params_theta = {k : v.clone() for k, v in params.items()}
    params_theta["bias"] += theta_max * targetseq
    
    chains_theta = init_chains(nchains, L, q, device=device, dtype=dtype)
    chains_theta = sampler(chains_theta, params_theta, nsweeps_theta)
    seqID_max = get_seqid(chains_theta, targetseq)
            
    # Find theta_max to generate 10% target sequences in the sample
    print("Finding theta_max to generate 10% target sequences in the sample...")
    p_wt =  (seqID_max == L).sum().item() / nchains
    nsweep_find_theta = 100
    while p_wt <= 0.1:
        theta_max += 0.01 * theta_max
        print(f"{(p_wt * 100):.2f}% sequences collapse to WT")
        print(f"Number of sequences collapsed to WT is less than 10%. Increasing theta max to: {theta_max:.2f}...")
        params_theta["bias"] = params["bias"] + theta_max * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweep_find_theta)
        seqID = get_seqid(chains_theta, targetseq)
        p_wt = (seqID == L).sum().item() / nchains
    
    # initiaize Thermodynamic Integration
    print("Thermodynamic Integration...")
    int_step = nsteps
    F_max = np.log(p_wt) + torch.mean(compute_energy(chains_theta[seqID == L], params_theta))
    thetas = torch.linspace(0, theta_max, int_step) 
    factor = theta_max / (2 * int_step)
    F, S, integral = F_max, 0, 0
    torch.set_printoptions(precision=2)
    time_start = time.time()

    for i, theta in enumerate(thetas):
        
        # sampling and compute seqID
        params_theta["bias"] = params["bias"] + theta * targetseq
        chains_theta = sampler(chains_theta, params_theta, nsweeps)
        seqID = get_seqid(chains_theta, targetseq)
        mean_seqID = seqID.mean()
        
        # step of integration to compute entropy
        if i == 0 or i == int_step - 1:
            F += factor * torch.mean(seqID) 
            integral += factor * mean_seqID
        else:
            F += 2 * factor * mean_seqID
            integral += 2 * factor * mean_seqID
        S = ave_energy_0 - F
   
        # checkpoint
        logs["Epoch"] = i
        logs["Theta"] = float(theta)
        logs["Free Energy"] = F.item()
        logs["Entropy"] = S.item()
        logs["Time"] = time.time() - time_start
        with open(file_log, "a") as f:
            f.write(" ".join([f"{value:<15.3f}" if isinstance(value, float) else f"{value:<15}" for value in logs.values()]) + "\n")
        
    # pbar.close()
    print(f"Process completed. Results saved in {file_log}.")
    
    return S.item()
  