from typing import Dict, Any
from abc import ABC, abstractmethod

import torch
import h5py
import wandb

from adabmDCA.io import save_chains, save_params
from adabmDCA.statmech import _get_acceptance_rate
# from adabmDCA.checkpoint import Checkpoint


class Checkpoint(ABC):
    """Base class for checkpointing model parameters and chains during training.
    
    This abstract class provides functionality to save the model's parameters and chains 
    at regular intervals during training, and to log the progress of the training process.
    It supports optional integration with Weights & Biases for experiment tracking.
    """

    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: torch.Tensor | None = None,
        use_wandb: bool = False,
    ):
        """Initialize the Checkpoint class.

        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the training arguments. Can be a dict or an object 
                with a __dict__ attribute.
            params (Dict[str, torch.Tensor] | None, optional): Initial parameters of the model. 
                If provided, a deep copy is stored. Defaults to None.
            chains (torch.Tensor | None, optional): Initial chains. If provided, a copy is stored. 
                Defaults to None.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. 
                Defaults to False.
        """
        if not isinstance(args, dict):
            args = vars(args)
            
        self.file_paths = file_paths
        self.tokens = tokens
        
        self.wandb = use_wandb
        if self.wandb:
            wandb.init(project="adabmDCA", config=args)
            
        if params is not None:
            self.params = {key: value.clone() for key, value in params.items()}
        else:
            self.params = None
        if chains is not None:
            self.chains = chains.clone()
        else:
            self.chains = None
        self.max_epochs = args["nepochs"]
        self.checkpt_interval = 50
        
        self.logs = {
            "Epochs": 0,
            "Pearson": 0.0,
            "Entropy": 0.0,
            "Density": 0.0,
            "Time": 0.0,
        }
        
        template = "{0:<20} {1:<50}\n"  
        with open(file_paths["log"], "w") as f:
            if args["label"] is not None:
                f.write(template.format("label:", args["label"]))
            else:
                f.write(template.format("label:", "N/A"))
            
            f.write(template.format("model:", str(args["model"])))
            f.write(template.format("input MSA:", str(args["data"])))
            f.write(template.format("alphabet:", args["alphabet"]))
            f.write(template.format("sampler:", args["sampler"]))
            f.write(template.format("nchains:", args["nchains"]))
            f.write(template.format("nsweeps:", args["nsweeps_dec"]))
            f.write(template.format("lr:", args["lr"]))
            f.write(template.format("pseudo count:", args["pseudocount"]))
            f.write(template.format("data type:", args["dtype"]))
            f.write(template.format("target Pearson Cij:", args["target"]))
            if args["model"] == "edDCA":
                f.write(template.format("density target:", args["density"]))
                f.write(template.format("decimation rate:", args["drate"]))
            f.write(template.format("random seed:", args["seed"]))
            f.write("\n")

    def header_log(self) -> None:
        """Write the header row to the log file.
        
        Creates a formatted header string with all log keys and appends it to the log file.
        """
        header_string = " ".join([f"{key:<10}" for key in self.logs.keys()])
        with open(self.file_paths["log"], "a") as f:
            f.write(header_string + "\n")

    def log(
        self,
        record: Dict[str, Any],
    ) -> None:
        """Log training metrics and write them to the log file.
        
        Updates the internal log dictionary with the provided record and writes the 
        current state to the log file. If Weights & Biases is enabled, also logs to W&B.

        Args:
            record (Dict[str, Any]): Key-value pairs to be added to the log dictionary. 
                Keys must exist in self.logs. Values can be tensors or primitive types.
        
        Raises:
            ValueError: If a key in the record is not recognized (not in self.logs).
        """
        for key, value in record.items():
            if key not in self.logs.keys():
                raise ValueError(f"Key {key} not recognized.")
        
            if isinstance(value, torch.Tensor):
                self.logs[key] = value.item()
            else:
                self.logs[key] = value
                
        if self.wandb:
            wandb.log(self.logs)        
        out_string = " ".join([f"{value:<10.3f}" if isinstance(value, float) else f"{value:<10}" for value in self.logs.values()])
        with open(self.file_paths["log"], "a") as f:
            f.write(out_string + "\n")

    @abstractmethod
    def check(
        self,
        updates: int,
        *args,
        **kwargs,
    ) -> bool:
        """Check if a checkpoint condition has been reached.
        
        This method must be implemented by subclasses to define when checkpoints occur.
        
        Args:
            updates (int): Number of gradient updates performed.
            *args: Additional positional arguments (subclass-specific).
            **kwargs: Additional keyword arguments (subclass-specific).

        Returns:
            bool: True if a checkpoint has been reached, False otherwise.
        """
        pass

    @abstractmethod 
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> None:
        """Save the model's parameters and chains to disk.
        
        This method must be implemented by subclasses to define the saving behavior.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model (fields and couplings).
            mask (torch.Tensor): Binary mask of the model's coupling matrix representing 
                the interaction graph topology.
            chains (torch.Tensor): Current state of the Markov chains.
            log_weights (torch.Tensor): Logarithm of the chain weights, used for 
                Annealed Importance Sampling (AIS).
        """
        pass


# class LinearCheckpoint(Checkpoint):
#     def __init__(
#         self,
#         file_paths: dict,
#         tokens: str,
#         args: dict,
#         params: Dict[str, torch.Tensor] | None = None,
#         chains: torch.Tensor | None = None,
#         checkpt_interval: int = 50,
#         use_wandb: bool = False,
#         **kwargs,
#     ):
#         super().__init__(
#             file_paths=file_paths,
#             tokens=tokens,
#             args=args,
#             params=params,
#             chains=chains,
#             use_wandb=use_wandb,
#         )
#         self.checkpt_interval = checkpt_interval
        
    
#     def check(
#         self,
#         updates: int,
#         *args,
#         **kwargs,
#     ) -> bool:
#         """Checks if a checkpoint has been reached.
        
#         Args:
#             updates (int): Number of gradient updates performed.

#         Returns:
#             bool: Whether a checkpoint has been reached.
#         """
#         return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)
    
    
#     def save(
#         self,
#         params: Dict[str, torch.Tensor],
#         mask: torch.Tensor,
#         chains: torch.Tensor,
#         log_weights: torch.Tensor,
#     ) -> None:
#         """Saves the chains and the parameters of the model.

#         Args:
#             params (Dict[str, torch.Tensor]): Parameters of the model.
#             mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
#             chains (torch.Tensor): Chains.
#             log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
#         """            
#         save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
#         save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)


class DecCheckpoint(Checkpoint):
    """Checkpoint implementation based on coupling matrix density thresholds.
    
    This checkpoint strategy saves the model state when the density of the coupling 
    matrix reaches predefined thresholds. It's designed for decimation-based training 
    where the interaction graph is progressively sparsified.
    """

    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        params: Dict[str, torch.Tensor] | None = None,
        chains: torch.Tensor | None = None,
        checkpt_steps: list[float] | None = None,
        use_wandb: bool = False,
        target_density: float | None = None,
        n_steps: int = 10,
        **kwargs,
    ):
        """Initialize the DecCheckpoint class.
        
        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the training arguments.
            params (Dict[str, torch.Tensor] | None, optional): Initial parameters of the model. 
                Defaults to None.
            chains (torch.Tensor | None, optional): Initial chains. Defaults to None.
            checkpt_steps (list[float] | None, optional): List of density thresholds at which 
                to save checkpoints. If None, generates geometric steps from 0.99 to 
                target_density. Defaults to None.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. 
                Defaults to False.
            target_density (float | None, optional): Final target density for the coupling 
                matrix. Used to generate checkpoint steps if checkpt_steps is None. 
                Defaults to None.
            n_steps (int, optional): Number of checkpoint steps to generate if checkpt_steps 
                is None. Defaults to 10.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            args=args,
            params=params,
            chains=chains,
            use_wandb=use_wandb,
        )
        
        # Generate geometric checkpoints from 0.99 to target_density
        if checkpt_steps is None:
            x, y, N = 0.99, target_density, n_steps
            ratio = (y / x) ** (1 / (N - 1))
            self.checkpt_steps = [x * (ratio ** i) for i in range(N)]
            # Round to 2 decimal places
            self.checkpt_steps = [round(step, 2) for step in self.checkpt_steps]
        else:
            # Ensure descending order
            self.checkpt_steps = sorted(checkpt_steps, reverse=True)

    def check(
        self,
        density: float,
        *args,
        **kwargs,
    ) -> bool:
        """Check if a density threshold checkpoint has been reached.
        
        This method checks if the current coupling matrix density has crossed any 
        checkpoint thresholds. It automatically removes all passed thresholds, 
        allowing for skipping of intermediate checkpoints if the density decreases 
        rapidly.
        
        Args:
            density (float): Current density of the model's coupling matrix.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            bool: True if a checkpoint threshold has been crossed, False otherwise.
        """
        # Return False if all checkpoints have been reached
        if len(self.checkpt_steps) == 0:
            return False
        
        # Skip all intermediate checkpoints and jump to the lowest one reached
        checkpoint_reached = False
        while len(self.checkpt_steps) > 0 and density <= self.checkpt_steps[0]:
            self.checkpt_steps.pop(0)
            checkpoint_reached = True

        if checkpoint_reached:
            print(f"Checkpoint reached at density {density:.3f}")
        
        return checkpoint_reached

    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        log_weights: torch.Tensor,
        density: float,
    ) -> None:
        """Save the model's parameters and chains with density-labeled filenames.
        
        Creates checkpoint files with the current density value appended to the filename,
        allowing for tracking of model evolution across different sparsity levels.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model (fields and couplings).
            mask (torch.Tensor): Binary mask of the model's coupling matrix representing 
                the interaction graph topology.
            chains (torch.Tensor): Current state of the Markov chains.
            log_weights (torch.Tensor): Logarithm of the chain weights, used for 
                Annealed Importance Sampling (AIS).
            density (float): Current density of the model's coupling matrix. Used to 
                label the output files.
        """
        # Create file paths with density value
        params_file = self.file_paths["params"].parent / f"{self.file_paths['params'].stem}_density_{density:.3f}{self.file_paths['params'].suffix}"
        chains_file = self.file_paths["chains"].parent / f"{self.file_paths['chains'].stem}_density_{density:.3f}{self.file_paths['chains'].suffix}"
        
        save_params(fname=params_file, params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=chains_file, chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)


# def get_checkpoint(chpt: str) -> Checkpoint:
#     if chpt == "linear":
#         return LinearCheckpoint
#     elif chpt == "acceptance":
#         return AcceptanceCheckpoint
#     else:
#         raise ValueError(f"Checkpoint type {chpt} not recognized.")        