import argparse
from pathlib import Path


def add_args_dca(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("General DCA arguments")
    
    dca_args.add_argument("-d", "--data",         type=Path,  required=True,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=Path,  default='DCA_model',  help="(Defaults to DCA_model). Path to the folder where to save the model.")
    dca_args.add_argument("-m", "--model",        type=str,   default="bmDCA",      help="(Defaults to bmDCA). Type of model to be trained.", choices=["bmDCA", "eaDCA", "edDCA"])
    dca_args.add_argument("-t", "--test",         type=Path,  default=None,         help="(Defaults to None). Filename of the dataset to be used for testing the model.")
    dca_args.add_argument("-p", "--path_params",  type=Path,  default=None,         help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring the training.")
    dca_args.add_argument("-c", "--path_chains",  type=Path,  default=None,         help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--nsweeps",            type=int,   default=10,           help="(Defaults to 50). Number of sweeps for each gradient estimation.")
    dca_args.add_argument("--sampler",            type=str,   default="gibbs",      help="(Defaults to gibbs). Sampling method to be used.", choices=["metropolis", "gibbs"])
    dca_args.add_argument("--nchains",            type=int,   default=10000,        help="(Defaults to 10000). Number of Markov chains to run in parallel.")
    dca_args.add_argument("--target",             type=float, default=0.95,         help="(Defaults to 0.95). Pearson correlation coefficient on the two-sites statistics to be reached.")
    dca_args.add_argument("--nepochs",            type=int,   default=50000,        help="(Defaults to 50000). Maximum number of epochs allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to None). Pseudo count for the single and two-sites statistics. Acts as a regularization. If None, it is set to 1/Meff.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--wandb",              action="store_true",              help="If provided, logs the training on Weights and Biases.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.")
    
    return parser


def add_args_reweighting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    weight_args = parser.add_argument_group("Sequence reweighting arguments")
    
    weight_args.add_argument("-w", "--weights",      type=Path,  default=None,         help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    weight_args.add_argument("--clustering_seqid",   type=float, default=0.8,          help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")
    weight_args.add_argument("--no_reweighting",     action="store_true",              help="If provided, the reweighting of the sequences is not performed.")

    return parser


def add_args_checkpoint(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    checkpoint_args = parser.add_argument_group("Checkpoint arguments")
    
    checkpoint_args.add_argument("--checkpoints",     type=str,   default="linear",     help="(Defaults to 'linear'). Choses the type of checkpoint criterium to be used.", choices=["linear", "acceptance"])
    checkpoint_args.add_argument("--target_acc_rate", type=float, default=0.5,          help="(Defaults to 0.5). Target acceptance rate for deciding when to save the model when the 'acceptance' checkpoint is used.")
    
    return parser


def add_args_edDCA(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    eddca_args = parser.add_argument_group("edDCA arguments")
    
    eddca_args.add_argument("--nsweeps_dec",      type=int,   default=100,          help="(Defaults to 100). Number of sweeps for each gradient estimation.")
    eddca_args.add_argument("--density",          type=float, default=0.02,         help="(Defaults to 0.02). Target density to be reached.")
    eddca_args.add_argument("--drate",            type=float, default=0.001,         help="(Defaults to 0.01). Fraction of remaining couplings to be pruned at each decimation step.")

    return parser


def add_args_entropy(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Optional arguments
    parser.add_argument("--theta_max",          type=float,  default=5,            help="(Defaults to 5). Maximum integration strength") 
    parser.add_argument("--nsteps",             type=int,    default=100,          help="(Defaults to 100). Number of integration steps.")
    parser.add_argument("--nsweeps_step",       type=int,    default=250,          help="(Defaults to 250). Number of chain updates for each integration step.")
    parser.add_argument("--nsweeps_theta",      type=int,    default=1000,         help="(Defaults to 1000). Number of chain updates to equilibrate chains at theta_max.")
    parser.add_argument("--nsweeps_zero",       type=int,    default=5000,         help="(Defaults to 5000). Number of chain updates to equilibrate chains at theta=0.")
    parser.add_argument("--checkpt_steps",      type=str,    default=None,         help="(Defaults to None). Space-separated list of density thresholds at which to save checkpoints (e.g., '0.9 0.8 0.5'). If None, automatic geometric steps are generated.")

    return parser


def add_args_train(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_dca(parser)
    parser = add_args_edDCA(parser)
    parser = add_args_reweighting(parser)
    parser = add_args_checkpoint(parser)
    parser = add_args_entropy(parser)
    
    return parser


