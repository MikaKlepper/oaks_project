# pipeline/argparser.pyc
import argparse

def get_args():
    parser = argparse.ArgumentParser("Animal-Level Toxicology Pipeline")

    # Base config
    parser.add_argument("--config", type=str,
                        default="configs/base_config.yaml",
                        help="Path to base YAML config file")

    # Stage to run
    parser.add_argument("--stage", type=str, required=False,
        choices=[
            "split", "check", "aggregate",
            "train", "eval", "test", "all"
        ],
        help="Pipeline stage to execute"
    )
    # Model override
    parser.add_argument("--model", type=str, default=None,
        help="Encoder name (e.g. UNI, UNI2, VIRCHOW, etc.)")

    # Probe override
    parser.add_argument("--probe", type=str, default=None,
        choices=["linear", "mlp", "knn", "logreg", "svm_linear", "svm_rbf"],
        help="Probe type override"
    )

    # feature type, animal or slide
    parser.add_argument("--ftype", type=str, default=None,
        choices=["animal", "slide"],
        help="Type of features to use"
    )

    # Few-shot K
    parser.add_argument("--k", type=int, default=None,
        help="Few-shot subset size (k)")
    
    # Aggegation type
    parser.add_argument("--agg", type=str, default=None,
        choices=["mean", "max", "min", "sum"],
        help="Aggregation type")

    # Optional custom subset CSV
    parser.add_argument( "--subset_csv", type=str, default=None,
                         help="Optional custom subset CSV to use instead of train/val/test"
    )
    
    # Training hyperparameters
    parser.add_argument("--optimizer", type=str, help="Override optimizer type: adam, adamw, sgd, rmsprop")
    parser.add_argument("--loss", type=str, help="Loss function: crossentropy, mse, bce")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, help="Momentum for SGD/RMSProp")
    parser.add_argument("--device", type=str, help="Training device: cpu, cuda, mps")


        # Optional fractional subset
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=None,
        help="Optional fraction of the split to use (e.g., 0.1 for 10%)"
    )

    # Eval checkpoint
    parser.add_argument("--model_path", type=str, default=None,
        help="Path to trained probe checkpoint")

    # Behavior flags
    parser.add_argument("--dry_run", action="store_true",
        help="Simulate pipeline without running heavy computations")

    parser.add_argument("--overwrite", action="store_true",
        help="Allow overwriting experiment folders")

    return parser.parse_args()
