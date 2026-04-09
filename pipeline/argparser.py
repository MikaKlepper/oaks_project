# pipeline/argparser.py
import argparse

def get_args():
    parser = argparse.ArgumentParser("Animal-Level Toxicology Pipeline")

    # Base config
    parser.add_argument("--config", type=str,
                        default="configs/base_config.yaml",
                        help="Path to base YAML config file")

    # Stage controller
    parser.add_argument("--stage", type=str, required=False,
        choices=["train", "eval", "test", "all"],
        help="Pipeline stage to execute"
    )

    # Model override
    parser.add_argument("--model", type=str, default=None)

    # Probe override
    parser.add_argument("--probe", type=str, default=None,
        choices=["linear", "mlp", "knn", "logreg", "svm_linear", "svm_rbf", "abmil", "clam", "dsmil", "flow"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)

    # flow specific args
    # Normalizing flow specific
    parser.add_argument("--flow_layers", type=int, default=None,
        help="Number of normalizing flow layers")

    parser.add_argument("--flow_hidden", type=int, default=None,
        help="Hidden dimension for flow coupling networks")

    parser.add_argument("--flow_input_dim", type=int, default=None,
        help="Input dimension for the flow after optional PCA")

    parser.add_argument("--flow_train_max_tiles", type=int, default=None,
        help="Maximum number of training tiles sampled per bag")

    parser.add_argument("--flow_topk_frac", type=float, default=None,
        help="Top-k tile fraction used for slide aggregation (e.g. 0.05)")

    parser.add_argument("--flow_tau_percentile", type=float, default=None,
        help="Percentile threshold for anomaly detection (e.g. 95)")

    parser.add_argument("--flow_pca_fit_max_tiles", type=int, default=None,
        help="Maximum number of tiles used when fitting PCA")

    # Feature type
    parser.add_argument("--ftype", type=str, default=None,
        choices=["animal", "slide"])

    # Few-shot K
    parser.add_argument("--k", type=int, default=None)

    # Aggregation type
    parser.add_argument("--agg", type=str, default=None,
        choices=["mean", "max", "min"])

    # Subset CSVs for training, evaluation, and testing
    parser.add_argument("--train_subset_csv", type=str, default=None,
        help="Subset CSV to use only during training")

    parser.add_argument("--eval_subset_csv", type=str, default=None,
        help="Subset CSV to use only during evaluation")

    parser.add_argument("--test_subset_csv", type=str, default=None,
        help="Subset CSV to use only during testing")

    # Training hyperparameters
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--dataset", type=str, default=None, choices=["tggates", "ucb"], help="Dataset to use")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=["liver_hypertrophy", "any_abnormality"],
        help="Prediction target to use",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        help="Optional experiment tag used to separate outputs/checkpoints",
    )
    parser.add_argument("--calibrate", action="store_true",
        help="Enable test-set calibration / fine-tuning mode")
    parser.add_argument("--calibration_samples", type=int, default=None,
        help="Number of calibration samples to draw from the calibration source CSV")
    parser.add_argument("--calibration_source_csv", type=str, default=None,
        help="Source CSV from which calibration samples are drawn")
    parser.add_argument("--calibration_seed", type=int, default=None,
        help="Random seed for calibration subset sampling")

    # # Checkpoint for eval
    # parser.add_argument("--model_path", type=str, default=None)


    return parser.parse_args()
