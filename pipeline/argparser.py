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
        choices=["linear", "mlp", "knn", "logreg", "svm_linear", "svm_rbf", "abmil", "clam", "dsmil"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)

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

    parser.add_argument("--dataset", type=str, default=None, choices=["tggates", "ucb"], help="Dataset to use")

    # # Checkpoint for eval
    # parser.add_argument("--model_path", type=str, default=None)


    return parser.parse_args()
