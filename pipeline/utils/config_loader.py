# utils/config_loader.py

from pathlib import Path
from omegaconf import OmegaConf
from .dir_builder import build_feature_dirs


# create dataset registry
DATASET_REGISTRY = {
    "tggates": {
        "folder": "TG-GATES",
        "metadata": "TG-GATES/metadata.csv",
        # "metadata": "TG-GATES/metadata.xlsx",
    },
    "ucb": {
        "folder": "UCB",
        "metadata": "UCB/metadata.csv",
        # "metadata": "UCB/metadata.xlsx",
    },
}

# datasets that can only be used for test
TEST_ONLY_DATASETS = {"ucb"}


def infer_dataset_key(cfg):
    """
    Resolve and validate dataset selection.

    Dataset must be explicitly specified via CLI (--dataset)
    or via datasets.name in the YAML config.
    """

    if not cfg.datasets.get("name"):
        raise ValueError(
            "Dataset must be specified explicitly via --dataset or datasets.name. "
            f"Available datasets: {list(DATASET_REGISTRY)}"
        )

    key = str(cfg.datasets.name).lower()

    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{key}'. "
            f"Available datasets: {list(DATASET_REGISTRY)}"
        )

    split = cfg.datasets.get("split")
    if key in TEST_ONLY_DATASETS and split != "test":
        raise ValueError(
            f"Dataset '{key}' can only be used for test stage, "
            f"but got split='{split}'."
        )

    return key

def incorporate_cli_args(cfg, args):
    """
    Incorpororporate command-line interface (CLI) arguments into a pipeline configuration.

    Parameters:
    - cfg (OmegaConf): The base configuration to modify.
    - args (argparse.Namespace): The CLI arguments to incorporate.

    Returns:
    - OmegaConf: The modified configuration with CLI overrides applied.
    """

    if args is None:
        return cfg

    cli = OmegaConf.create()

    # features (slide or animal level) + encoder
    if args.model:
        cli.features = OmegaConf.create()
        cli.features.encoder = args.model.upper()

    if args.ftype:
        cli.setdefault("features", OmegaConf.create()).feature_type = args.ftype

    # probe type
    if args.probe:
        cli.probe = OmegaConf.create()
        cli.probe.type = args.probe

    # hidden dim and layers
    if args.hidden_dim is not None:
        cli.setdefault("probe", OmegaConf.create()).hidden_dim = args.hidden_dim

    if args.layers is not None:
        cli.setdefault("probe", OmegaConf.create()).num_layers = args.layers

    # k-shot few-shot learning
    if args.k is not None:
        cli.fewshot = OmegaConf.create()
        cli.fewshot.k = args.k

    # aggregation method
    if args.agg:
        cli.aggregation = OmegaConf.create()
        cli.aggregation.type = args.agg

    # runtime overrides (optimizer, loss, device, lr, batch_size, epochs, momentum, weight_decay, num_workers)
    cli.setdefault("runtime", OmegaConf.create())
    for name in [
        "optimizer", "loss", "device",
        "lr", "batch_size", "epochs",
        "momentum", "weight_decay",
        "num_workers",
    ]:
        val = getattr(args, name, None)
        if val is not None:
            cli.runtime[name] = val

    # datasets
    cli.datasets = OmegaConf.create()

    # explicit dataset name (ucb, tggates, etc.)
    if getattr(args, "dataset", None):
        cli.datasets.name = args.dataset

    # train split is default for training stage,
    # but can be overridden by explicit subset CSV or few-shot k
    if args.stage == "train":
        cli.datasets.split = "train"

        # explicit subset CSV overrides everything
        if args.train_subset_csv:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = args.train_subset_csv

        # few-shot fallback
        elif args.k is not None:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = (
                f"{cfg.data.data_root}/TG-GATES/FewShotCompoundBalanced/"
                f"train_fewshot_k{args.k}.csv"
            )
        else:
            cli.datasets.use_subset = False

    # validation split is default for eval stage,
    # but can be overridden by explicit subset CSV
    elif args.stage == "eval":
        cli.datasets.split = "val"

        if args.eval_subset_csv:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = args.eval_subset_csv
        else:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = (
                f"{cfg.data.data_root}/TG-GATES/Subsets/val_balanced_subset.csv"
            )

    # test split is default for test stage,
    # but can be overridden by explicit subset CSV
    elif args.stage == "test":
        cli.datasets.split = "test"
        cli.datasets.use_subset = True

        if getattr(args, "test_subset_csv", None):
            cli.datasets.subset_csv = args.test_subset_csv
        else:
            cli.datasets.subset_csv = (
                f"{cfg.data.data_root}/TG-GATES/Splits/test.csv"
            )

    merged = OmegaConf.merge(cfg, cli)
    return merged


def load_merged_config(config_path, args=None):
    """
    Load base YAML config, apply CLI overrides,
    resolve dataset configuration, feature directories,
    and finalize model settings.
    """

    config_path = Path(config_path)
    base_cfg = OmegaConf.load(config_path)

    # 1) apply CLI overrides
    cfg = incorporate_cli_args(base_cfg, args)
    cfg.setdefault("datasets", OmegaConf.create())

    # 2) resolve dataset configuration
    dataset_key = infer_dataset_key(cfg)
    cfg.datasets.name = dataset_key

    # dataset used for training (default = same as eval dataset)
    cfg.datasets.train_name = cfg.datasets.name

    # test-only datasets always use TG-GATES-trained models
    if cfg.datasets.name in TEST_ONLY_DATASETS:
        cfg.datasets.train_name = "tggates"


    dataset_info = DATASET_REGISTRY[dataset_key]
    dataset_folder = dataset_info["folder"]

    # metadata + split CSVs
    cfg.data.metadata_csv = (
        Path(cfg.data.data_root) / dataset_info["metadata"]
    )

    cfg.datasets.train = Path(cfg.data.data_root) / dataset_folder / "Splits/train.csv"
    cfg.datasets.val   = Path(cfg.data.data_root) / dataset_folder / "Splits/val.csv"
    cfg.datasets.test  = Path(cfg.data.data_root) / dataset_folder / "Splits/test.csv"

    # 3) probe aggregation method
    if cfg.probe.type.lower() in {"abmil", "clam", "dsmil"}:
        cfg.aggregation.type = "MIL"
    

    # 4) build feature directories based on dataset and split
    dirs = build_feature_dirs(
    features_root=str(Path(cfg.features.features_root)),
    encoder=cfg.features.encoder,
    cache_root=cfg.features.cache_root,
    split=cfg.datasets.split,
    aggregation=cfg.aggregation.type,
    dataset_key=dataset_key,
    dataset_folder=dataset_folder,
    )

    cfg.data.raw_slide_dir = str(dirs["raw_slide_dir"])
    cfg.data.slide_dir = str(dirs["slide_dir"])
    cfg.data.animal_dir = str(dirs["animal_dir"])

    # 5) encoder embedding dimension
    pipeline_root = Path(__file__).resolve().parents[1]
    enc_cfg = OmegaConf.load(
        pipeline_root / "configs" / "models" / "encoder_dims.yaml"
    )
    cfg.features.embed_dim = enc_cfg.encoder_dims[cfg.features.encoder]

    # 6) probe-specific override YAML (if exists)
    probe_yaml = (
        pipeline_root / "configs" / "probes" / f"{cfg.probe.type}.yaml"
    )
    if probe_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(probe_yaml))

    return cfg
