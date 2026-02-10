# utils/config_loader.py

from pathlib import Path
from omegaconf import OmegaConf
from .dir_builder import build_feature_dirs


def incorporate_cli_args(cfg, args):
    """
    Incorpororporate command-line interface (CLI) arguments into a pipeline configuration.

    Parameters:
    - cfg (OmegaConf): The base configuration to modify.
    - args (argparse.Namespace): The CLI arguments to incorporate.

    Returns:
    - OmegaConf: The modified configuration with CLI overrides applied.
    - entries: List[str]: A list of CLI overrides applied, formatted as "key=value" for logging purposes.
    """

    if args is None:
        return cfg, []

    cli = OmegaConf.create()
    entries = []

    # features + encoder
    if args.model:
        cli.features = OmegaConf.create()
        cli.features.encoder = args.model.upper()
        entries.append(f"features.encoder={args.model}")

    if args.ftype:
        cli.setdefault("features", OmegaConf.create()).feature_type = args.ftype

    # probe
    if args.probe:
        cli.probe = OmegaConf.create()
        cli.probe.type = args.probe
        entries.append(f"probe.type={args.probe}")

    # hidden dim and layers
    if args.hidden_dim is not None:
        cli.setdefault("probe", OmegaConf.create()).hidden_dim = args.hidden_dim

    if args.layers is not None:
        cli.setdefault("probe", OmegaConf.create()).num_layers = args.layers

    # fewshot
    if args.k is not None:
        cli.fewshot = OmegaConf.create()
        cli.fewshot.k = args.k

    # aggregation
    if args.agg:
        cli.aggregation = OmegaConf.create()
        cli.aggregation.type = args.agg

    # runtime settings
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

    # datasets and subsets
    cli.datasets = OmegaConf.create()

    # train split is default for training stage, but can be overridden by explicit subset CSV or fewshot k
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
                f"{cfg.data.data_root}/FewShotCompoundBalanced/"
                f"train_fewshot_k{args.k}.csv"
            )
        else:
            cli.datasets.use_subset = False

    # validation split is default for eval stage, but can be overridden by explicit subset CSV
    elif args.stage == "eval":
        cli.datasets.split = "val"

        # optional validation subset
        if args.eval_subset_csv:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = args.eval_subset_csv

        # default to balanced val subset for evaluation if no explicit CSV provided
        else: 
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = (
                f"{cfg.data.data_root}/Subsets/val_balanced_subset.csv"
            )

    # test split is default for test stage, but can be overridden by explicit subset CSV
    elif args.stage == "test":
        cli.datasets.split = "test"

        # Optional test subset (explicit only)
        if getattr(args, "test_subset_csv", None):
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = args.test_subset_csv

        # Default to test subset
        else:
            cli.datasets.use_subset = True
            cli.datasets.subset_csv = (
                f"{cfg.data.data_root}/Splits/test.csv"
            )

    # merge CLI overrides into base config
    merged = OmegaConf.merge(cfg, cli)
    return merged, entries


# Main function to load and merge config
def load_merged_config(config_path, args=None):
    """
    Load base YAML config, apply CLI overrides,
    resolve feature directories, and finalize model settings.
    """
    config_path = Path(config_path)
    base_cfg = OmegaConf.load(config_path)

    # 1) apply CLI overrides
    cfg, _ = incorporate_cli_args(base_cfg, args)

    # 2) build directories for raw and cached features
    dirs = build_feature_dirs(
        features_root=cfg.features.features_root,
        encoder=cfg.features.encoder,
        cache_root=cfg.features.cache_root,
        split=cfg.datasets.split,
        aggregation=cfg.aggregation.type,
    )

    # assign dirs to cfg.data
    cfg.data.raw_slide_dir = str(dirs["raw_slide_dir"])
    cfg.data.slide_dir     = str(dirs["slide_dir"])
    cfg.data.animal_dir    = str(dirs["animal_dir"])

    # 3) apply encoder embedding dimension
    pipeline_root = Path(__file__).resolve().parents[1]
    enc_file = pipeline_root / "configs" / "models" / "encoder_dims.yaml"
    enc_cfg = OmegaConf.load(enc_file)
    cfg.features.embed_dim = enc_cfg.encoder_dims[cfg.features.encoder]

    # 4) probe-specific override YAML (if exists)
    probe_yaml = (
        pipeline_root / "configs" / "probes" / f"{cfg.probe.type}.yaml"
    )
    if probe_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(probe_yaml))

    return cfg
