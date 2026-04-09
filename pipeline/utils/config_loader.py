# utils/config_loader.py

from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
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


def _feature_backend(cfg) -> str:
    return str(cfg.features.get("backend", "legacy")).lower()


def resolve_feature_storage(cfg, dataset_key: str, dataset_folder: str) -> dict:
    """
    Resolve feature access paths for either the legacy directory backend or the
    registry-backed feature bank backend.
    """
    backend = _feature_backend(cfg)
    encoder = str(cfg.features.encoder).upper()

    if backend == "feature_bank":
        bank_root = Path(getattr(cfg.features, "bank_root", "feature_bank"))
        registry_path = getattr(cfg.features, "registry_path", None)
        if registry_path in {None, "", "null"}:
            registry_path = bank_root / "registry" / "features.sqlite"
        else:
            registry_path = Path(registry_path)

        local_bank_root = getattr(cfg.features, "local_bank_root", None)
        if local_bank_root in {None, "", "null"}:
            local_bank_root = None
        else:
            local_bank_root = Path(local_bank_root)

        aggregation = str(cfg.aggregation.type).lower()
        shared_raw_root = bank_root / "raw" / dataset_key
        shared_derived_root = bank_root / "derived" / dataset_key / aggregation

        active_bank_root = (
            local_bank_root if local_bank_root is not None else bank_root
        )
        active_raw_root = active_bank_root / "raw" / dataset_key
        active_derived_root = active_bank_root / "derived" / dataset_key / aggregation

        return {
            "backend": backend,
            "raw_slide_dir": active_raw_root,
            "raw_slide_dir_source": f"feature_bank:{shared_raw_root / f'{encoder}.h5'}",
            "slide_dir": active_derived_root,
            "animal_dir": active_derived_root,
            "registry_path": registry_path,
            "bank_root": bank_root,
            "local_bank_root": local_bank_root,
        }

    dirs = build_feature_dirs(
        features_root=str(Path(cfg.features.features_root)),
        encoder=encoder,
        cache_root=cfg.features.cache_root,
        split=cfg.datasets.split,
        aggregation=cfg.aggregation.type,
        dataset_key=dataset_key,
        dataset_folder=dataset_folder,
    )
    dirs["backend"] = backend
    dirs["registry_path"] = None
    dirs["bank_root"] = None
    dirs["local_bank_root"] = None
    return dirs


def _split_files_exist(split_dir: Path) -> bool:
    return all((split_dir / f"{name}.csv").exists() for name in ("train", "val", "test"))


def _split_ids_from_df(df: pd.DataFrame) -> pd.Series:
    if "subject_organ_UID" in df.columns:
        return df["subject_organ_UID"].astype(str)
    if "animal_number" in df.columns:
        return df["animal_number"].astype(str)
    if "slide_id" in df.columns:
        return df["slide_id"].astype(str)
    if "slide_filename" in df.columns:
        return df["slide_filename"].astype(str)
    raise ValueError("Calibration source CSV must contain an ID column.")


def ensure_calibration_subsets(cfg, dataset_key: str) -> tuple[Path | None, Path | None]:
    """
    Generate calibration train/test subsets once and reuse them.
    """
    if not cfg.calibration.enabled or dataset_key not in TEST_ONLY_DATASETS:
        return None, None

    num_samples = cfg.calibration.get("num_samples")
    if num_samples is None:
        return None, None

    source_csv = cfg.calibration.get("source_csv") or cfg.datasets.get("subset_csv")
    if source_csv is None:
        source_csv = Path(cfg.data.data_root) / "UCB" / "ucb_test.csv"
    source_csv = Path(source_csv)

    generated_dir = (
        Path(cfg.splitting.get("generated_root", "outputs/generated_splits"))
        / dataset_key
        / "calibration"
        / str(cfg.data.target_task)
        / f"n{num_samples}_seed{cfg.calibration.seed}"
    )
    train_csv = generated_dir / "train.csv"
    test_csv = generated_dir / "test.csv"

    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    generated_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_csv)
    if num_samples <= 0 or num_samples >= len(df):
        raise ValueError(
            f"calibration.num_samples must be between 1 and {len(df) - 1}, got {num_samples}"
        )

    cal_df = df.sample(n=num_samples, random_state=cfg.calibration.seed).reset_index(drop=True)
    cal_ids = set(_split_ids_from_df(cal_df))
    remaining_df = df[~_split_ids_from_df(df).isin(cal_ids)].reset_index(drop=True)

    cal_df.to_csv(train_csv, index=False)
    remaining_df.to_csv(test_csv, index=False)
    return train_csv, test_csv


def ensure_target_specific_splits(cfg, dataset_key: str, dataset_folder: str) -> Path:
    """
    Resolve the split directory for the active target task.

    For TG-GATES abnormality runs, generate the split files once if they do not
    already exist. Hypertrophy continues to fall back to the legacy split files.
    """
    split_root = Path(cfg.data.data_root) / dataset_folder / "Splits"
    target_task = str(cfg.data.target_task).lower()
    generated_root = Path(cfg.splitting.get("generated_root", "outputs/generated_splits"))

    target_dir = split_root / target_task
    if _split_files_exist(target_dir):
        return target_dir, "target_specific_existing"

    generated_target_dir = generated_root / dataset_key / target_task
    if _split_files_exist(generated_target_dir):
        return generated_target_dir, "generated_existing"

    legacy_dir = split_root
    if dataset_key in TEST_ONLY_DATASETS:
        return legacy_dir, "test_only_dataset"

    if target_task == "liver_hypertrophy" and _split_files_exist(legacy_dir):
        return legacy_dir, "legacy_default"

    if dataset_key == "tggates" and target_task == "any_abnormality":
        from split import (
            group_labels_per_compound,
            prepare_splits_files,
            repeat_partitions,
            summarize_three_splits,
        )

        generated_target_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(cfg.data.metadata_csv)
        labels_by_drug, _ = group_labels_per_compound(df, cfg.data.organ)

        train_val_drugs, test_drugs, train_val_groups, test_groups, counter_train_val, counter_test = repeat_partitions(
            labels_by_drug,
            num_repeats=cfg.splitting.get("num_repeats", 1000),
        )
        train_drugs, val_drugs, train_groups, val_groups, counter_train, counter_val = repeat_partitions(
            train_val_groups,
            num_repeats=cfg.splitting.get("num_repeats", 1000),
        )

        prepare_splits_files(
            df,
            cfg.data.organ,
            train_drugs,
            val_drugs,
            test_drugs,
            output_dir=generated_target_dir,
        )
        summarize_three_splits(
            counter_train,
            counter_val,
            counter_test,
            generated_target_dir / "summary.csv",
        )
        return generated_target_dir, "generated_new"

    if _split_files_exist(legacy_dir):
        return legacy_dir, "legacy_fallback"

    raise FileNotFoundError(
        f"Could not find split CSVs for dataset='{dataset_key}' and target_task='{target_task}'."
    )


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
    calibration_enabled = bool(cfg.get("calibration", {}).get("enabled", False))
    if key in TEST_ONLY_DATASETS and split != "test" and not calibration_enabled:
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

    # normalizing flow specific parameters
    if args.flow_layers is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_layers = args.flow_layers

    if args.flow_hidden is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_hidden = args.flow_hidden

    if args.flow_input_dim is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_input_dim = args.flow_input_dim

    if args.flow_train_max_tiles is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_train_max_tiles = args.flow_train_max_tiles

    if args.flow_topk_frac is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_topk_frac = args.flow_topk_frac

    if args.flow_tau_percentile is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_tau_percentile = args.flow_tau_percentile

    if args.flow_pca_fit_max_tiles is not None:
        cli.setdefault("probe", OmegaConf.create()).flow_pca_fit_max_tiles = args.flow_pca_fit_max_tiles

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
        "num_workers", "seed",
    ]:
        val = getattr(args, name, None)
        if val is not None:
            cli.runtime[name] = val

    # datasets
    cli.datasets = OmegaConf.create()

    # explicit dataset name (ucb, tggates, etc.)
    if getattr(args, "dataset", None):
        cli.datasets.name = args.dataset

    if getattr(args, "target", None):
        cli.data = OmegaConf.create()
        cli.data.target_task = args.target

    if getattr(args, "experiment_tag", None):
        cli.experiment = OmegaConf.create()
        cli.experiment.tag = args.experiment_tag

    if getattr(args, "calibrate", False):
        cli.calibration = OmegaConf.create()
        cli.calibration.enabled = True

    if getattr(args, "calibration_samples", None) is not None:
        cli.setdefault("calibration", OmegaConf.create()).num_samples = args.calibration_samples

    if getattr(args, "calibration_source_csv", None):
        cli.setdefault("calibration", OmegaConf.create()).source_csv = args.calibration_source_csv

    if getattr(args, "calibration_seed", None) is not None:
        cli.setdefault("calibration", OmegaConf.create()).seed = args.calibration_seed

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


def resolve_target_definition(cfg):
    """
    Resolve a high-level target task into the concrete label definition used
    by prepare_dataset.py.
    """
    task = str(cfg.data.get("target_task", "liver_hypertrophy")).lower()
    dataset = str(cfg.datasets.name).lower()

    cfg.setdefault("data", OmegaConf.create())

    if task == "liver_hypertrophy":
        if dataset == "ucb":
            cfg.data.target_mode = "column"
            cfg.data.target_column = "Hypertrophy"
            cfg.data.target_positive_value = True
            cfg.data.target_finding = None
        else:
            cfg.data.target_mode = "finding"
            cfg.data.target_finding = "hypertrophy"
            cfg.data.target_column = None
            cfg.data.target_positive_value = None
        return cfg

    if task == "any_abnormality":
        if dataset == "ucb":
            cfg.data.target_mode = "column"
            cfg.data.target_column = "No microscopic finding"
            cfg.data.target_positive_value = False
            cfg.data.target_finding = None
        else:
            cfg.data.target_mode = "any_abnormality"
            cfg.data.target_column = None
            cfg.data.target_positive_value = None
            cfg.data.target_finding = None
        return cfg

    raise ValueError(
        f"Unsupported target task '{task}'. "
        "Use 'liver_hypertrophy' or 'any_abnormality'."
    )


def log_config_resolution(cfg, split_dir: Path, split_source: str, dirs: dict):
    target_mode = cfg.data.target_mode
    target_column = cfg.data.get("target_column")
    target_finding = cfg.data.get("target_finding")
    positive_value = cfg.data.get("target_positive_value")
    backend = dirs.get("backend", "legacy")

    print(
        f"[Config] Dataset={cfg.datasets.name} | Split={cfg.datasets.split} | "
        f"TargetTask={cfg.data.target_task}"
    )
    print(
        f"[Config] Calibration -> enabled={cfg.calibration.enabled}, "
        f"num_samples={cfg.calibration.get('num_samples')}, "
        f"base_dataset={cfg.calibration.get('base_dataset')}, "
        f"init_from_base={cfg.calibration.get('init_from_base')}"
    )
    print(
        f"[Config] Target definition -> mode={target_mode}, "
        f"column={target_column}, finding={target_finding}, positive_value={positive_value}"
    )
    print(f"[Config] Split directory -> {split_dir} ({split_source})")
    print(f"[Config] Active subset CSV -> {cfg.datasets.get('subset_csv')}")
    print(f"[Config] Feature backend -> {backend}")
    if backend == "feature_bank":
        print(f"[Config] Feature bank root -> {cfg.features.bank_root}")
        print(f"[Config] Feature registry -> {cfg.features.registry_path}")
        print(f"[Config] Local bank root -> {cfg.features.get('local_bank_root')}")
        print(f"[Config] Raw bank shard root -> {cfg.data.raw_slide_dir}")
        print(f"[Config] Derived bank shard root -> {cfg.data.slide_dir}")
    else:
        print(
            f"[Config] Raw feature directory -> {cfg.data.raw_slide_dir} "
            f"({cfg.data.raw_slide_dir_source})"
        )
    print(f"[Config] Experiment tag -> {cfg.experiment.tag}")


def _format_tag_value(value):
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("/", "_")


def build_experiment_tag(cfg, include_calibration: bool = True):
    parts = [
        f"bs{_format_tag_value(cfg.runtime.batch_size)}",
        f"ep{_format_tag_value(cfg.runtime.epochs)}",
        f"lr{_format_tag_value(cfg.runtime.lr)}",
        f"wd{_format_tag_value(cfg.runtime.weight_decay)}",
        f"seed{_format_tag_value(cfg.runtime.get('seed', 42))}",
    ]

    probe_type = str(cfg.probe.type).lower()
    if probe_type == "flow":
        parts.extend([
            f"fdim{_format_tag_value(cfg.probe.flow_input_dim)}",
            f"fl{_format_tag_value(cfg.probe.flow_layers)}",
            f"fh{_format_tag_value(cfg.probe.flow_hidden)}",
            f"ftiles{_format_tag_value(cfg.probe.flow_train_max_tiles)}",
            f"ftopk{_format_tag_value(cfg.probe.flow_topk_frac)}",
            f"ftau{_format_tag_value(cfg.probe.flow_tau_percentile)}",
        ])
    elif probe_type == "mlp":
        parts.extend([
            f"hd{_format_tag_value(cfg.probe.hidden_dim)}",
            f"layers{_format_tag_value(cfg.probe.num_layers)}",
        ])
    elif probe_type == "knn":
        parts.append(f"knn{_format_tag_value(cfg.probe.knn_neighbors)}")

    if include_calibration and cfg.calibration.get("enabled", False):
        num_samples = cfg.calibration.get("num_samples")
        if num_samples is not None:
            parts.append(f"cal{_format_tag_value(num_samples)}")
        parts.append(f"cseed{_format_tag_value(cfg.calibration.get('seed', 42))}")

    return "__".join(parts)


def resolve_experiment_tag(cfg):
    """
    Build a deterministic experiment tag from the active hyperparameters.
    """
    cfg.setdefault("experiment", OmegaConf.create())
    cfg.setdefault("calibration", OmegaConf.create())
    cfg.calibration.base_experiment_tag = build_experiment_tag(
        cfg, include_calibration=False
    )
    current = cfg.experiment.get("tag")
    if current not in {None, "", "auto"}:
        return cfg

    cfg.experiment.tag = build_experiment_tag(cfg, include_calibration=True)
    return cfg


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
    cfg.setdefault("calibration", OmegaConf.create())

    # 2) resolve dataset configuration
    dataset_key = infer_dataset_key(cfg)
    cfg.datasets.name = dataset_key

    # dataset used for training (default = same as eval dataset)
    cfg.datasets.train_name = cfg.datasets.name

    # test-only datasets always use TG-GATES-trained models
    if cfg.datasets.name in TEST_ONLY_DATASETS:
        cfg.datasets.train_name = (
            cfg.datasets.name if cfg.calibration.enabled else "tggates"
        )


    dataset_info = DATASET_REGISTRY[dataset_key]
    dataset_folder = dataset_info["folder"]

    # metadata + split CSVs
    cfg.data.metadata_csv = (
        Path(cfg.data.data_root) / dataset_info["metadata"]
    )

    # 2b) resolve the target label definition for the selected dataset
    cfg = resolve_target_definition(cfg)

    calib_train_csv, calib_test_csv = ensure_calibration_subsets(cfg, dataset_key)

    split_dir, split_source = ensure_target_specific_splits(
        cfg, dataset_key, dataset_folder
    )
    cfg.datasets.train = split_dir / "train.csv"
    cfg.datasets.val = split_dir / "val.csv"
    cfg.datasets.test = split_dir / "test.csv"

    if cfg.calibration.enabled and dataset_key in TEST_ONLY_DATASETS:
        if cfg.datasets.split == "train" and calib_train_csv is not None:
            cfg.datasets.use_subset = True
            cfg.datasets.subset_csv = calib_train_csv
            split_source = f"{split_source}+calibration_train"
        elif cfg.datasets.split == "test" and calib_test_csv is not None:
            cfg.datasets.use_subset = True
            current_subset = cfg.datasets.get("subset_csv")
            calibration_source = cfg.calibration.get("source_csv")
            if calibration_source is None:
                calibration_source = Path(cfg.data.data_root) / "UCB" / "ucb_test.csv"
            if (
                not current_subset
                or Path(str(current_subset)) == Path(str(calibration_source))
            ):
                cfg.datasets.subset_csv = calib_test_csv
            split_source = f"{split_source}+calibration_holdout"

    # 3) probe aggregation method
    if cfg.probe.type.lower() in {"abmil", "clam", "dsmil", "flow"}:
        cfg.aggregation.type = "MIL"
    

    # 4) resolve feature storage based on configured backend
    dirs = resolve_feature_storage(cfg, dataset_key, dataset_folder)

    cfg.data.raw_slide_dir = str(dirs["raw_slide_dir"])
    cfg.data.raw_slide_dir_source = dirs["raw_slide_dir_source"]
    cfg.data.slide_dir = str(dirs["slide_dir"])
    cfg.data.animal_dir = str(dirs["animal_dir"])
    cfg.features.registry_path = (
        str(dirs["registry_path"]) if dirs["registry_path"] is not None else None
    )
    cfg.features.bank_root = (
        str(dirs["bank_root"]) if dirs["bank_root"] is not None else cfg.features.get("bank_root")
    )
    cfg.features.local_bank_root = (
        str(dirs["local_bank_root"]) if dirs["local_bank_root"] is not None else cfg.features.get("local_bank_root")
    )

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

    # 7) finalize experiment tag after all overrides are applied
    cfg = resolve_experiment_tag(cfg)

    log_config_resolution(
        cfg,
        split_dir=split_dir,
        split_source=split_source,
        dirs=dirs,
    )

    return cfg
