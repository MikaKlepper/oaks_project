# utils/config_loader.py

from pathlib import Path

from omegaconf import OmegaConf

from logger import log_config_resolution
from utils.cli_overrides import incorporate_cli_args
from utils.dataset_registry import (
    apply_dataset_defaults,
    apply_train_dataset_defaults,
    infer_dataset_key,
    resolve_target_definition,
)
from utils.split_resolver import resolve_dataset_splits


def load_merged_config(config_path, args=None):
    """
    Load base YAML config, apply CLI overrides,
    resolve dataset configuration, feature directories,
    and finalize model settings.
    """
    config_path = Path(config_path)
    base_cfg = OmegaConf.load(config_path)
    
    # Incorporate CLI overrides
    cfg = incorporate_cli_args(base_cfg, args)
    for section in ("datasets", "calibration", "features", "runtime"):
        cfg.setdefault(section, OmegaConf.create())

    cfg.features.backend = "feature_bank" # always use feature bank backend; registry will point to correct location

    # Infer dataset key and apply dataset-specific defaults
    dataset_key = infer_dataset_key(cfg)
    cfg.datasets.name = dataset_key

    # Apply train dataset defaults, which can differ from the main dataset for test-only datasets.
    apply_train_dataset_defaults(cfg)

    # give the metadata and the corresponding dataset folder to the rest of the pipeline, 
    # since they are often needed for resolving splits and features
    _, dataset_folder = apply_dataset_defaults(cfg, dataset_key)
    
    # Resolve target definition, so "target_task" is all_abnormality vs liver_hypertrophy 
    cfg = resolve_target_definition(cfg)
    
    # Resolve dataset splits
    split_dir = resolve_dataset_splits(cfg, dataset_key, dataset_folder)

    # Resolve aggregation type
    if cfg.probe.type.lower() in {"abmil", "clam", "dsmil", "flow"}:
        cfg.aggregation.type = "MIL"

    # Resolve feature directories and registry path, applying defaults if not specified.
    cfg.features.bank_root = cfg.features.get("bank_root") or "feature_bank"
    cfg.features.local_bank_root = cfg.features.get("local_bank_root") or None
    cfg.features.registry_path = (
        cfg.features.get("registry_path")
        or str(Path(cfg.features.bank_root) / "registry" / "features.sqlite")
    )

    # Resolve feature embedding dimension
    pipeline_root = Path(__file__).resolve().parents[1]
    enc_cfg = OmegaConf.load(pipeline_root / "configs" / "models" / "encoder_dims.yaml")
    cfg.features.embed_dim = enc_cfg.encoder_dims[cfg.features.encoder]

    # Merge probe-specific config if it exists
    probe_yaml = pipeline_root / "configs" / "probes" / f"{cfg.probe.type}.yaml"
    if probe_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(probe_yaml))

    # Set experiment tag
    cfg.setdefault("experiment", OmegaConf.create())
    cfg.setdefault("calibration", OmegaConf.create())
    cfg.setdefault("runtime", OmegaConf.create())
    cfg.calibration.setdefault("base_experiment_tag", "default")
    if cfg.calibration.get("seed") is None:
        cfg.calibration.seed = 42
    if cfg.runtime.get("seed") is None:
        cfg.runtime.seed = 42
    if cfg.experiment.get("tag") in {None, "", "auto"}:
        num_samples = cfg.calibration.get("num_samples")
        cfg.experiment.tag = (
            f"calibration_n{num_samples}"
            if cfg.calibration.get("enabled", False) and num_samples is not None
            else ("calibration" if cfg.calibration.get("enabled", False) else "default")
        )

    log_config_resolution(
        cfg,
        split_dir=split_dir,
    )

    return cfg
