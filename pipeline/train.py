import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from argparser import get_args
from data.collate_MIL import collate_mil
from data.create_dataset_MIL import ToxicologyMILDataset
from data.create_datasets import ToxicologyDataset
from data.dataset_check import check_subset_consistency
from data.prepare_dataset import prepare_dataset_inputs
from logger import setup_logger
from probes import FlowProbe, TorchProbe, build_probe, default_probe_path
from utils.config_loader import load_merged_config
from utils.dataset_registry import apply_dataset_defaults, apply_train_dataset_defaults, resolve_target_definition
from utils.experiment_registry import append_experiment_row
from utils.split_resolver import resolve_dataset_splits

MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}


def _is_torch_like(probe) -> bool:
    return isinstance(probe, (TorchProbe, FlowProbe))


def _build_dataset(prepared):
    probe_type = prepared["probe"]["type"].lower()

    if probe_type not in MIL_PROBES:
        check_subset_consistency(prepared)
        return ToxicologyDataset(prepared), None

    dataset = ToxicologyMILDataset(prepared)
    if probe_type == "flow":
        logging.info("[Train] Using FlowProbe, which only uses training NORMAL samples for fitting.")
        normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        dataset = torch.utils.data.Subset(dataset, normal_indices)
        logging.info(f"[Train] Subsetted to {len(dataset)} NORMAL samples for Flow Probe training.")
    return dataset, collate_mil


def _base_experiment_root(cfg) -> Path:
    return Path(
        str(cfg.experiment_root)
        .replace(f"/{cfg.datasets.name}/", f"/{cfg.calibration.base_dataset}/")
        .replace(f"/{cfg.experiment.tag}", f"/{cfg.calibration.base_experiment_tag}")
    )


def _build_base_cfg(cfg, base_exp_root: Path):
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    base_dataset = base_cfg.calibration.base_dataset
    k = base_cfg.fewshot.get("k")

    base_cfg.datasets.name = base_dataset
    base_cfg.datasets.split = "train"
    base_cfg.datasets.use_subset = k is not None
    base_cfg.datasets.subset_csv = (
        f"{base_cfg.data.data_root}/TG-GATES/FewShotCompoundBalanced/train_fewshot_k{k}.csv"
        if k is not None
        else None
    )
    base_cfg.calibration.enabled = False
    base_cfg.calibration.num_samples = None
    base_cfg.calibration.source_csv = None
    base_cfg.experiment.tag = base_cfg.calibration.base_experiment_tag
    base_cfg.experiment_root = str(base_exp_root)

    apply_train_dataset_defaults(base_cfg)
    _, dataset_folder = apply_dataset_defaults(base_cfg, base_dataset)
    base_cfg = resolve_target_definition(base_cfg)
    resolve_dataset_splits(base_cfg, base_dataset, dataset_folder)
    return base_cfg


def _maybe_warm_start_from_base(cfg, prepared, probe):
    if not (cfg.calibration.enabled and cfg.calibration.init_from_base):
        return

    base_exp_root = _base_experiment_root(cfg)
    base_ckpt_path = default_probe_path(prepared, base_exp_root, _is_torch_like(probe))
    if not base_ckpt_path.exists():
        logging.info(
            f"[Train] Calibration mode -> base checkpoint missing at {base_ckpt_path}. "
            "Training base run first."
        )
        run_train(_build_base_cfg(cfg, base_exp_root))
        setup_logger(Path(cfg.experiment_root))

    if not base_ckpt_path.exists():
        raise RuntimeError(
            "Calibration requested, but base checkpoint is still missing after base training: "
            f"{base_ckpt_path}"
        )

    logging.info(f"[Train] Calibration mode -> warm-starting from {base_ckpt_path}")
    probe.load(base_ckpt_path)


def run_train(cfg):
    """Run one training stage and write checkpoint + registry row."""
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== TRAIN ==========")

    prepared = prepare_dataset_inputs(cfg)
    dataset, collate_fn = _build_dataset(prepared)

    input_dim = prepared["data"]["embed_dim"]
    num_classes = prepared["data"]["num_classes"]
    probe = build_probe(prepared, input_dim, num_classes)

    ckpt_path = default_probe_path(prepared, exp_root, _is_torch_like(probe))

    if ckpt_path.exists():
        logging.info("[Train] Checkpoint already exists -> skipping training.")
        return

    _maybe_warm_start_from_base(cfg, prepared, probe)

    if (
        cfg.calibration.enabled
        and not _is_torch_like(probe)
        and len(set(prepared["data"]["labels"])) < 2
    ):
        logging.warning(
            "[Train] Calibration subset has a single class for a sklearn probe. "
            "Skipping fit and saving warm-started checkpoint."
        )
        probe.save(ckpt_path)
        logging.info(f"[Train] Saved checkpoint -> {ckpt_path}")
        logging.info("========== TRAIN DONE ==========")
        return

    logging.info("[Train] Starting training…")
    probe.fit(dataset, collate_fn=collate_fn)
    probe.save(ckpt_path)
    logging.info(f"[Train] Saved checkpoint -> {ckpt_path}")
    logging.info("========== TRAIN DONE ==========")


if __name__ == "__main__":
    args = get_args()
    cfg = load_merged_config(args.config, args=args)
    run_train(cfg)

    import gc

    torch.cuda.empty_cache()
    gc.collect()
    raise SystemExit(0)
