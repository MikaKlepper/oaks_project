from __future__ import annotations

from omegaconf import OmegaConf


def _put(cli, section: str, key: str, value) -> None:
    if value is None:
        return
    if section not in cli or cli[section] is None:
        cli[section] = OmegaConf.create()
    cli[section][key] = value


def incorporate_cli_args(cfg, args):
    if args is None:
        return cfg

    cli = OmegaConf.create()

    if getattr(args, "model", None):
        _put(cli, "features", "encoder", args.model.upper())
    _put(cli, "features", "feature_type", getattr(args, "ftype", None))

    if getattr(args, "probe", None):
        _put(cli, "probe", "type", args.probe)
    for key in [
        "hidden_dim",
        "flow_layers",
        "flow_hidden",
        "flow_input_dim",
        "flow_train_max_tiles",
        "flow_topk_frac",
        "flow_tau_percentile",
        "flow_pca_fit_max_tiles",
    ]:
        _put(cli, "probe", key, getattr(args, key, None))
    _put(cli, "probe", "num_layers", getattr(args, "layers", None))

    for key in [
        "optimizer",
        "loss",
        "device",
        "lr",
        "batch_size",
        "epochs",
        "momentum",
        "weight_decay",
        "num_workers",
        "seed",
    ]:
        _put(cli, "runtime", key, getattr(args, key, None))

    _put(cli, "datasets", "name", getattr(args, "dataset", None))
    if getattr(args, "target", None):
        _put(cli, "data", "target_task", args.target)
    if getattr(args, "experiment_tag", None):
        _put(cli, "experiment", "tag", args.experiment_tag)

    if getattr(args, "calibrate", False):
        _put(cli, "calibration", "enabled", True)
    _put(cli, "calibration", "num_samples", getattr(args, "calibration_samples", None))
    _put(cli, "calibration", "source_csv", getattr(args, "calibration_source_csv", None))
    _put(cli, "calibration", "seed", getattr(args, "calibration_seed", None))

    _put(cli, "fewshot", "k", getattr(args, "k", None))
    _put(cli, "aggregation", "type", getattr(args, "agg", None))

    stage = getattr(args, "stage", None)
    if stage == "train":
        _put(cli, "datasets", "split", "train")
        if getattr(args, "train_subset_csv", None):
            _put(cli, "datasets", "use_subset", True)
            _put(cli, "datasets", "subset_csv", args.train_subset_csv)
        elif getattr(args, "k", None) is not None and str(getattr(args, "target", "")).lower() == "liver_hypertrophy":
            _put(cli, "datasets", "use_subset", True)
            _put(
                cli,
                "datasets",
                "subset_csv",
                f"{cfg.data.data_root}/TG-GATES/FewShotCompoundBalanced/train_fewshot_k{args.k}.csv",
            )
        else:
            _put(cli, "datasets", "use_subset", False)
    elif stage == "eval":
        _put(cli, "datasets", "split", "val")
        default_eval_subset = None
        if str(getattr(args, "target", "")).lower() == "liver_hypertrophy":
            default_eval_subset = f"{cfg.data.data_root}/TG-GATES/Splits/val.csv"
        eval_subset = getattr(args, "eval_subset_csv", None) or default_eval_subset
        _put(cli, "datasets", "use_subset", bool(eval_subset))
        if eval_subset:
            _put(cli, "datasets", "subset_csv", eval_subset)
    elif stage == "test":
        _put(cli, "datasets", "split", "test")
        default_test_subset = None
        if str(getattr(args, "target", "")).lower() == "liver_hypertrophy":
            default_test_subset = f"{cfg.data.data_root}/TG-GATES/Splits/test.csv"
        test_subset = getattr(args, "test_subset_csv", None) or default_test_subset
        _put(cli, "datasets", "use_subset", bool(test_subset))
        if test_subset:
            _put(cli, "datasets", "subset_csv", test_subset)

    return OmegaConf.merge(cfg, cli)
