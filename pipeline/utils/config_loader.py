# # pipeline/utils/config_loader.py

# imports
from pathlib import Path
from omegaconf import OmegaConf
from .dir_builder import build_feature_dirs



# helper function for cli args
def incorporate_cli_args(cfg, args):
    """
    Incorpororporate CLI arguments into a config object.

    Args:
        cfg (OmegaConf): The config object to update.
        args (argparse.Namespace): The CLI arguments to incorporate.

    Returns:
        cfg (OmegaConf): The updated config object.
        cli_cfg_entries (list[str]): A list of strings representing the
            updated config entries.
    """

    cli_cfg = {}
    cli_cfg_entries = []

    if getattr(args, "model", None):
        cli_cfg.setdefault("features", {})["encoder"] = args.model.upper()
        cli_cfg_entries.append(f"features.encoder={args.model}")

    if getattr(args, "probe", None):
        cli_cfg.setdefault("probe", {})["type"] = args.probe
        cli_cfg_entries.append(f"probe.type={args.probe}")

    # store k but DO NOT assign default subset yet
    k = getattr(args, "k", None)
    if k is not None:
        cli_cfg.setdefault("fewshot", {})["k"] = k
        cli_cfg_entries.append(f"fewshot.k={k}")

    if getattr(args, "agg", None) is not None:
        cli_cfg.setdefault("aggregation", {})["type"] = args.agg
        cli_cfg_entries.append(f"aggregation.type={args.agg}")

    if getattr(args, "ftype", None):
        cli_cfg.setdefault("features", {})["type"] = args.ftype
        cli_cfg_entries.append(f"features.type={args.ftype}")

    # set training hyperparameters
    if getattr(args, "optimizer", None):
        cli_cfg.setdefault("train", {})["optimizer"] = args.optimizer
        cli_cfg_entries.append(f"train.optimizer={args.optimizer}")
    if getattr(args, "loss", None):
        cli_cfg.setdefault("train", {})["loss"] = args.loss
        cli_cfg_entries.append(f"train.loss={args.loss}")
    if getattr(args, "weight_decay", None) is not None:
        cli_cfg.setdefault("train", {})["weight_decay"] = args.weight_decay
        cli_cfg_entries.append(f"train.weight_decay={args.weight_decay}")
    if getattr(args, "momentum", None) is not None:
        cli_cfg.setdefault("train", {})["momentum"] = args.momentum
        cli_cfg_entries.append(f"train.momentum={args.momentum}")
    if getattr(args, "device", None):
        cli_cfg.setdefault("train", {})["device"] = args.device
        cli_cfg_entries.append(f"train.device={args.device}")

    # handle dataset subset selection with the following
    # priority:
    #   1) subset_csv
    #   2) subset_fraction
    #   3) automatic k-default
    #   4) nothing -> full dataset

    ds = cli_cfg.setdefault("datasets", {})

    # stage handling (unchanged)
    if getattr(args, "stage", None):
        stage = args.stage
        
        ds = cli_cfg.setdefault("datasets", {})
        
        if stage == "train":
            ds["split"] = "train"
            cli_cfg_entries.append("datasets.split=train")

        elif stage == "eval":
            ds["split"] = "val"
            ds["use_subset"] = True
            ds["subset_csv"] = f"{cfg.data.data_root}/Subsets/val_balanced_subset.csv"
            cli_cfg_entries.append("datasets.split=val")
            cli_cfg_entries.append(f"datasets.subset_csv={ds['subset_csv']}")
            cli_cfg_entries.append("datasets.use_subset=True")

        elif stage == "test":
            ds["split"] = "test"
            ds["use_subset"] = True
            #ds["subset_csv"] = f"{cfg.data.data_root}/Subsets/test_balanced_subset.csv"
            cli_cfg_entries.append("datasets.split=test")
            #cli_cfg_entries.append(f"datasets.subset_csv={ds['subset_csv']}")
            cli_cfg_entries.append("datasets.use_subset=True")

        elif stage == "all":
            ds["split"] = "all"
            cli_cfg_entries.append("datasets.split=all")
        
    if k is not None:
        stage_now = cli_cfg.get("datasets", {}).get("split", None)

        if stage_now == "train":
            ds = cli_cfg.setdefault("datasets", {})
            fewshot_csv = f"{cfg.data.data_root}/FewShotCompoundBalanced/train_fewshot_k{k}.csv"
            ds["use_subset"] = True
            ds["subset_csv"] = fewshot_csv
            cli_cfg_entries.append(f"datasets.subset_csv={fewshot_csv}")
            cli_cfg_entries.append("datasets.use_subset=True")

     # custom subset_csv -> ALWAYS wins
    if getattr(args, "subset_csv", None):
        ds["use_subset"] = True
        ds["subset_csv"] = args.subset_csv
        cli_cfg_entries.append(f"datasets.subset_csv={args.subset_csv}")
        cli_cfg_entries.append("datasets.use_subset=True")

    # 2) subset fraction overrides automatic k-default
    elif getattr(args, "subset_fraction", None) is not None:
        ds["use_subset"] = True
        ds["subset_fraction"] = args.subset_fraction
        cli_cfg_entries.append(f"datasets.subset_fraction={args.subset_fraction}")
        cli_cfg_entries.append("datasets.use_subset=True")

    # merge
    if cli_cfg:
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg, cli_cfg_entries


def load_merged_config(config_path, args=None):
    """
    Load a merged config object from a base yaml file and apply CLI arguments.

    The function follows the following steps:

    1. Load the base yaml file into an OmegaConf object.
    2. Apply CLI arguments to the config object.
    3. Determine override files based on the CLI arguments.
    4. Load the override files into separate OmegaConf objects.
    5. Merge the override objects into the base config object.
    6. Save the final merged config object to a yaml file.

    Args:
        config_path (Path): The base config path.
        args (argparse.Namespace): The CLI arguments to apply.

    Returns:
        cfg (OmegaConf): The final merged config object.
    """
    config_path = Path(config_path) # base config path

    # step 1 : load base yaml file
    base_config = OmegaConf.load(config_path)

    # step 2 : apply cli args
    cfg, cli_cfg_entries = incorporate_cli_args(base_config, args)

    # build feature directories and update config
    dirs = build_feature_dirs(cfg.features.features_root, cfg.features.encoder)

    # Assign dynamic dirs back into config
    for key, path in dirs.items():
        cfg.features[key] = str(path)

    # step 3: determine override files after cli
    encoder = cfg["features"]["encoder"]
    probe = cfg["probe"]["type"]
    #k = cfg["fewshot"]["k"]

    # step 4 : load encoder dims file (NO merge)
    encoder_dims_path = config_path.parent / "models/encoder_dims.yaml"
    encoder_dims_cfg = OmegaConf.load(encoder_dims_path)
    cfg.features.embed_dim = encoder_dims_cfg.encoder_dims[encoder]

    # 5) load probe 
    probe_override_path = config_path.parent / f"probes/{probe}.yaml"

    yaml_entries= []
    if probe_override_path.exists():
        probe_override = OmegaConf.load(probe_override_path)
        cfg = OmegaConf.merge(cfg, probe_override)
        yaml_entries.append(str(probe_override_path.relative_to(config_path.parent)))

    cfg.user_input = [f"cli: {e}" for e in cli_cfg_entries] + yaml_entries

    # model_override_path = config_path.parent / f"models/{encoder}.yaml"
    # probe_override_path = config_path.parent / f"probes/{probe}.yaml"
    # #fewshot_override_path = config_path.parent / f"fewshot/k{k}.yaml" if k is not None else None

    # override_paths = [model_override_path, probe_override_path]

    # # step 4: load overrides
    # override_cfgs = [OmegaConf.load(p) for p in override_paths if p is not None and p.exists()]
    # if override_cfgs:
    #     cfg = OmegaConf.merge(cfg, *override_cfgs)

    # yaml_entries= [str(p.relative_to(config_path.parent)) for p in override_paths if p is not None and p.exists()]
    # cfg.user_input = [f"cli: {e}" for e in cli_cfg_entries] + yaml_entries

    # step 5: apply placeholders and convert to python dict
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    # step 6: save final merged config
    exp_root = Path(cfg_resolved["experiment_root"])
    exp_root.mkdir(parents=True, exist_ok=True)
    config_out = exp_root / "config.yaml"

    final_cfg = OmegaConf.create(cfg_resolved)
    OmegaConf.save(config=final_cfg, f=config_out)
    print(f"[CONFIG] Saved final merged config -> {config_out}")

    return final_cfg,cfg_resolved

