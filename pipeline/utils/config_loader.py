# # pipeline/utils/config_loader.py

# imports
from pathlib import Path
from omegaconf import OmegaConf

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
        cli_cfg.setdefault("features", {})["encoder"] = args.model
        cli_cfg_entries.append(f"features.encoder={args.model}")
    if getattr(args, "probe", None):
        cli_cfg.setdefault("probe", {})["type"] = args.probe
        cli_cfg_entries.append(f"probe.type={args.probe}")
    if getattr(args, "k", None):
        cli_cfg.setdefault("fewshot", {})["k"] = args.k
        cli_cfg_entries.append(f"fewshot.k={args.k}")
    if getattr(args, "agg", "mean"):
        cli_cfg.setdefault("aggregation", {})["type"] = args.agg
        cli_cfg_entries.append(f"aggregation.type={args.agg}")
    if getattr(args, "subset_csv", None):
        ds = cli_cfg.setdefault("datasets", {})
        ds["use_subset"] = True
        ds["subset_csv"] = args.subset_csv
        cli_cfg_entries.append(f"datasets.subset_csv={args.subset_csv}")
    if getattr(args, "subset_fraction", None):
        ds = cli_cfg.setdefault("datasets", {})
        ds["use_subset"] = True
        ds["subset_fraction"] = args.subset_fraction
        cli_cfg_entries.append(f"datasets.subset_fraction={args.subset_fraction}")
    if getattr(args, "stage", None):
        if args.stage == "train":
            cli_cfg.setdefault("datasets", {})["split"] = "train"
            cli_cfg_entries.append("datasets.split=train")
        elif args.stage == "eval":
            cli_cfg.setdefault("datasets", {})["split"] = "val"
            cli_cfg_entries.append("datasets.split=val")
        elif args.stage == "test":
            cli_cfg.setdefault("datasets", {})["split"] = "test"
            cli_cfg_entries.append("datasets.split=test")

    if cli_cfg:
        cfg =OmegaConf.merge(cfg, cli_cfg)

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

    # step 3: determine override files after cli
    encoder = cfg["features"]["encoder"]
    probe = cfg["probe"]["type"]
    k = cfg["fewshot"]["k"]

    model_override_path = config_path.parent / f"models/{encoder}.yaml"
    probe_override_path = config_path.parent / f"probes/{probe}.yaml"
    fewshot_override_path = config_path.parent / f"fewshot/k{k}.yaml" if k is not None else None

    override_paths = [model_override_path, probe_override_path, fewshot_override_path]

    # step 4: load overrides
    override_cfgs = [OmegaConf.load(p) for p in override_paths if p is not None and p.exists()]
    if override_cfgs:
        cfg = OmegaConf.merge(cfg, *override_cfgs)

    yaml_entries= [str(p.relative_to(config_path.parent)) for p in override_paths if p is not None and p.exists()]
    cfg.override = [f"cli: {e}" for e in cli_cfg_entries] + yaml_entries

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

