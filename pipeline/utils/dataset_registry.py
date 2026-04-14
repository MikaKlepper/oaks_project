from __future__ import annotations

from pathlib import Path

# Note: this module is responsible for inferring dataset-related configuration settings,
# such as dataset folder, metadata location, target definition, and split resolution.
# It does not handle feature directory resolution, which is done separately by the registry.

DATASET_REGISTRY = {
    "tggates": {"folder": "TG-GATES", "metadata": "TG-GATES/metadata.csv"},
    "ucb": {"folder": "UCB", "metadata": "UCB/metadata.csv"},
}

TEST_ONLY_DATASETS = {"ucb"}
# Note: the "default" rules are applied if the dataset doesn't have specific rules for the task.
TARGET_RULES = {
    "liver_hypertrophy": {
        "ucb": {
            "target_mode": "column",
            "target_column": "Hypertrophy",
            "target_positive_value": True,
            "target_finding": None,
        },
        "tggates": {
            "target_mode": "finding",
            "target_column": None,
            "target_positive_value": None,
            "target_finding": "hypertrophy",
        },
    },
    "any_abnormality": {
        "ucb": {
            "target_mode": "column",
            "target_column": "No microscopic finding",
            "target_positive_value": False,
            "target_finding": None,
        },
        "tggates": {
            "target_mode": "any_abnormality",
            "target_column": None,
            "target_positive_value": None,
            "target_finding": None,
        },
    },
}

# Note: the "default" rules are applied if the dataset doesn't have specific rules for the task.
def _target_rule_for(task: str, dataset: str) -> dict:
    if task not in TARGET_RULES:
        raise ValueError(
            f"Unsupported target task '{task}'. "
            "Use 'liver_hypertrophy' or 'any_abnormality'."
        )
    return TARGET_RULES[task].get(dataset, TARGET_RULES[task]["tggates"])


def infer_dataset_key(cfg) -> str:
    dataset_name = cfg.datasets.get("name")
    if not dataset_name:
        dataset_name = "tggates"

    key = str(dataset_name).lower()
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

# Note: this function is used to apply dataset-specific defaults, such as metadata location. 
# It also returns the dataset folder for convenience, since it's often needed for resolving splits and features.
def apply_dataset_defaults(cfg, dataset_key: str) -> tuple[dict, str]:
    dataset_info = DATASET_REGISTRY[dataset_key]
    dataset_folder = dataset_info["folder"]
    cfg.data.metadata_csv = Path(cfg.data.data_root) / dataset_info["metadata"]
    return dataset_info, dataset_folder

# Note: this function is used to apply train dataset defaults, which can be different from the main dataset for test-only datasets.
# for fine tuning on ucb examples it can be used
def apply_train_dataset_defaults(cfg) -> None:
    if cfg.calibration.enabled:
        cfg.datasets.train_name = "ucb"
        return
    if cfg.datasets.name in TEST_ONLY_DATASETS:
        cfg.datasets.train_name = "tggates"
        return
    cfg.datasets.train_name = cfg.datasets.name

# Note: this function is used to resolve the target definition based on the task and dataset.
def resolve_target_definition(cfg):
    task = str(cfg.data.get("target_task", "liver_hypertrophy")).lower()
    dataset = str(cfg.datasets.name).lower()
    rule = _target_rule_for(task, dataset)
    for key, value in rule.items():
        setattr(cfg.data, key, value)
    return cfg
