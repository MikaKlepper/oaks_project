from __future__ import annotations

from pathlib import Path

from utils.feature_bank_registry import FeatureBankRegistry


MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}


def build_registry_from_cfg(cfg) -> FeatureBankRegistry:
    shared_root = Path(getattr(cfg.features, "bank_root", "feature_bank"))
    registry_path = Path(
        getattr(
            cfg.features,
            "registry_path",
            shared_root / "registry" / "features.sqlite",
        )
    )
    local_root = getattr(cfg.features, "local_bank_root", None)
    prefer_local = bool(getattr(cfg.features, "prefer_local_mirror", True))
    return FeatureBankRegistry(
        db_path=registry_path,
        shared_bank_root=shared_root,
        local_bank_root=local_root,
        prefer_local=prefer_local,
    )


def _base_feature_bank_info(registry: FeatureBankRegistry) -> dict:
    return {
        "feature_backend": "feature_bank",
        "feature_bank_root": str(registry.shared_bank_root),
        "feature_bank_local_root": (
            str(registry.local_bank_root) if registry.local_bank_root else None
        ),
        "feature_registry_path": str(registry.db_path),
        "active_feature_entries": {},
        "missing_required_ids": [],
        "feature_entry_mode": None,
        "raw_feature_entries": None,
        "feature_entries": None,
        "missing_raw_feature_ids": [],
        "missing_feature_ids": [],
    }


def _set_raw_entries(feature_bank_info: dict, entries: dict, missing_ids: list[str]) -> dict:
    feature_bank_info["active_feature_entries"] = entries
    feature_bank_info["missing_required_ids"] = missing_ids
    feature_bank_info["feature_entry_mode"] = "raw"
    feature_bank_info["raw_feature_entries"] = entries
    feature_bank_info["missing_raw_feature_ids"] = missing_ids
    return feature_bank_info


def _set_derived_entries(feature_bank_info: dict, entries: dict, missing_ids: list[str]) -> dict:
    feature_bank_info["active_feature_entries"] = entries
    feature_bank_info["missing_required_ids"] = missing_ids
    feature_bank_info["feature_entry_mode"] = "derived"
    feature_bank_info["feature_entries"] = entries
    feature_bank_info["missing_feature_ids"] = missing_ids
    return feature_bank_info


def resolve_prepared_feature_bank(cfg, df) -> dict:
    registry = build_registry_from_cfg(cfg)
    probe_type = str(cfg.probe.type).lower()
    dataset = str(
        cfg.datasets.get("train_name", cfg.datasets.name)
        if cfg.calibration.enabled and str(cfg.datasets.split).lower() == "train"
        else cfg.datasets.name
    ).lower()
    encoder = str(cfg.features.encoder).upper()
    feature_type = str(cfg.features.feature_type).lower()

    feature_bank_info = _base_feature_bank_info(registry)

    if probe_type in MIL_PROBES:
        slide_ids = sorted(set(df["slide_id"].astype(str).tolist()))
        entries = registry.resolve_feature_entries(
            dataset=dataset,
            encoder=encoder,
            sample_type="slide",
            sample_ids=slide_ids,
            storage_kind="raw",
            aggregation="none",
        )
        missing = sorted(set(slide_ids) - set(entries.keys()))
        return _set_raw_entries(feature_bank_info, entries, missing)

    sample_ids = (
        df["subject_organ_UID"].astype(str).tolist()
        if feature_type == "animal"
        else df["slide_id"].astype(str).tolist()
    )
    entries = registry.resolve_feature_entries(
        dataset=dataset,
        encoder=encoder,
        sample_type=feature_type,
        sample_ids=sample_ids,
        storage_kind="derived",
        aggregation=str(cfg.aggregation.type).lower(),
    )
    missing = sorted(set(map(str, sample_ids)) - set(entries.keys()))
    return _set_derived_entries(feature_bank_info, entries, missing)
