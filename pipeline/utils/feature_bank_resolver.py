from __future__ import annotations

from pathlib import Path

from utils.feature_bank_registry import FeatureBankRegistry


MIL_PROBES = {"abmil", "clam", "dsmil", "flow"}


def feature_bank_enabled(cfg) -> bool:
    return str(getattr(cfg.features, "backend", "legacy")).lower() == "feature_bank"


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


def resolve_prepared_feature_bank(cfg, df) -> dict:
    registry = build_registry_from_cfg(cfg)
    probe_type = str(cfg.probe.type).lower()
    dataset = str(cfg.datasets.name).lower()
    encoder = str(cfg.features.encoder).upper()
    feature_type = str(cfg.features.feature_type).lower()

    result = {
        "feature_backend": "feature_bank",
        "feature_bank_root": str(registry.shared_bank_root),
        "feature_bank_local_root": (
            str(registry.local_bank_root) if registry.local_bank_root else None
        ),
        "feature_registry_path": str(registry.db_path),
        "raw_feature_artifacts": None,
        "feature_artifacts": None,
    }

    if probe_type in MIL_PROBES:
        slide_ids = sorted(set(df["slide_id"].astype(str).tolist()))
        artifacts = registry.resolve_artifacts(
            dataset=dataset,
            encoder=encoder,
            sample_type="slide",
            sample_ids=slide_ids,
            storage_kind="raw",
            aggregation="none",
        )
        missing = sorted(set(slide_ids) - set(artifacts.keys()))
        if missing:
            raise FileNotFoundError(
                f"Feature bank is missing {len(missing)} raw slide artifacts for "
                f"dataset={dataset}, encoder={encoder}. First missing: {missing[:10]}"
            )
        result["raw_feature_artifacts"] = artifacts
        return result

    sample_ids = (
        df["subject_organ_UID"].astype(str).tolist()
        if feature_type == "animal"
        else df["slide_id"].astype(str).tolist()
    )
    artifacts = registry.resolve_artifacts(
        dataset=dataset,
        encoder=encoder,
        sample_type=feature_type,
        sample_ids=sample_ids,
        storage_kind="derived",
        aggregation=str(cfg.aggregation.type).lower(),
    )
    missing = sorted(set(map(str, sample_ids)) - set(artifacts.keys()))
    if missing:
        raise FileNotFoundError(
            f"Feature bank is missing {len(missing)} derived artifacts for "
            f"dataset={dataset}, encoder={encoder}, sample_type={feature_type}, "
            f"aggregation={str(cfg.aggregation.type).lower()}. "
            f"First missing: {missing[:10]}"
        )
    result["feature_artifacts"] = artifacts
    return result
