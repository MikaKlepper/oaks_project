from __future__ import annotations

import sqlite3
from pathlib import Path


class FeatureBankRegistry:
    def __init__(
        self,
        db_path: str | Path,
        shared_bank_root: str | Path,
        local_bank_root: str | Path | None = None,
        prefer_local: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.shared_bank_root = Path(shared_bank_root)
        self.local_bank_root = Path(local_bank_root) if local_bank_root else None
        self.prefer_local = prefer_local

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _materialize_path(self, relative_path: str) -> Path:
        shared_path = self.shared_bank_root / relative_path
        if self.prefer_local and self.local_bank_root is not None:
            local_path = self.local_bank_root / relative_path
            if local_path.exists():
                return local_path
        return shared_path

    def resolve_artifacts(
        self,
        *,
        dataset: str,
        encoder: str,
        sample_type: str,
        sample_ids: list[str],
        storage_kind: str,
        aggregation: str,
    ) -> dict[str, dict]:
        normalized_ids = [str(x) for x in sample_ids]
        if not normalized_ids:
            return {}

        placeholders = ", ".join("?" for _ in normalized_ids)
        query = f"""
            SELECT
                feature_key,
                dataset,
                encoder,
                sample_type,
                sample_id,
                storage_kind,
                aggregation,
                hdf5_relative_path,
                hdf5_key,
                shape_json,
                dtype,
                bytes,
                is_present
            FROM feature_registry
            WHERE dataset = ?
              AND encoder = ?
              AND sample_type = ?
              AND storage_kind = ?
              AND aggregation = ?
              AND sample_id IN ({placeholders})
        """
        params = [
            str(dataset).lower(),
            str(encoder).upper(),
            str(sample_type).lower(),
            str(storage_kind).lower(),
            str(aggregation).lower(),
            *normalized_ids,
        ]

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            columns = [desc[0] for desc in conn.execute(query, params).description]

        result: dict[str, dict] = {}
        for row in rows:
            item = dict(zip(columns, row))
            item["resolved_hdf5_path"] = str(
                self._materialize_path(item["hdf5_relative_path"])
            )
            result[str(item["sample_id"])] = item
        return result
