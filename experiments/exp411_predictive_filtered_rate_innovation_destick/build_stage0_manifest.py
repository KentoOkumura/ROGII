"""Build the preregistered exp411 fixed32 Stage 0 sample manifest.

The persistent-well membership is a diagnostic scope imported from exp408.
Everything used for matching controls is target-free: exp226 fold identity,
visible-prefix length, suffix length, and suffix raw-GR missing fraction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_NAME = "exp411_predictive_filtered_rate_innovation_destick"
EXPECTED_PERSISTENT_SHA256 = (
    "ce245abce24dae98d37b6e0a2adf73fa57a29e0e53864bee983aa916238ea51e"
)
EXPECTED_FOLD_DECOMPRESSED_SHA256 = (
    "709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609"
)
PERSISTENT_FOLD_COUNTS = {0: 4, 1: 3, 2: 3, 3: 3, 4: 3}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(label: str, well: str) -> str:
    return hashlib.sha256(f"exp411|{label}|{well}".encode()).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def load_fold_map(path: Path) -> pd.DataFrame:
    observed = sha256_decompressed(path)
    if observed != EXPECTED_FOLD_DECOMPRESSED_SHA256:
        raise ValueError(f"exp226 decompressed SHA changed: {observed}")
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
        chunksize=250_000,
    ):
        pieces.append(chunk.drop_duplicates())
    frame = pd.concat(pieces, ignore_index=True).drop_duplicates()
    counts = frame.groupby("well_id", sort=True)["fold"].nunique()
    if not counts.eq(1).all():
        raise ValueError("an exp226 well appears in multiple folds")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    result = (
        frame.rename(columns={"well_id": "well"})
        .sort_values("well", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(result) != 773 or result["well"].nunique() != 773:
        raise ValueError(f"exp226 fold inventory must contain 773 wells, got {len(result)}")
    return result


def raw_target_free_metadata(train_dir: Path, wells: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well in sorted(wells):
        path = train_dir / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["GR", "TVT_input"])
        visible = frame["TVT_input"].notna().to_numpy()
        suffix = ~visible
        if not suffix.any() or not visible.any():
            raise ValueError(f"{well}: expected visible prefix and hidden suffix")
        raw_gr = pd.to_numeric(frame.loc[suffix, "GR"], errors="coerce").to_numpy()
        rows.append(
            {
                "well": well,
                "prefix_rows": int(visible.sum()),
                "suffix_rows": int(suffix.sum()),
                "suffix_raw_gr_missing_fraction": float((~np.isfinite(raw_gr)).mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    for source, target in (
        ("suffix_rows", "suffix_row_count_quartile"),
        ("suffix_raw_gr_missing_fraction", "raw_gr_missing_quartile"),
        ("prefix_rows", "prefix_row_count_quartile"),
    ):
        rank = result[source].rank(method="first")
        result[target] = pd.qcut(rank, q=4, labels=False).astype(np.int8)
    return result


def select_manifest(
    metadata: pd.DataFrame,
    persistent_wells: set[str],
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["persistent_scope"] = frame["well"].isin(persistent_wells)
    selected_persistent: list[pd.Series] = []
    for fold, required in PERSISTENT_FOLD_COUNTS.items():
        eligible = frame.loc[
            frame["persistent_scope"] & frame["fold"].eq(fold)
        ].copy()
        eligible["selection_hash"] = eligible["well"].map(
            lambda well: stable_hash("persistent", str(well))
        )
        eligible = eligible.sort_values(
            ["selection_hash", "well"], kind="mergesort"
        )
        if len(eligible) < required:
            raise ValueError(f"fold {fold}: insufficient persistent wells")
        selected_persistent.extend(
            row for _, row in eligible.head(required).iterrows()
        )

    controls = frame.loc[~frame["persistent_scope"]].copy()
    used_controls: set[str] = set()
    output: list[dict[str, Any]] = []
    match_columns = (
        "suffix_row_count_quartile",
        "raw_gr_missing_quartile",
        "prefix_row_count_quartile",
    )
    ordered_persistent = sorted(
        selected_persistent,
        key=lambda row: (int(row["fold"]), stable_hash("persistent", str(row["well"]))),
    )
    for persistent in ordered_persistent:
        persistent_well = str(persistent["well"])
        output.append(
            {
                **persistent.to_dict(),
                "role": "persistent",
                "matched_persistent_well": persistent_well,
                "quartile_match_distance": 0,
                "selection_hash": stable_hash("persistent", persistent_well),
            }
        )
        eligible = controls.loc[
            controls["fold"].eq(int(persistent["fold"]))
            & ~controls["well"].isin(used_controls)
        ].copy()
        if eligible.empty:
            raise ValueError(f"{persistent_well}: no unused control in fold")
        distances = np.zeros(len(eligible), dtype=np.int64)
        for column in match_columns:
            distances += np.abs(
                eligible[column].to_numpy(np.int64) - int(persistent[column])
            )
        eligible["quartile_match_distance"] = distances
        eligible["selection_hash"] = eligible["well"].map(
            lambda well: stable_hash("control", str(well))
        )
        control = eligible.sort_values(
            ["quartile_match_distance", "selection_hash", "well"],
            kind="mergesort",
        ).iloc[0]
        control_well = str(control["well"])
        used_controls.add(control_well)
        output.append(
            {
                **control.to_dict(),
                "role": "control",
                "matched_persistent_well": persistent_well,
            }
        )

    manifest = pd.DataFrame(output)
    columns = [
        "well",
        "role",
        "fold",
        "matched_persistent_well",
        "prefix_rows",
        "suffix_rows",
        "suffix_raw_gr_missing_fraction",
        "suffix_row_count_quartile",
        "raw_gr_missing_quartile",
        "prefix_row_count_quartile",
        "quartile_match_distance",
        "selection_hash",
    ]
    manifest = manifest.loc[:, columns].sort_values(
        ["fold", "role", "selection_hash"], kind="mergesort"
    )
    manifest = manifest.reset_index(drop=True)
    role_counts = manifest["role"].value_counts().to_dict()
    if role_counts != {"persistent": 16, "control": 16}:
        raise ValueError(f"fixed32 role counts changed: {role_counts}")
    if manifest["well"].nunique() != 32:
        raise ValueError("fixed32 wells are not unique")
    expected_fold_counts = {
        fold: count * 2 for fold, count in PERSISTENT_FOLD_COUNTS.items()
    }
    if manifest.groupby("fold").size().to_dict() != expected_fold_counts:
        raise ValueError("fixed32 fold counts changed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--persistent-wells",
        type=Path,
        default=Path(
            "experiments/exp408_hmm_message_rate_basin_audit/assets/target_wells.csv"
        ),
    )
    parser.add_argument(
        "--fold-source",
        type=Path,
        default=Path(
            "/tmp/kaggle-output/"
            "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/"
            "train_v1/artifacts/"
            "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_"
            "train_oof_predictions.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/exp411_predictive_filtered_rate_innovation_destick/assets"
        ),
    )
    args = parser.parse_args()

    persistent_sha = sha256_file(args.persistent_wells)
    if persistent_sha != EXPECTED_PERSISTENT_SHA256:
        raise ValueError(f"exp408 persistent scope SHA changed: {persistent_sha}")
    persistent = set(
        pd.read_csv(args.persistent_wells, dtype={"well": str})["well"].astype(str)
    )
    if len(persistent) != 450:
        raise ValueError(f"expected 450 persistent wells, got {len(persistent)}")
    folds = load_fold_map(args.fold_source)
    raw = raw_target_free_metadata(args.train_dir, folds["well"].astype(str).tolist())
    metadata = folds.merge(raw, on="well", how="inner", validate="one_to_one")
    manifest = select_manifest(metadata, persistent)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "stage0_fixed32_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "manifest": {
            "path": str(manifest_path),
            "rows": len(manifest),
            "file_sha256": sha256_file(manifest_path),
            "logical_sha256": logical_frame_sha256(manifest),
        },
        "inputs": {
            "persistent_scope_path": str(args.persistent_wells),
            "persistent_scope_sha256": persistent_sha,
            "persistent_scope_wells": len(persistent),
            "fold_source_path": str(args.fold_source),
            "fold_source_raw_sha256": sha256_file(args.fold_source),
            "fold_source_decompressed_sha256": sha256_decompressed(args.fold_source),
            "raw_train_dir": str(args.train_dir),
            "raw_wells": len(raw),
        },
        "selection": {
            "persistent_fold_counts": PERSISTENT_FOLD_COUNTS,
            "roles": manifest["role"].value_counts().sort_index().to_dict(),
            "fold_counts": manifest.groupby("fold").size().sort_index().to_dict(),
            "exact_control_quartile_matches": int(
                manifest.loc[manifest["role"].eq("control"), "quartile_match_distance"]
                .eq(0)
                .sum()
            ),
            "maximum_control_quartile_distance": int(
                manifest.loc[
                    manifest["role"].eq("control"), "quartile_match_distance"
                ].max()
            ),
        },
        "leakage_contract": {
            "truth_columns_read": [],
            "error_columns_read": [],
            "episode_columns_read": [],
            "persistent_membership_used_for_scope_only": True,
            "control_matching_columns": [
                "fold",
                "suffix_row_count_quartile",
                "raw_gr_missing_quartile",
                "prefix_row_count_quartile",
            ],
        },
    }
    summary_path = args.output_dir / "stage0_fixed32_manifest_metadata.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
