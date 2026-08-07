"""Build the preregistered exp412 cause-stratified fixed32 manifest.

Cause membership is used only to choose the diagnostic sample.  Control
matching uses target-free fold identity, visible-prefix length, suffix length,
and suffix raw-GR missing fraction.  The resulting role/fold columns must not
be passed to either HMM pass.
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

EXPERIMENT_NAME = "exp412_beta_filter_rate_disagreement_two_pass_reset"
EXPECTED_EPISODE_SUMMARY_SHA256 = (
    "b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0"
)
EXPECTED_PERSISTENT_SCOPE_SHA256 = (
    "ce245abce24dae98d37b6e0a2adf73fa57a29e0e53864bee983aa916238ea51e"
)
EXPECTED_FOLD_DECOMPRESSED_SHA256 = (
    "709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609"
)
CAUSES = {
    "backward_cause": "backward_smoothing_reversal",
    "forward_cause": "forward_transition_prior_hysteresis",
}
CAUSE_WELLS_PER_ROLE = 8


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
    return hashlib.sha256(f"exp412|{label}|{well}".encode()).hexdigest()


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


def load_cause_inventory(path: Path) -> pd.DataFrame:
    observed = sha256_file(path)
    if observed != EXPECTED_EPISODE_SUMMARY_SHA256:
        raise ValueError(f"exp408 episode summary SHA changed: {observed}")
    frame = pd.read_csv(
        path,
        usecols=["well", "fold", "cause"],
        dtype={"well": str},
    ).drop_duplicates()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    counts = frame.groupby("well", sort=True)["fold"].nunique()
    if not counts.eq(1).all():
        raise ValueError("an exp408 cause well appears in multiple folds")
    return frame.sort_values(["cause", "fold", "well"], kind="mergesort")


def select_cause_rows(
    metadata: pd.DataFrame,
    cause_inventory: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    metadata_by_well = metadata.set_index("well", drop=False)
    for role, cause in CAUSES.items():
        eligible = (
            cause_inventory.loc[cause_inventory["cause"].eq(cause), ["well", "fold"]]
            .drop_duplicates("well")
            .copy()
        )
        eligible = eligible.loc[~eligible["well"].isin(used)].copy()
        eligible["selection_hash"] = eligible["well"].map(
            lambda well: stable_hash(role.removesuffix("_cause"), str(well))
        )
        role_wells: list[str] = []
        for fold in range(5):
            fold_rows = eligible.loc[
                eligible["fold"].eq(fold) & ~eligible["well"].isin(role_wells)
            ].sort_values(["selection_hash", "well"], kind="mergesort")
            if fold_rows.empty:
                raise ValueError(f"{role}: fold {fold} has no unused eligible well")
            role_wells.append(str(fold_rows.iloc[0]["well"]))
        remaining = eligible.loc[~eligible["well"].isin(role_wells)].sort_values(
            ["selection_hash", "well"], kind="mergesort"
        )
        needed = CAUSE_WELLS_PER_ROLE - len(role_wells)
        if len(remaining) < needed:
            raise ValueError(f"{role}: insufficient remaining cause wells")
        role_wells.extend(remaining.head(needed)["well"].astype(str).tolist())
        for well in role_wells:
            if well in used:
                raise ValueError(f"cause well selected twice: {well}")
            used.add(well)
            row = metadata_by_well.loc[well].to_dict()
            selected.append(
                {
                    **row,
                    "role": role,
                    "cause": cause,
                    "matched_cause_well": well,
                    "matched_cause_role": role,
                    "quartile_match_distance": 0,
                    "selection_hash": stable_hash(
                        role.removesuffix("_cause"), well
                    ),
                }
            )
    return selected


def select_manifest(
    metadata: pd.DataFrame,
    cause_inventory: pd.DataFrame,
    persistent_wells: set[str],
) -> pd.DataFrame:
    cause_rows = select_cause_rows(metadata, cause_inventory)
    controls = metadata.loc[~metadata["well"].isin(persistent_wells)].copy()
    if len(controls) != 323:
        raise ValueError(f"nonpersistent control pool must contain 323 wells, got {len(controls)}")
    used_controls: set[str] = set()
    output = list(cause_rows)
    match_columns = (
        "suffix_row_count_quartile",
        "raw_gr_missing_quartile",
        "prefix_row_count_quartile",
    )
    for cause_row in sorted(
        cause_rows,
        key=lambda row: (
            int(row["fold"]),
            str(row["role"]),
            str(row["selection_hash"]),
        ),
    ):
        eligible = controls.loc[
            controls["fold"].eq(int(cause_row["fold"]))
            & ~controls["well"].isin(used_controls)
        ].copy()
        if eligible.empty:
            raise ValueError(f"{cause_row['well']}: no unused control in fold")
        distance = np.zeros(len(eligible), dtype=np.int64)
        for column in match_columns:
            distance += np.abs(
                eligible[column].to_numpy(np.int64) - int(cause_row[column])
            )
        eligible["quartile_match_distance"] = distance
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
                "cause": "",
                "matched_cause_well": str(cause_row["well"]),
                "matched_cause_role": str(cause_row["role"]),
            }
        )

    columns = [
        "well",
        "role",
        "cause",
        "fold",
        "matched_cause_well",
        "matched_cause_role",
        "prefix_rows",
        "suffix_rows",
        "suffix_raw_gr_missing_fraction",
        "suffix_row_count_quartile",
        "raw_gr_missing_quartile",
        "prefix_row_count_quartile",
        "quartile_match_distance",
        "selection_hash",
    ]
    manifest = pd.DataFrame(output).loc[:, columns]
    manifest = manifest.sort_values(
        ["fold", "role", "selection_hash"], kind="mergesort"
    ).reset_index(drop=True)
    expected_roles = {"backward_cause": 8, "forward_cause": 8, "control": 16}
    if manifest["role"].value_counts().to_dict() != expected_roles:
        raise ValueError("fixed32 role counts changed")
    if len(manifest) != 32 or manifest["well"].nunique() != 32:
        raise ValueError("fixed32 wells are not unique")
    for role in CAUSES:
        if set(manifest.loc[manifest["role"].eq(role), "fold"]) != set(range(5)):
            raise ValueError(f"{role} must cover all five folds")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--episode-summary",
        type=Path,
        default=Path(
            "experiments/exp408_hmm_message_rate_basin_audit/artifacts/kaggle_v3/"
            "exp408_hmm_message_rate_basin_audit_episode_summary.csv"
        ),
    )
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
            "experiments/exp412_beta_filter_rate_disagreement_two_pass_reset/assets"
        ),
    )
    args = parser.parse_args()

    persistent_sha = sha256_file(args.persistent_wells)
    if persistent_sha != EXPECTED_PERSISTENT_SCOPE_SHA256:
        raise ValueError(f"exp408 persistent scope SHA changed: {persistent_sha}")
    persistent = set(
        pd.read_csv(args.persistent_wells, dtype={"well": str})["well"].astype(str)
    )
    if len(persistent) != 450:
        raise ValueError(f"expected 450 persistent wells, got {len(persistent)}")
    folds = load_fold_map(args.fold_source)
    raw = raw_target_free_metadata(args.train_dir, folds["well"].astype(str).tolist())
    metadata = folds.merge(raw, on="well", how="inner", validate="one_to_one")
    causes = load_cause_inventory(args.episode_summary)
    manifest = select_manifest(metadata, causes, persistent)

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
            "episode_summary_path": str(args.episode_summary),
            "episode_summary_sha256": sha256_file(args.episode_summary),
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
            "roles": manifest["role"].value_counts().sort_index().to_dict(),
            "fold_counts": manifest.groupby("fold").size().sort_index().to_dict(),
            "cause_fold_counts": {
                role: (
                    manifest.loc[manifest["role"].eq(role)]
                    .groupby("fold")
                    .size()
                    .sort_index()
                    .to_dict()
                )
                for role in CAUSES
            },
            "control_pool_wells": 323,
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
            "truth_columns_read_for_matching": [],
            "error_columns_read_for_matching": [],
            "cause_membership_used_for_sample_selection_only": True,
            "cause_membership_passed_to_hmm": False,
            "control_matching_columns": [
                "fold",
                "suffix_row_count_quartile",
                "raw_gr_missing_quartile",
                "prefix_row_count_quartile",
            ],
        },
    }
    metadata_path = args.output_dir / "stage0_fixed32_manifest_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
