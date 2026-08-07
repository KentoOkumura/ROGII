from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP264 = ROOT / "experiments" / "exp264_exp263_candidate_confidence_dual_selector"
EXP218 = ROOT / "experiments" / "exp218_gr_wavelet_rotation_confidence_features_on_exp148"
EXP263 = ROOT / "experiments" / "exp263_last_anchor_better_candidate_confidence_pair_cache"
OUTPUT_DIR = EXP264 / "artifacts" / "feature_availability_audit"

FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SELECTOR_FORBIDDEN = {
    feature
    for formation in FORMATION_COLUMNS
    for feature in (
        f"ctx__raw__{formation.lower()}",
        f"ctx__raw_delta_last__{formation.lower()}",
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def horizontal_schema(split_dir: Path) -> dict[str, Any]:
    paths = sorted(split_dir.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No horizontal wells under {split_dir}")
    by_file = {path.name: read_csv_header(path) for path in paths}
    common = set(by_file[paths[0].name])
    union: set[str] = set()
    for columns in by_file.values():
        common.intersection_update(columns)
        union.update(columns)
    return {
        "files": len(paths),
        "common_columns": sorted(common),
        "union_columns": sorted(union),
        "schema_variants": sorted({tuple(columns) for columns in by_file.values()}),
    }


def typewell_schema(split_dir: Path) -> dict[str, Any]:
    paths = sorted(split_dir.glob("*__typewell.csv"))
    if not paths:
        raise FileNotFoundError(f"No typewells under {split_dir}")
    variants = sorted({tuple(read_csv_header(path)) for path in paths})
    common = set(variants[0])
    union: set[str] = set()
    for columns in variants:
        common.intersection_update(columns)
        union.update(columns)
    return {
        "files": len(paths),
        "common_columns": sorted(common),
        "union_columns": sorted(union),
        "schema_variants": variants,
    }


def selector_audit(
    train_horizontal: dict[str, Any],
    test_horizontal: dict[str, Any],
    test_typewell: dict[str, Any],
) -> pd.DataFrame:
    catalog_path = EXP264 / "kaggle/output/stage_a_v1/artifacts/feature_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    selected = catalog.loc[catalog["selected"].astype(str).str.lower().eq("true")].copy()
    if len(selected) != 100:
        raise ValueError(f"Expected historical selector schema with 100 features, got {len(selected)}")

    test_horizontal_columns = set(test_horizontal["common_columns"])
    test_typewell_columns = set(test_typewell["common_columns"])
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        feature = str(row.feature)
        current_test_generated = True
        fold_safe = True
        evidence = "exp263 Stage 1 v3 candidate/confidence parity or target-free derived feature"
        action = "retain_for_rebuilt_stage_a"

        if feature in SELECTOR_FORBIDDEN:
            source = feature.rsplit("__", 1)[-1].upper()
            current_test_generated = source in test_horizontal_columns
            fold_safe = False
            evidence = (
                f"{source} is present in train horizontal schema but absent from actual current-test "
                "horizontal schema"
            )
            action = "drop"
        elif feature.startswith("ctx__raw__"):
            source = feature.rsplit("__", 1)[-1].upper()
            current_test_generated = source in test_horizontal_columns
            fold_safe = current_test_generated
            evidence = f"actual current-test horizontal column {source}"
        elif feature.startswith("ctx__raw_delta_last__"):
            source = feature.rsplit("__", 1)[-1].upper()
            current_test_generated = source in test_horizontal_columns and "TVT_input" in test_horizontal_columns
            fold_safe = current_test_generated
            evidence = f"actual current-test horizontal {source} plus known-prefix TVT_input"
        elif feature.startswith("ctx__typewell__"):
            current_test_generated = {"TVT", "GR"}.issubset(test_typewell_columns)
            fold_safe = current_test_generated
            evidence = "actual current-test typewell TVT/GR"

        rows.append(
            {
                "feature": feature,
                "group": str(row.group),
                "historical_selected": True,
                "train_source_available": True,
                "current_test_generated": bool(current_test_generated),
                "fold_safe": bool(fold_safe),
                "hidden_safe": bool(current_test_generated and fold_safe),
                "status": "pass" if current_test_generated and fold_safe else "fail",
                "evidence": evidence,
                "action": action,
            }
        )

    result = pd.DataFrame(rows)
    failed = set(result.loc[~result["hidden_safe"], "feature"])
    if failed != SELECTOR_FORBIDDEN:
        raise ValueError(f"Unexpected selector availability failures: {sorted(failed)}")
    return result


def exp218_invalid_base_features() -> set[str]:
    invalid = {"sig_std", "sig_mean_d"}
    for formation in FORMATION_COLUMNS:
        invalid.update(
            {
                f"tvtF_{formation}",
                f"tvtFw_{formation}",
                f"tvtF50_{formation}",
                f"bw_{formation}",
                f"bww_{formation}",
                f"bw50_{formation}",
                f"bw_early_{formation}",
                f"bw_mid_{formation}",
                f"frm_rmse_{formation}",
            }
        )
    invalid.update(
        {
            "form_mean_d",
            "form_std_d",
            "form_rng_d",
            "spatial_ancc_d",
            "spatial_knn_dist",
            "dense_ancc",
            "dense_std",
            "dense_dist",
            "tvt_dense_d",
            "tvt_densew_d",
            "tvt_dense50_d",
            "dense_rmse",
            "dense_bias",
            "dense_nb_std",
            "pf_vs_spatial",
            "pf_vs_dense",
            "spatial_vs_dense",
            "beam_vs_spatial",
        }
    )
    return invalid


def exp218_audit() -> pd.DataFrame:
    manifest_path = EXP264 / "kaggle/output/stage_d_v2/artifacts/stage_d_model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    features = [str(item) for item in manifest["feature_surfaces"]["matched_control"]]
    base_features = set(manifest["exp218_input"]["base_cache"]["feature_columns"])
    if len(features) != 380 or len(base_features) != 196:
        raise ValueError("Unexpected exp218 feature counts")

    invalid_base = exp218_invalid_base_features()
    if not invalid_base.issubset(base_features):
        raise ValueError(f"Unknown invalid base features: {sorted(invalid_base - base_features)}")
    invalid_learned = {feature for feature in features if feature.startswith("ll_learned_")}
    invalid_grwr_formation = {
        "grwr_candidate_tvt_std",
        "grwr_candidate_tvt_range",
        "grwr_dwt_energy_ratio_w065_x_candidate_std",
        "grwr_fft_rotation_ratio_x_candidate_range",
        "grwr_dwt_minus_raw_ncc_gap_x_candidate_range",
    }
    invalid_grwr_learned = {"grwr_ll_entropy_x_dwt_energy_ratio_w065"}
    invalid = invalid_base | invalid_learned | invalid_grwr_formation | invalid_grwr_learned

    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature in base_features:
            family = "base_replay"
        elif feature.startswith("uproj_"):
            family = "u_projection"
        elif feature.startswith("ll_"):
            family = "learned_likelihood"
        elif feature.startswith("grwr_"):
            family = "gr_wavelet_rotation"
        else:
            raise ValueError(f"Unclassified exp218 feature: {feature}")

        status = "pass"
        dependency = "target_free_same_source_generator"
        evidence = "exp218 inference v1 generated all 380 current-test columns finite"
        action = "retain_in_clean_273_surface"
        if feature in invalid_base:
            status = "fail"
            dependency = "full_train_formation_reference"
            evidence = (
                "FormationPlaneKNN/DenseANCCImputer fit all 773 training wells before downstream CV; "
                "outer-valid wells can use training-only formation values from peer outer-valid wells"
            )
            action = "drop_or_rebuild_inside_each_outer_fold"
        elif feature in invalid_learned:
            status = "fail"
            dependency = "exp111_fold0_target_trained_score"
            evidence = (
                "exp145 applies the two exp111 fold0 target-trained models to all 773 train wells; "
                "downstream folds 1-4 therefore receive non-OOF score features"
            )
            action = "drop_or_generate_nested_scores"
        elif feature in invalid_grwr_formation:
            status = "fail"
            dependency = "transitive_dense_formation_candidate"
            evidence = "GRWR candidate spread includes tvt_dense/tvt_densew/tvt_dense50"
            action = "drop_or_recompute_without_formation_candidates"
        elif feature in invalid_grwr_learned:
            status = "fail"
            dependency = "transitive_exp111_fold0_score"
            evidence = "GRWR interaction directly uses ll learned probability entropy"
            action = "drop_or_generate_nested_score_interaction"

        rows.append(
            {
                "feature": feature,
                "family": family,
                "current_test_generated": True,
                "fold_safe": status == "pass",
                "hidden_safe": status == "pass",
                "status": status,
                "dependency": dependency,
                "evidence": evidence,
                "action": action,
            }
        )

    result = pd.DataFrame(rows)
    failed = set(result.loc[result["status"].eq("fail"), "feature"])
    if failed != invalid:
        raise ValueError("exp218 invalid feature classifier is inconsistent")
    if len(failed) != 107 or int(result["status"].eq("pass").sum()) != 273:
        raise ValueError("Expected exp218 107 invalid / 273 retained features")
    return result


def write_readout(
    selector: pd.DataFrame,
    exp218: pd.DataFrame,
    schemas: dict[str, Any],
) -> None:
    selector_failed = selector.loc[selector["status"].eq("fail")]
    exp218_failed = exp218.loc[exp218["status"].eq("fail")]
    exp218_family = (
        exp218.groupby(["family", "status"], as_index=False)
        .size()
        .pivot(index="family", columns="status", values="size")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for column in ("pass", "fail"):
        if column not in exp218_family:
            exp218_family[column] = 0

    summary = {
        "schema_version": "1.0.0",
        "status": "completed_blocking_leakage_found",
        "selector": {
            "historical_features": int(len(selector)),
            "hidden_safe_features": int(selector["hidden_safe"].sum()),
            "invalid_features": int((~selector["hidden_safe"]).sum()),
            "invalid_feature_names": selector_failed["feature"].tolist(),
            "recommended_action": "drop_12_and_rebuild_stage_a",
        },
        "exp218": {
            "historical_features": int(len(exp218)),
            "hidden_safe_features": int(exp218["hidden_safe"].sum()),
            "invalid_features": int((~exp218["hidden_safe"]).sum()),
            "invalid_by_dependency": exp218_failed.groupby("dependency").size().to_dict(),
            "family_counts": exp218_family.to_dict(orient="records"),
            "recommended_minimal_surface": "clean_273_drop_only",
            "matched_control_reusable_without_retraining": False,
        },
        "actual_raw_schema": schemas,
        "evidence": {
            "historical_selector_catalog_sha256": sha256_file(
                EXP264 / "kaggle/output/stage_a_v1/artifacts/feature_catalog.csv"
            ),
            "stage_d_manifest_sha256": sha256_file(
                EXP264 / "kaggle/output/stage_d_v2/artifacts/stage_d_model_manifest.json"
            ),
            "exp218_metrics_sha256": sha256_file(EXP218 / "metrics.json"),
            "exp263_metrics_sha256": sha256_file(EXP263 / "metrics.json"),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selector.to_csv(OUTPUT_DIR / "selector_feature_availability.csv", index=False)
    exp218.to_csv(OUTPUT_DIR / "exp218_feature_availability.csv", index=False)
    exp218.loc[exp218["hidden_safe"], ["feature", "family"]].to_csv(
        OUTPUT_DIR / "exp218_clean_273_allowlist.csv", index=False
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    family_lines = "\n".join(
        f"| `{row.family}` | {int(row['pass'])} | {int(row['fail'])} |"
        for _, row in exp218_family.sort_values("family").iterrows()
    )
    readout = f"""# exp264 feature availability audit

## 結論

- 旧selector 100特徴: hidden-safe 88、無効12。formation raw/delta 12特徴を削除してStage Aから再生成する。
- exp218 matched-control 380特徴: hidden-safe 273、無効107。既存380列controlとそのOOFを再利用しない。
- exp218の380列はcurrent-testでfinite生成できたが、schema/finite parityだけではfold-safeを意味しない。

## selector

actual train horizontal共通列は`{', '.join(schemas['train_horizontal']['common_columns'])}`、actual current-testは
`{', '.join(schemas['test_horizontal']['common_columns'])}`。`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`の
raw値とlast-known差分、計12特徴がcurrent-testで生成不能だった。残る88特徴はactual raw/typewell schema、
exp263 Stage 1 v3 candidate/value confidence parity、target-free決定変換のいずれかへ追跡できた。

## exp218 380

| family | pass | fail |
| --- | ---: | ---: |
{family_lines}

無効107列の内訳:

- 74列: full-trainでfitしたFormationPlaneKNN/DenseANCCImputerへ依存。各well自身は除外しても、
  outer-valid well同士のtraining-only formation値を参照できるためfold-safeではない。
- 27列: exp111 fold0のtarget-trained scoreをexp145が全773 train wellsへ適用。downstream fold 1-4では
  outer-valid wellがexp111学習側に含まれるためnested stackingになっていない。
- 5列: GRWR candidate spreadがformation依存`tvt_dense*`を推移的に使用。
- 1列: GRWR interactionが非nestedなlearned probability entropyを使用。

最小修正は107列を落とした273列surfaceでcontrol/add-onlyを同一条件再学習すること。formation 74列を
outer-fold内で再生成し、learned score 27列をnested化する案は別設計・別コストとして扱う。
"""
    (OUTPUT_DIR / "README.md").write_text(readout)


def main() -> None:
    train_dir = ROOT / "data/raw/train"
    test_dir = ROOT / "data/raw/test"
    schemas = {
        "train_horizontal": horizontal_schema(train_dir),
        "test_horizontal": horizontal_schema(test_dir),
        "train_typewell": typewell_schema(train_dir),
        "test_typewell": typewell_schema(test_dir),
    }
    selector = selector_audit(
        schemas["train_horizontal"],
        schemas["test_horizontal"],
        schemas["test_typewell"],
    )
    exp218 = exp218_audit()
    write_readout(selector, exp218, schemas)
    print(
        json.dumps(
            {
                "selector": selector["status"].value_counts().to_dict(),
                "exp218": exp218["status"].value_counts().to_dict(),
                "output_dir": str(OUTPUT_DIR),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
