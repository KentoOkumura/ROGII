from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from copcf_rawtest_regeneration import attach_regenerated_copcf
from hmm_exp226_candidate_selector_on_exp183 import (
    EXP209_HMM_TRAIN_FEATURES,
    EXP223_TRAIN_FEATURES,
    OUTPUT_PREFIX,
    add_feature_enrichment,
    build_long_frame,
    candidate_specs_from_config,
    find_artifact,
    prediction_sha256,
    sha256_path,
    to_jsonable,
    variant_specs_from_config,
    viterbi_select,
)
from settings import ExperimentPaths, get_nested

EXP073_TEST_FEATURES = "exp063_full_replay_repro_guard_test_features.csv.gz"
MODEL_MANIFEST = f"{OUTPUT_PREFIX}_model_manifest.json"
EXP226_TEST_SUMMARY = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_"
    "test_prediction_summary.csv"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trusted_module(
    source: Path, *, module_name: str, work_dir: Path
) -> tuple[ModuleType, dict[str, str]]:
    """Copy a pinned Kaggle input source before importing its named helper module."""
    copied = work_dir / f"{module_name}.py"
    shutil.copy2(source, copied)
    spec = importlib.util.spec_from_file_location(module_name, copied)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trusted source module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"source": str(source), "sha256": _sha256_file(source), "copied": str(copied)}


def _module_from_source_artifact(
    artifact_filename: str,
    *,
    module_name: str,
    work_dir: Path,
) -> tuple[ModuleType, dict[str, str]]:
    artifact = find_artifact(artifact_filename)
    source = artifact.parent.parent / "exact_hmm_smoother.py"
    if not source.exists():
        raise FileNotFoundError(f"expected HMM source beside {artifact}: {source}")
    return _load_trusted_module(source, module_name=module_name, work_dir=work_dir)


def _required_base_columns(config: dict[str, Any]) -> list[str]:
    enrichment = get_nested(config, "ranker.feature_enrichment") or {}
    columns = {
        "id",
        "well",
        "last_known_tvt",
        "pf_ancc",
        "pf_ancc_std",
        "beam_mean_d",
        "sc_ens_d",
        "hyb_d",
        "likpf_mean_d",
        "eval_len",
        "md_since",
    }
    columns.update(str(value) for value in enrichment.get("auxiliary_columns", []))
    return sorted(columns)


def _load_base_test_frame(
    config: dict[str, Any], paths: ExperimentPaths
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP073_TEST_FEATURES,
        get_nested(config, "data.exp073_test_feature_cache_local"),
    )
    required = _required_base_columns(config)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"exp073 raw-test cache lacks required columns: {missing}")
    frame = pd.read_csv(source, usecols=required, dtype={"id": str, "well": str}, low_memory=False)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    if frame.duplicated("id").any() or frame["id"].isna().any() or frame["well"].isna().any():
        raise ValueError("exp073 raw-test cache has duplicate or missing id/well")

    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    expected_ids = sample["id"].astype(str)
    if len(frame) != len(sample) or set(frame["id"]) != set(expected_ids):
        raise ValueError("exp073 raw-test cache ID set does not match sample_submission")
    frame = frame.set_index("id").loc[expected_ids].reset_index()
    return frame, {
        "path": str(source),
        "sha256": sha256_path(source),
        "decompressed_sha256": sha256_path(source, decompressed=True),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "sample_submission_id_contract": "pass",
    }


def _base_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    last = out["last_known_tvt"].to_numpy(np.float32)
    for name, delta in {
        "beam_mean": "beam_mean_d",
        "likpf_mean": "likpf_mean_d",
        "sc_ens": "sc_ens_d",
        "hyb": "hyb_d",
        "tvt_dense": "tvt_dense_d",
        "tvt_densew": "tvt_densew_d",
        "tvt_dense50": "tvt_dense50_d",
    }.items():
        out[name] = (last + out[delta].to_numpy(np.float32)).astype(np.float32)
    if not np.isfinite(
        out[["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]].to_numpy()
    ).all():
        raise ValueError("base raw-test candidate values contain non-finite values")
    return out


def _hmm_test_rows(
    module: ModuleType,
    *,
    test_dir: Path,
    hmm_config: dict[str, Any],
    self_gr: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in module.list_well_ids(test_dir):
        horizontal, typewell = module.load_well(well, test_dir)
        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
        if not known.any():
            raise ValueError(f"raw test well {well} has no finite TVT_input prefix")
        eval_index = np.flatnonzero(~known).astype(np.int64)
        if len(eval_index) == 0:
            continue
        last = horizontal.loc[known].iloc[-1]
        last_tvt = float(last["TVT_input"])
        last_md = float(last["MD"])
        kwargs: dict[str, Any] = dict(hmm_config)
        if self_gr is not None:
            kwargs.update(
                {
                    "self_gr_config": self_gr["surface"],
                    "self_gr_alpha": float(self_gr["alpha"]),
                    "self_gr_clip": float(self_gr["clip"]),
                    "self_gr_mode": str(self_gr["mode"]),
                }
            )
        result = module.run_hmm2(horizontal, typewell, **kwargs)
        actual = np.asarray(result["ev_index"], dtype=np.int64)
        if not np.array_equal(actual, eval_index):
            raise ValueError(f"HMM eval row contract differs for test well {well}")
        item = pd.DataFrame(
            {
                "id": [f"{well}_{int(idx)}" for idx in actual],
                "well": str(well),
                "last_known_tvt_hmm": np.float32(last_tvt),
                "md_since_hmm": (
                    pd.to_numeric(horizontal.loc[actual, "MD"], errors="coerce").to_numpy(
                        np.float32
                    )
                    - np.float32(last_md)
                ),
                "hmm_mean_tvt": np.asarray(result["mean_eval"], dtype=np.float32),
                "hmm_std": np.asarray(result["std_eval"], dtype=np.float32),
                "hmm_loglik": np.float32(result["loglik"]),
            }
        )
        if self_gr is not None:
            item["self_gr_quality"] = np.asarray(result["self_gr_quality"], dtype=np.float32)
            item["self_gr_peak_gap"] = np.asarray(result["self_gr_peak_gap"], dtype=np.float32)
            item["self_gr_typewell_agreement"] = np.asarray(
                result["self_gr_typewell_agreement"], dtype=np.float32
            )
            item["self_gr_valid"] = np.asarray(result["self_gr_valid"], dtype=np.float32)
        rows.append(item)
    out = pd.concat(rows, ignore_index=True)
    if out.duplicated(["id", "well"]).any():
        raise ValueError("HMM raw-test rows contain duplicate id/well")
    return out


def _merge_required(base: pd.DataFrame, extra: pd.DataFrame, *, name: str) -> pd.DataFrame:
    merged = base.merge(extra, on=["id", "well"], how="left", validate="one_to_one")
    if len(merged) != len(base):
        raise ValueError(f"{name} changed row count")
    required = [column for column in extra.columns if column not in {"id", "well"}]
    if merged[required].isna().any().any():
        examples = merged.loc[merged[required].isna().any(axis=1), ["id", "well"]].head(5)
        raise ValueError(f"{name} does not cover base raw-test rows: {examples.to_dict('records')}")
    return merged


def _attach_hmm_candidates(
    frame: pd.DataFrame, config: dict[str, Any], paths: ExperimentPaths
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work_dir = paths.artifacts_dir / "trusted_upstream_modules"
    work_dir.mkdir(parents=True, exist_ok=True)
    exact_module, exact_meta = _module_from_source_artifact(
        EXP209_HMM_TRAIN_FEATURES,
        module_name="exp237_exp209_exact_hmm",
        work_dir=work_dir,
    )
    self_module, self_meta = _module_from_source_artifact(
        EXP223_TRAIN_FEATURES,
        module_name="exp237_exp223_selfgr_hmm",
        work_dir=work_dir,
    )
    hmm = get_nested(config, "inference.hmm") or {}
    exact = _hmm_test_rows(
        exact_module,
        test_dir=paths.test_data_dir,
        hmm_config=dict(hmm.get("exact", {})),
    ).rename(
        columns={
            "last_known_tvt_hmm": "last_known_tvt_exact_hmm",
            "md_since_hmm": "md_since_exact_hmm",
            "hmm_mean_tvt": "hmm_exact_mean_tvt",
            "hmm_std": "hmm_exact_std",
            "hmm_loglik": "hmm_exact_loglik",
        }
    )
    out = _merge_required(frame, exact, name="exp209 exact HMM raw-test")
    if not np.allclose(out["last_known_tvt"], out["last_known_tvt_exact_hmm"], atol=1e-3, rtol=0.0):
        raise ValueError("exp209 raw-test last_known_tvt contract differs from exp073 base cache")
    if not np.allclose(out["md_since"], out["md_since_exact_hmm"], atol=1e-3, rtol=0.0):
        raise ValueError("exp209 raw-test md_since contract differs from exp073 base cache")
    out = out.drop(columns=["last_known_tvt_exact_hmm", "md_since_exact_hmm"])
    out["blend_likpf_hmm_w500"] = (
        0.5 * out["likpf_mean"].to_numpy(np.float32)
        + 0.5 * out["hmm_exact_mean_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    out["hmm_exact_minus_likpf_mean"] = (
        out["hmm_exact_mean_tvt"].to_numpy(np.float32) - out["likpf_mean"].to_numpy(np.float32)
    ).astype(np.float32)

    self_raw = dict(hmm.get("self_gr") or {})
    self_rows = _hmm_test_rows(
        self_module,
        test_dir=paths.test_data_dir,
        hmm_config=dict(self_raw.get("exact_hmm", hmm.get("exact", {}))),
        self_gr={
            "alpha": self_raw["alpha"],
            "clip": self_raw["clip"],
            "mode": self_raw["mode"],
            "surface": dict(self_raw["surface"]),
        },
    ).rename(
        columns={
            "last_known_tvt_hmm": "last_known_tvt_self_hmm",
            "md_since_hmm": "md_since_self_hmm",
            "hmm_mean_tvt": "hmm_selfgr_boost_only_a070_c100_mean_tvt",
            "hmm_std": "hmm_selfgr_std",
            "hmm_loglik": "hmm_selfgr_loglik",
        }
    )
    out = _merge_required(out, self_rows, name="exp223 self-GR HMM raw-test")
    if not np.allclose(out["last_known_tvt"], out["last_known_tvt_self_hmm"], atol=1e-3, rtol=0.0):
        raise ValueError("exp223 raw-test last_known_tvt contract differs from exp073 base cache")
    out = out.drop(columns=["last_known_tvt_self_hmm", "md_since_self_hmm"])
    return out, {"exp209_module": exact_meta, "exp223_module": self_meta}


def _attach_exp226_candidate(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    marker = find_artifact(
        EXP226_TEST_SUMMARY, get_nested(config, "data.exp226_test_summary_local")
    )
    submission = marker.parent.parent / "submission.csv"
    if not submission.exists():
        raise FileNotFoundError(
            f"exp226 raw-test submission not found beside {marker}: {submission}"
        )
    values = pd.read_csv(submission, dtype={"id": str})
    if set(values.columns) != {"id", "tvt"}:
        raise ValueError(f"unexpected exp226 submission columns: {values.columns.tolist()}")
    values["id"] = values["id"].astype(str)
    values["well"] = values["id"].str.rsplit("_", n=1).str[0]
    values = values.rename(columns={"tvt": "exp226_v6_k16_geometry_gr_u_projection"})
    values["exp226_v6_k16_geometry_gr_u_projection"] = pd.to_numeric(
        values["exp226_v6_k16_geometry_gr_u_projection"], errors="coerce"
    ).astype(np.float32)
    out = _merge_required(
        frame,
        values[["id", "well", "exp226_v6_k16_geometry_gr_u_projection"]],
        name="exp226 K16 raw-test",
    )
    return out, {
        "summary": str(marker),
        "summary_sha256": sha256_path(marker),
        "submission": str(submission),
        "submission_sha256": sha256_path(submission),
    }


def _attach_multiobs(
    frame: pd.DataFrame, config: dict[str, Any], paths: ExperimentPaths
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scorer = get_nested(config, "inference.multi_observation_likelihood") or {}
    offsets = np.asarray(
        [int(value) for value in scorer.get("observation_offsets", [-24, -12, 0, 12, 24])],
        dtype=np.int32,
    )
    window = int(scorer.get("gr_rolling_window", 5))
    gr_scale = float(scorer.get("gr_scale", 18.0))
    range_scale = float(scorer.get("out_of_range_scale", 80.0))
    names = ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]
    rows: list[pd.DataFrame] = []
    by_well: list[dict[str, Any]] = []
    for well, positions in frame.groupby("well", sort=False).groups.items():
        position_list = list(positions)
        horizontal = pd.read_csv(
            paths.test_data_dir / f"{well}__horizontal_well.csv",
            usecols=["GR", "TVT_input"],
        )
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known = tvt_input.notna().to_numpy()
        if not known.any():
            raise ValueError(f"no finite TVT_input prefix for multi-observation well {well}")
        prefix_len = int(np.flatnonzero(known)[-1] + 1)
        prefix_tvt = (
            tvt_input.iloc[:prefix_len]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        gr = pd.to_numeric(horizontal["GR"], errors="coerce")
        fallback = float(gr.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            fallback = float(gr.mean()) if np.isfinite(float(gr.mean())) else 0.0
        full_gr = (
            gr.interpolate(limit_direction="both")
            .fillna(fallback)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )
        local = frame.loc[position_list]
        row_idx = local["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy(np.int32)
        values = local[names].to_numpy(np.float32)
        order = np.argsort(prefix_tvt)
        sorted_tvt = prefix_tvt[order]
        flat = values.reshape(-1)
        insertion = np.searchsorted(sorted_tvt, flat, side="left")
        left = np.clip(insertion - 1, 0, len(sorted_tvt) - 1)
        right = np.clip(insertion, 0, len(sorted_tvt) - 1)
        choose_right = np.abs(sorted_tvt[right] - flat) < np.abs(sorted_tvt[left] - flat)
        nearest = order[np.where(choose_right, right, left)].reshape(values.shape)
        eval_vectors = []
        candidate_vectors = []
        for offset in offsets:
            eval_vectors.append(full_gr[np.clip(row_idx + offset, 0, len(full_gr) - 1)])
            candidate_vectors.append(full_gr[np.clip(nearest + offset, 0, len(full_gr) - 1)])
        evaluation = np.stack(eval_vectors, axis=1).astype(np.float32)
        candidate_tensor = np.stack(candidate_vectors, axis=2).astype(np.float32)
        mae = np.mean(np.abs(candidate_tensor - evaluation[:, None, :]), axis=2)
        evaluation_z = (evaluation - evaluation.mean(axis=1, keepdims=True)) / (
            evaluation.std(axis=1, keepdims=True) + 1e-6
        )
        flat_tensor = candidate_tensor.reshape(len(values) * len(names), len(offsets))
        candidate_z = (flat_tensor - flat_tensor.mean(axis=1, keepdims=True)) / (
            flat_tensor.std(axis=1, keepdims=True) + 1e-6
        )
        candidate_z = candidate_z.reshape(len(values), len(names), len(offsets))
        ncc = np.mean(candidate_z * evaluation_z[:, None, :], axis=2)
        below = np.maximum(0.0, float(np.nanmin(prefix_tvt)) - values)
        above = np.maximum(0.0, values - float(np.nanmax(prefix_tvt)))
        score = np.clip(
            np.exp(-(mae / max(gr_scale, 1e-6)))
            * (0.25 + 0.75 * np.clip((ncc + 1.0) / 2.0, 0.0, 1.0))
            * np.exp(-((below + above) / max(range_scale, 1e-6))),
            0.0,
            1.0,
        ).astype(np.float32)
        best = score.argmax(axis=1)
        item = pd.DataFrame(
            {
                "id": local["id"].to_numpy(),
                "well": str(well),
                "multiobs_top1": values[np.arange(len(values)), best],
                "multiobs_score_max": score.max(axis=1),
                "multiobs_score_mean": score.mean(axis=1),
                "multiobs_score_gap": np.sort(score, axis=1)[:, -1] - np.sort(score, axis=1)[:, -2],
                "multiobs_top1_source_id": best.astype(np.float32),
                "multiobs_top1_mae": mae[np.arange(len(values)), best],
                "multiobs_top1_ncc": ncc[np.arange(len(values)), best],
            }
        )
        for index, name in enumerate(names):
            item[f"multiobs_score_{name}"] = score[:, index]
            item[f"multiobs_mae_{name}"] = mae[:, index]
            item[f"multiobs_ncc_{name}"] = ncc[:, index]
        rows.append(item)
        by_well.append(
            {
                "well": str(well),
                "rows": int(len(item)),
                "known_prefix_rows": prefix_len,
                "score_mean": float(score.mean()),
            }
        )
    multiobs = pd.concat(rows, ignore_index=True)
    return _merge_required(frame, multiobs, name="exp099 multi-observation raw-test"), {
        "rows": int(len(multiobs)),
        "wells": int(multiobs["well"].nunique()),
        "by_well": by_well,
    }


def _load_saved_models(*, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import Booster

    manifest_path = find_artifact(
        MODEL_MANIFEST, get_nested(config, "data.exp237_model_manifest_local")
    )
    manifest = json.loads(manifest_path.read_text()).get("models", [])
    if len(manifest) != int(get_nested(config, "validation.n_folds") or 5):
        raise ValueError(f"unexpected model count in {manifest_path}: {len(manifest)}")
    models: list[dict[str, Any]] = []
    for item in sorted(manifest, key=lambda value: int(value["fold"])):
        fold = int(item["fold"])
        model_path = manifest_path.parent / str(item["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"saved exp237 model missing: {model_path}")
        booster = Booster(model_file=str(model_path))
        expected = list(booster.feature_name())
        models.append(
            {
                "fold": fold,
                "booster": booster,
                "feature_names": expected,
                "path": str(model_path),
                "sha256": sha256_path(model_path),
            }
        )
    return models, {"manifest": str(manifest_path), "manifest_sha256": sha256_path(manifest_path)}


def _test_candidate_features(
    frame: pd.DataFrame, candidates: list[Any]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = frame.copy()
    values = np.column_stack(
        [
            pd.to_numeric(out[item.column], errors="coerce").to_numpy(np.float32)
            for item in candidates
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("raw-test candidate matrix contains non-finite values")
    last = out["last_known_tvt"].to_numpy(np.float32)
    for index, item in enumerate(candidates):
        out[f"{item.name}_minus_last"] = values[:, index] - last
    for left, first in enumerate(candidates):
        for right in candidates[left + 1 :]:
            out[f"{first.name}_vs_{right.name}_abs"] = np.abs(
                out[first.column].to_numpy(np.float32) - out[right.column].to_numpy(np.float32)
            )
    out["candidate_mean"] = values.mean(axis=1).astype(np.float32)
    out["candidate_std"] = values.std(axis=1).astype(np.float32)
    out["candidate_range"] = (values.max(axis=1) - values.min(axis=1)).astype(np.float32)
    out["target"] = np.float32(0.0)
    out["true_tvt"] = last
    return out, values, np.zeros(len(out), dtype=np.int16)


def _predict_scores(
    *,
    frame: pd.DataFrame,
    candidates: list[Any],
    candidate_values: np.ndarray,
    dummy_labels: np.ndarray,
    models: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    row_features = [
        column
        for column in frame.columns
        if column not in {"id", "well", "target", "true_tvt", "oracle_label", "oracle_candidate"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    long_frame, _ = build_long_frame(
        frame,
        np.arange(len(frame), dtype=np.int64),
        candidates,
        row_feature_columns=row_features,
        candidate_values=candidate_values,
        oracle_labels=dummy_labels,
        sample_rows=None,
        seed=0,
        config=config,
    )
    scores = np.zeros((len(frame), len(candidates)), dtype=np.float32)
    unavailable: set[str] = set()
    for item in models:
        expected = item["feature_names"]
        for column in expected:
            if column not in long_frame.columns:
                long_frame[column] = np.nan
                unavailable.add(column)
        values = long_frame[expected].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        medians = np.nanmedian(values, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(medians, np.where(missing)[1])
        prediction = item["booster"].predict(values).reshape(len(candidates), len(frame)).T
        scores += np.asarray(prediction, dtype=np.float32) / np.float32(len(models))
    return scores, {
        "available_row_feature_count": int(len(row_features)),
        "imputation_policy": (
            "rawtest_long_feature_median_per_model_schema; all_missing_columns_use_zero"
        ),
        "unavailable_long_feature_count": int(len(unavailable)),
        "unavailable_long_features": sorted(unavailable),
    }


def run_rawtest_inference(*, config: dict[str, Any], paths: ExperimentPaths) -> dict[str, Any]:
    started = time.time()
    output_dir = paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base, base_meta = _load_base_test_frame(config, paths)
    test_frame = _base_candidates(base)
    test_frame, hmm_meta = _attach_hmm_candidates(test_frame, config, paths)
    test_frame, exp226_meta = _attach_exp226_candidate(test_frame, config)
    test_frame, _, copcf_meta = attach_regenerated_copcf(test_frame, config, paths)
    test_frame, multiobs_meta = _attach_multiobs(test_frame, config, paths)
    test_frame, _, enrichment_meta = add_feature_enrichment(test_frame, config, max_rows=None)
    candidates = candidate_specs_from_config(config)
    test_frame, candidate_values, dummy_labels = _test_candidate_features(test_frame, candidates)

    models, model_meta = _load_saved_models(config=config)

    scores, coverage = _predict_scores(
        frame=test_frame,
        candidates=candidates,
        candidate_values=candidate_values,
        dummy_labels=dummy_labels,
        models=models,
        config=config,
    )
    candidate_names = [item.name for item in candidates]
    default_idx = candidate_names.index(str(get_nested(config, "selector.default_candidate")))
    allowed = get_nested(config, "selector.allowed_switch_candidates") or candidate_names
    allowed_idx = np.asarray([candidate_names.index(name) for name in allowed], dtype=np.int16)
    specs = variant_specs_from_config(config)
    if len(specs) != 1:
        raise ValueError(f"raw-test inference requires one fixed Viterbi rule, got {len(specs)}")
    selected = viterbi_select(
        frame=test_frame,
        predicted_error=scores,
        candidate_values=candidate_values,
        candidate_names=candidate_names,
        default_idx=default_idx,
        allowed_switch_idx=allowed_idx,
        spec=specs[0],
    )
    selected_tvt = candidate_values[np.arange(len(test_frame)), selected].astype(np.float32)
    prediction = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "well": test_frame["well"].astype(str),
            "selected_candidate": [candidate_names[index] for index in selected],
            "selected_candidate_index": selected.astype(np.int16),
            "selected_tvt": selected_tvt,
            "predicted_error": scores[np.arange(len(test_frame)), selected],
        }
    )
    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    submission = sample[["id"]].merge(
        prediction[["id", "selected_tvt"]], on="id", how="left", validate="one_to_one"
    )
    submission = submission.rename(columns={"selected_tvt": "tvt"})
    if (
        len(submission) != len(sample)
        or submission["tvt"].isna().any()
        or not np.isfinite(submission["tvt"]).all()
    ):
        raise ValueError("raw-test submission contract failed")
    prediction_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_selected_predictions.csv.gz"
    test_feature_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_candidate_features.csv.gz"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_inference_summary.json"
    prediction.to_csv(prediction_path, index=False, compression="gzip")
    test_frame.to_csv(test_feature_path, index=False, compression="gzip")
    submission.to_csv(paths.submission_path, index=False)
    distribution = prediction["selected_candidate"].value_counts(normalize=True).sort_index()
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "rawtest_inference_completed_not_submitted",
        "rows": int(len(prediction)),
        "wells": int(prediction["well"].nunique()),
        "selected_variant": specs[0].variant,
        "selection_distribution": {str(key): float(value) for key, value in distribution.items()},
        "base_test": base_meta,
        "hmm": hmm_meta,
        "exp226": exp226_meta,
        "copcf": copcf_meta,
        "multiobs": multiobs_meta,
        "enrichment": enrichment_meta,
        "models": model_meta,
        "feature_coverage": coverage,
        "artifacts": {
            "selected_predictions": prediction_path.name,
            "candidate_features": test_feature_path.name,
            "submission": paths.submission_path.name,
        },
        "sha256": {
            "selected_predictions_gzip": sha256_path(prediction_path),
            "candidate_features_gzip": sha256_path(test_feature_path),
            "submission": sha256_path(paths.submission_path),
            "prediction_content": prediction_sha256(
                prediction.rename(columns={"selected_tvt": "prediction"}), value_col="prediction"
            ),
        },
        "notes": [
            "No Kaggle competition submission is made by this notebook.",
            (
                "HMM candidates, exp099 multi-observation scores, and exp226 candidate are "
                "regenerated/read target-free for raw test."
            ),
            (
                "Cluster/typewell/spatial prior confidence features are regenerated from "
                "raw-test inputs and full-train reference curves after excluding every "
                "raw-test well ID from the labeled source pool."
            ),
            (
                "This is an explicitly user-authorized inference despite exp237 near and "
                "worst-well safety-guard failures."
            ),
        ],
        "runtime_seconds": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return summary
