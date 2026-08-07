from __future__ import annotations

import gc
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp239_distribution_matched_multicut_pseudotail"


def _sha256(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _find(filename: str) -> Path:
    candidates = [Path.cwd() / filename, Path.cwd() / "inputs" / filename]
    root = Path("/kaggle/input")
    if root.exists():
        candidates.extend(root.glob(f"**/{filename}"))
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path
    raise FileNotFoundError(filename)


def _request_well(request_id: str) -> str:
    return f"pt{request_id[:20]}"


def _write_masked_request_files(
    replay: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    raw_train_dir: Path,
    request_train_dir: Path,
) -> pd.DataFrame:
    request_train_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for request in replay.sort_values("request_id").itertuples(index=False):
        source = str(request.source_well)
        request_well = _request_well(str(request.request_id))
        horizontal = frames[source].copy()
        cutoff = int(request.cutoff_index)
        horizontal["TVT_input"] = np.nan
        horizontal.loc[:cutoff, "TVT_input"] = pd.to_numeric(
            horizontal.loc[:cutoff, "TVT"], errors="coerce"
        )
        horizontal.to_csv(request_train_dir / f"{request_well}__horizontal_well.csv", index=False)
        typewell = pd.read_csv(raw_train_dir / f"{source}__typewell.csv")
        typewell.to_csv(request_train_dir / f"{request_well}__typewell.csv", index=False)
        rows.append(
            {
                "request_id": str(request.request_id),
                "request_well": request_well,
                "source_well": source,
                "fold": int(request.fold),
                "cutoff_index": cutoff,
            }
        )
    return pd.DataFrame(rows)


def _install_source_well_exclusion(public_module: Any, mapping: dict[str, str]) -> None:
    for imputer in [public_module._FI, public_module._DI]:
        original = imputer.impute

        def translated(xy: np.ndarray, self_wid: str | None = None, *, _original=original):
            return _original(xy, self_wid=mapping.get(str(self_wid), self_wid))

        imputer.impute = translated


def _build_one_base_request(
    row: Any,
    request_train_dir: Path,
    sampled_rows: dict[str, set[int]],
    public_module: Any,
) -> pd.DataFrame:
    request_well = str(row.request_well)
    horizontal_path = request_train_dir / f"{request_well}__horizontal_well.csv"
    typewell_path = request_train_dir / f"{request_well}__typewell.csv"
    base = public_module.build_well(str(horizontal_path), str(typewell_path), True)
    if base is None:
        raise RuntimeError(f"exp072 base generation failed for {request_well}")
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path).sort_values("TVT")
    likelihood, indices, _meta = public_module.lik_pf(
        horizontal,
        typewell,
        seed_base=public_module.stable_seed("likpf", request_well),
    )
    likpf = {"id": [f"{request_well}_{int(index)}" for index in indices]}
    for key, values in likelihood.items():
        name = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
        likpf[name] = values.astype(np.float32)
    base = public_module.add_likpf_features(base, pd.DataFrame(likpf))
    row_index = pd.to_numeric(base["id"].str.extract(r"_(\d+)$", expand=False)).astype(int)
    keep = row_index.isin(sampled_rows[str(row.request_id)])
    base = base.loc[keep].copy()
    base["request_id"] = str(row.request_id)
    base["source_well"] = str(row.source_well)
    base["source_fold"] = int(row.fold)
    return base


def _build_pseudo_base(
    replay: pd.DataFrame,
    materialized: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    raw_train_dir: Path,
    work_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    import public_notebook_replay_audit as public

    request_train_dir = work_dir / "request_raw" / "train"
    request_map = _write_masked_request_files(replay, frames, raw_train_dir, request_train_dir)
    public.configure_public_runtime(
        data_dir=raw_train_dir.parent,
        output_dir=work_dir / "exp072",
        n_jobs=int(_nested(config, "model.exp218_augmentation.feature_generation.n_jobs", 8)),
        pf_seeds=int(_nested(config, "model.exp218_augmentation.feature_generation.pf_seeds", 128)),
        pf_particles=int(
            _nested(config, "model.exp218_augmentation.feature_generation.pf_particles", 500)
        ),
        fast=False,
        use_gpu="cpu",
        n_train_wells=None,
    )
    source_wells = sorted(frames)
    public.init_imputers(source_wells)
    _install_source_well_exclusion(
        public, request_map.set_index("request_well")["source_well"].to_dict()
    )
    sampled_rows = {
        str(request_id): set(group["row_index"].astype(int))
        for request_id, group in materialized.groupby("request_id", sort=False)
    }
    parts = Parallel(
        n_jobs=int(_nested(config, "model.exp218_augmentation.feature_generation.n_jobs", 8)),
        prefer="threads",
    )(
        delayed(_build_one_base_request)(row, request_train_dir, sampled_rows, public)
        for row in request_map.itertuples(index=False)
    )
    pseudo = pd.concat(parts, ignore_index=True)
    expected = int(len(materialized))
    if len(pseudo) != expected:
        raise AssertionError(f"pseudo base rows {len(pseudo)} != {expected}")
    return pseudo, request_train_dir, request_map


def _configure_pseudo_runtime(
    replay: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    raw_train_dir: Path,
    work_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, pd.DataFrame, Any]:
    import public_notebook_replay_audit as public

    request_train_dir = work_dir / "request_raw" / "train"
    request_map = (
        _write_masked_request_files(replay, frames, raw_train_dir, request_train_dir)
        .sort_values("request_id")
        .reset_index(drop=True)
    )
    public.configure_public_runtime(
        data_dir=raw_train_dir.parent,
        output_dir=work_dir / "exp072",
        n_jobs=int(_nested(config, "model.exp218_augmentation.feature_generation.n_jobs", 4)),
        pf_seeds=int(_nested(config, "model.exp218_augmentation.feature_generation.pf_seeds", 128)),
        pf_particles=int(
            _nested(config, "model.exp218_augmentation.feature_generation.pf_particles", 500)
        ),
        fast=False,
        use_gpu="cpu",
        n_train_wells=None,
    )
    public.init_imputers(sorted(frames))
    _install_source_well_exclusion(
        public, request_map.set_index("request_well")["source_well"].to_dict()
    )
    return request_train_dir, request_map, public


def _row_content_sha256(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype=np.uint64, copy=False
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_chunked_feature_cache_generation(
    *,
    replay: pd.DataFrame,
    materialized: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    raw_train_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import public_notebook_replay_audit as public

    started = time.time()
    expected_requests = int(_nested(config, "model.exp218_augmentation.pseudo_request_count", 800))
    expected_rows = int(_nested(config, "model.exp218_augmentation.expected_pseudo_rows", 799_961))
    expected_features = int(
        _nested(config, "model.exp218_augmentation.expected_feature_count", 380)
    )
    preflight = bool(
        _nested(
            config,
            "model.exp218_augmentation.feature_cache.preflight.enabled",
            False,
        )
    )
    if preflight:
        preflight_requests = int(
            _nested(
                config,
                "model.exp218_augmentation.feature_cache.preflight.request_count",
                25,
            )
        )
        selected_request_ids = replay.sort_values("request_id").head(preflight_requests)[
            "request_id"
        ]
        replay = replay[replay["request_id"].isin(selected_request_ids)].copy()
        materialized = materialized[materialized["request_id"].isin(selected_request_ids)].copy()
        expected_requests = preflight_requests
        expected_rows = len(materialized)
    if replay["request_id"].nunique() != expected_requests:
        raise AssertionError(
            f"cache requests {replay['request_id'].nunique()} != {expected_requests}"
        )
    if len(materialized) != expected_rows:
        raise AssertionError(f"cache rows {len(materialized)} != {expected_rows}")

    exp218_config = yaml.safe_load(_find("exp218_config.yaml").read_text())
    work_dir = Path("/kaggle/working/exp239_feature_cache_work")
    request_train_dir, request_map, public_runtime = _configure_pseudo_runtime(
        replay, frames, raw_train_dir, work_dir, config
    )
    sampled_rows = {
        str(request_id): set(group["row_index"].astype(int))
        for request_id, group in materialized.groupby("request_id", sort=False)
    }
    batch_requests = int(
        _nested(
            config,
            "model.exp218_augmentation.feature_cache.batch_request_count",
            25,
        )
    )
    n_jobs = int(_nested(config, "model.exp218_augmentation.feature_generation.n_jobs", 4))
    cache_dir = output_dir / f"{OUTPUT_PREFIX}_exp218_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    reference_features: list[str] | None = None

    total_batches = (len(request_map) + batch_requests - 1) // batch_requests
    for batch_index, start in enumerate(range(0, len(request_map), batch_requests)):
        batch_map = request_map.iloc[start : start + batch_requests].copy()
        parts = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_build_one_base_request)(row, request_train_dir, sampled_rows, public_runtime)
            for row in batch_map.itertuples(index=False)
        )
        pseudo_base = pd.concat(parts, ignore_index=True)
        del parts
        pseudo_base_features = public.feature_columns_for_variant(
            pseudo_base, "pixiux_likpf_public_replay"
        )
        pseudo_base_features = [
            col
            for col in pseudo_base_features
            if col not in {"request_id", "source_well", "source_fold"}
        ]
        pseudo_learned = _generate_pseudo_learned(pseudo_base, request_train_dir, config)
        pseudo_full, feature_columns = _assemble_full_features(
            pseudo_base,
            pseudo_learned,
            request_train_dir,
            exp218_config,
            base_feature_columns=pseudo_base_features,
        )
        if len(feature_columns) != expected_features:
            raise AssertionError(
                f"batch {batch_index} features {len(feature_columns)} != {expected_features}"
            )
        if reference_features is None:
            reference_features = list(feature_columns)
        elif feature_columns != reference_features:
            raise AssertionError(f"batch {batch_index} feature schema drift")
        metadata_columns = [
            "id",
            "well",
            "target",
            "request_id",
            "source_well",
            "source_fold",
        ]
        shard = pseudo_full[metadata_columns + feature_columns].copy()
        expected_batch_rows = int(materialized["request_id"].isin(batch_map["request_id"]).sum())
        if len(shard) != expected_batch_rows:
            raise AssertionError(f"batch {batch_index} rows {len(shard)} != {expected_batch_rows}")
        if not np.isfinite(shard["target"].to_numpy(np.float32)).all():
            raise AssertionError(f"batch {batch_index} target contains non-finite values")
        shard_path = cache_dir / f"part-{batch_index:04d}.parquet"
        shard.to_parquet(shard_path, index=False, compression="zstd")
        manifest_rows.append(
            {
                "batch_index": batch_index,
                "path": f"{cache_dir.name}/{shard_path.name}",
                "requests": int(batch_map["request_id"].nunique()),
                "rows": int(len(shard)),
                "columns": int(len(shard.columns)),
                "file_bytes": int(shard_path.stat().st_size),
                "file_sha256": _sha256(shard_path),
                "row_content_sha256": _row_content_sha256(shard),
                "elapsed_seconds": time.time() - started,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            }
        )
        print(
            f"feature-cache batch {batch_index + 1}/{total_batches}: "
            f"requests={len(batch_map)} rows={len(shard)} "
            f"peak_rss_mb={manifest_rows[-1]['peak_rss_mb']:.1f}",
            flush=True,
        )
        del pseudo_base, pseudo_learned, pseudo_full, shard, batch_map
        gc.collect()

    if reference_features is None:
        raise AssertionError("feature cache produced no batches")
    manifest = pd.DataFrame(manifest_rows)
    if int(manifest["requests"].sum()) != expected_requests:
        raise AssertionError("feature cache request total mismatch")
    if int(manifest["rows"].sum()) != expected_rows:
        raise AssertionError("feature cache row total mismatch")
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_exp218_feature_cache_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    schema_path = output_dir / f"{OUTPUT_PREFIX}_exp218_feature_cache_schema.csv"
    pd.DataFrame(
        {"feature_index": range(len(reference_features)), "feature": reference_features}
    ).to_csv(schema_path, index=False, lineterminator="\n")
    request_manifest_path = output_dir / f"{OUTPUT_PREFIX}_exp218_feature_cache_requests.csv"
    request_map.to_csv(request_manifest_path, index=False, lineterminator="\n")
    result = {
        "stage": "cpu_feature_cache_preflight" if preflight else "cpu_feature_cache",
        "preflight": preflight,
        "requests": expected_requests,
        "rows": expected_rows,
        "features": len(reference_features),
        "shards": len(manifest),
        "batch_request_count": batch_requests,
        "feature_columns_sha256": hashlib.sha256(
            "\n".join(reference_features).encode()
        ).hexdigest(),
        "manifest_sha256": _sha256(manifest_path),
        "schema_sha256": _sha256(schema_path),
        "request_manifest_sha256": _sha256(request_manifest_path),
        "elapsed_seconds": time.time() - started,
        "peak_rss_mb": float(manifest["peak_rss_mb"].max()),
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_retrained": False,
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_exp218_feature_cache_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def run_official_feature_cache_generation(
    *,
    raw_train_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218

    started = time.time()
    expected_rows = int(
        _nested(config, "model.exp218_augmentation.expected_official_rows", 3_783_989)
    )
    expected_features = int(
        _nested(config, "model.exp218_augmentation.expected_feature_count", 380)
    )
    row_batch_count = int(
        _nested(
            config,
            "model.exp218_augmentation.official_feature_cache.row_batch_count",
            250_000,
        )
    )
    expected_shards = (expected_rows + row_batch_count - 1) // row_batch_count
    configured_shards = int(
        _nested(
            config,
            "model.exp218_augmentation.official_feature_cache.expected_shards",
            expected_shards,
        )
    )
    if configured_shards != expected_shards:
        raise AssertionError(f"official expected shards {configured_shards} != {expected_shards}")

    exp218_config = yaml.safe_load(_find("exp218_config.yaml").read_text())
    official_base, official_base_features, _ = exp218.load_exp072_full_replay_cache_frame(None)
    official_learned, _ = exp218.load_learned_likelihood_ml_features(None)
    official_full, feature_columns = _assemble_full_features(
        official_base,
        official_learned,
        raw_train_dir,
        exp218_config,
        base_feature_columns=official_base_features,
    )
    del official_base, official_learned
    gc.collect()
    if len(official_full) != expected_rows:
        raise AssertionError(f"official cache rows {len(official_full)} != {expected_rows}")
    if len(feature_columns) != expected_features:
        raise AssertionError(
            f"official cache features {len(feature_columns)} != {expected_features}"
        )
    if not np.isfinite(official_full["target"].to_numpy(np.float32)).all():
        raise AssertionError("official cache target contains non-finite values")

    cache_dir = output_dir / f"{OUTPUT_PREFIX}_official_exp218_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_columns = ["id", "well", "target"]
    manifest_rows: list[dict[str, Any]] = []
    total_batches = (len(official_full) + row_batch_count - 1) // row_batch_count
    for batch_index, start in enumerate(range(0, len(official_full), row_batch_count)):
        stop = min(start + row_batch_count, len(official_full))
        shard = official_full.iloc[start:stop][metadata_columns + feature_columns].copy()
        shard_path = cache_dir / f"part-{batch_index:04d}.parquet"
        shard.to_parquet(shard_path, index=False, compression="zstd")
        manifest_rows.append(
            {
                "batch_index": batch_index,
                "path": f"{cache_dir.name}/{shard_path.name}",
                "row_start": start,
                "row_stop": stop,
                "rows": int(len(shard)),
                "wells": int(shard["well"].nunique()),
                "columns": int(len(shard.columns)),
                "file_bytes": int(shard_path.stat().st_size),
                "file_sha256": _sha256(shard_path),
                "row_content_sha256": _row_content_sha256(shard),
                "elapsed_seconds": time.time() - started,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            }
        )
        print(
            f"official-cache batch {batch_index + 1}/{total_batches}: "
            f"rows={len(shard)} peak_rss_mb={manifest_rows[-1]['peak_rss_mb']:.1f}",
            flush=True,
        )
        del shard
        gc.collect()

    manifest = pd.DataFrame(manifest_rows)
    if len(manifest) != expected_shards or int(manifest["rows"].sum()) != expected_rows:
        raise AssertionError("official cache manifest totals mismatch")
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_official_exp218_feature_cache_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    schema_path = output_dir / f"{OUTPUT_PREFIX}_official_exp218_feature_cache_schema.csv"
    pd.DataFrame({"feature_index": range(len(feature_columns)), "feature": feature_columns}).to_csv(
        schema_path, index=False, lineterminator="\n"
    )
    result = {
        "stage": "cpu_official_feature_cache",
        "rows": expected_rows,
        "wells": int(official_full["well"].nunique()),
        "features": len(feature_columns),
        "shards": len(manifest),
        "row_batch_count": row_batch_count,
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_columns).encode()).hexdigest(),
        "manifest_sha256": _sha256(manifest_path),
        "schema_sha256": _sha256(schema_path),
        "elapsed_seconds": time.time() - started,
        "peak_rss_mb": float(manifest["peak_rss_mb"].max()),
        "lightgbm_configs": 0,
        "folds": 0,
        "boosters": 0,
        "parent_control_retrained": False,
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_official_exp218_feature_cache_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _generate_pseudo_learned(
    pseudo_base: pd.DataFrame,
    request_train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    import learned_likelihood_rawtest_feature_generator_parity as learned
    from pf_multi_observation_likelihood_probe import build_multi_observation_candidate_frame

    candidates = learned.candidate_specs_from_config(config)
    source = learned.ensure_candidate_value_columns(pseudo_base, candidates)
    candidate_names = [spec.name for spec in candidates]
    candidate_frame = source[["id", "well", *candidate_names]].copy()
    multiobs, _summary = build_multi_observation_candidate_frame(
        source,
        candidate_frame,
        train_dir=request_train_dir,
        candidate_names=candidate_names,
        config=dict(_nested(config, "generator.multi_observation_likelihood", {})),
    )
    source = source.merge(multiobs, on=["id", "well"], how="left", validate="one_to_one")
    schema_path = learned.find_artifact(learned.DEFAULT_EXP111_SCHEMA)
    manifest_path = learned.find_artifact(learned.DEFAULT_EXP111_MANIFEST)
    row_features = learned.load_feature_schema(schema_path)
    model_features = learned.exp111_model_feature_columns(row_features)
    classifier, error_model, _meta = learned.load_exp111_models(manifest_path=manifest_path)
    required = learned.source_required_columns(config, candidates)
    features, _long = learned.generate_ml_features_from_frame(
        source[required],
        candidates=candidates,
        row_feature_columns=row_features,
        model_feature_columns=model_features,
        classifier=classifier,
        error_model=error_model,
        config=config,
    )
    return features


def _assemble_full_features(
    base: pd.DataFrame,
    learned_source: pd.DataFrame,
    train_dir: Path,
    exp218_config: dict[str, Any],
    *,
    base_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218

    frame = base.reset_index(drop=True).copy()
    missing_base = [col for col in base_feature_columns if col not in frame]
    if missing_base:
        raise AssertionError(f"missing exp072 base features: {missing_base[:10]}")
    frame, _anchor = exp218.add_anchor_columns(frame, train_dir)
    projection, projection_groups, _ = exp218.build_u_projection_features(
        frame,
        source_specs=dict(_nested(exp218_config, "model.u_projection.sources", {})),
        degree=int(_nested(exp218_config, "model.u_projection.degree", 3)),
        robust_iters=int(_nested(exp218_config, "model.u_projection.robust_iters", 3)),
        clip_sigma=float(_nested(exp218_config, "model.u_projection.clip_sigma", 4.0)),
    )
    projection_cols = [col for col in projection if col not in {"id", "well"}]
    frame = pd.concat(
        [
            frame,
            projection[projection_cols].reset_index(drop=True).astype(np.float32),
        ],
        axis=1,
    )
    learned_features, learned_groups, _ = exp218.build_learned_likelihood_features(
        learned_source, frame, dict(_nested(exp218_config, "model.learned_likelihood_features", {}))
    )
    learned_cols = [col for col in learned_features if col not in {"id", "well"}]
    frame = pd.concat(
        [
            frame,
            learned_features[learned_cols].reset_index(drop=True).astype(np.float32),
        ],
        axis=1,
    )
    grwr, grwr_groups, _, _ = exp218.build_gr_wavelet_rotation_confidence_features(
        frame,
        train_dir=train_dir,
        config=dict(_nested(exp218_config, "model.gr_wavelet_rotation_confidence_features", {})),
    )
    grwr_cols = [col for col in grwr if col not in {"id", "well"}]
    frame = pd.concat([frame, grwr[grwr_cols].reset_index(drop=True).astype(np.float32)], axis=1)
    groups = {**projection_groups, **learned_groups, **grwr_groups}
    variants = _nested(exp218_config, "model.feature_ablation.active_variants", [])
    selected = next(item for item in variants if item.get("enabled", True))
    feature_columns = exp218.feature_columns_for_variant(base_feature_columns, groups, selected)
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise AssertionError(f"duplicate assembled feature columns: {duplicate_columns[:10]}")
    return frame, feature_columns


def _load_pseudo_feature_cache(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest_path = _find(f"{OUTPUT_PREFIX}_exp218_feature_cache_manifest.csv")
    schema_path = _find(f"{OUTPUT_PREFIX}_exp218_feature_cache_schema.csv")
    summary_path = _find(f"{OUTPUT_PREFIX}_exp218_feature_cache_summary.json")
    manifest = pd.read_csv(manifest_path).sort_values("batch_index")
    schema = pd.read_csv(schema_path).sort_values("feature_index")
    summary = json.loads(summary_path.read_text())
    expected_rows = int(_nested(config, "model.exp218_augmentation.expected_pseudo_rows", 799_961))
    expected_requests = int(_nested(config, "model.exp218_augmentation.pseudo_request_count", 800))
    expected_features = int(
        _nested(config, "model.exp218_augmentation.expected_feature_count", 380)
    )
    expected_shards = int(
        _nested(config, "model.exp218_augmentation.feature_cache.expected_shards", 32)
    )
    feature_columns = schema["feature"].astype(str).tolist()
    if summary.get("preflight"):
        raise AssertionError("refusing to train from feature-cache preflight output")
    if len(manifest) != expected_shards or len(feature_columns) != expected_features:
        raise AssertionError("feature-cache shard/schema count mismatch")
    if int(manifest["rows"].sum()) != expected_rows:
        raise AssertionError("feature-cache row count mismatch")
    if int(manifest["requests"].sum()) != expected_requests:
        raise AssertionError("feature-cache request count mismatch")
    parts: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        shard_path = manifest_path.parent / str(row.path)
        if not shard_path.exists():
            shard_path = _find(Path(str(row.path)).name)
        if _sha256(shard_path) != str(row.file_sha256):
            raise AssertionError(f"feature-cache file SHA mismatch: {shard_path.name}")
        shard = pd.read_parquet(shard_path)
        if len(shard) != int(row.rows):
            raise AssertionError(f"feature-cache row mismatch: {shard_path.name}")
        if _row_content_sha256(shard) != str(row.row_content_sha256):
            raise AssertionError(f"feature-cache content SHA mismatch: {shard_path.name}")
        parts.append(shard)
        print(
            f"loaded feature-cache shard {int(row.batch_index) + 1}/{len(manifest)} "
            f"rows={len(shard)}",
            flush=True,
        )
    pseudo = pd.concat(parts, ignore_index=True)
    del parts
    if len(pseudo) != expected_rows:
        raise AssertionError("concatenated feature-cache row count mismatch")
    required = {
        "id",
        "well",
        "target",
        "request_id",
        "source_well",
        "source_fold",
        *feature_columns,
    }
    missing = sorted(required - set(pseudo.columns))
    if missing:
        raise AssertionError(f"feature-cache missing columns: {missing[:10]}")
    return pseudo, feature_columns, summary


def _cache_contract(
    *,
    manifest_name: str,
    schema_name: str,
    summary_name: str,
    expected_rows: int,
    expected_shards: int,
    expected_features: int,
) -> tuple[Path, pd.DataFrame, list[str], dict[str, Any]]:
    manifest_path = _find(manifest_name)
    schema_path = _find(schema_name)
    summary_path = _find(summary_name)
    manifest = pd.read_csv(manifest_path).sort_values("batch_index").reset_index(drop=True)
    schema = pd.read_csv(schema_path).sort_values("feature_index").reset_index(drop=True)
    summary = json.loads(summary_path.read_text())
    features = schema["feature"].astype(str).tolist()
    if len(manifest) != expected_shards or int(manifest["rows"].sum()) != expected_rows:
        raise AssertionError(f"cache manifest totals mismatch: {manifest_name}")
    if len(features) != expected_features:
        raise AssertionError(f"cache feature count mismatch: {schema_name}")
    if _sha256(manifest_path) != str(summary["manifest_sha256"]):
        raise AssertionError(f"cache manifest SHA mismatch: {manifest_name}")
    if _sha256(schema_path) != str(summary["schema_sha256"]):
        raise AssertionError(f"cache schema SHA mismatch: {schema_name}")
    feature_sha = hashlib.sha256("\n".join(features).encode()).hexdigest()
    if feature_sha != str(summary["feature_columns_sha256"]):
        raise AssertionError(f"cache feature columns SHA mismatch: {schema_name}")
    return manifest_path, manifest, features, summary


def _cache_shard_path(manifest_path: Path, relative_path: str) -> Path:
    shard_path = manifest_path.parent / relative_path
    if not shard_path.exists():
        shard_path = _find(Path(relative_path).name)
    return shard_path


def _stream_dual_cache_to_memmaps(config: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    expected_official = int(
        _nested(config, "model.exp218_augmentation.expected_official_rows", 3_783_989)
    )
    expected_pseudo = int(
        _nested(config, "model.exp218_augmentation.expected_pseudo_rows", 799_961)
    )
    expected_features = int(
        _nested(config, "model.exp218_augmentation.expected_feature_count", 380)
    )
    official_shards = int(
        _nested(config, "model.exp218_augmentation.official_feature_cache.expected_shards", 16)
    )
    pseudo_shards = int(
        _nested(config, "model.exp218_augmentation.feature_cache.expected_shards", 32)
    )
    official_contract = _cache_contract(
        manifest_name=f"{OUTPUT_PREFIX}_official_exp218_feature_cache_manifest.csv",
        schema_name=f"{OUTPUT_PREFIX}_official_exp218_feature_cache_schema.csv",
        summary_name=f"{OUTPUT_PREFIX}_official_exp218_feature_cache_summary.json",
        expected_rows=expected_official,
        expected_shards=official_shards,
        expected_features=expected_features,
    )
    pseudo_contract = _cache_contract(
        manifest_name=f"{OUTPUT_PREFIX}_exp218_feature_cache_manifest.csv",
        schema_name=f"{OUTPUT_PREFIX}_exp218_feature_cache_schema.csv",
        summary_name=f"{OUTPUT_PREFIX}_exp218_feature_cache_summary.json",
        expected_rows=expected_pseudo,
        expected_shards=pseudo_shards,
        expected_features=expected_features,
    )
    official_manifest_path, official_manifest, features, official_summary = official_contract
    pseudo_manifest_path, pseudo_manifest, pseudo_features, pseudo_summary = pseudo_contract
    if pseudo_features != features:
        raise AssertionError("official/pseudo cached exp218 feature schema mismatch")
    if int(pseudo_manifest["requests"].sum()) != int(
        _nested(config, "model.exp218_augmentation.pseudo_request_count", 800)
    ):
        raise AssertionError("pseudo cache request count mismatch")

    work_dir.mkdir(parents=True, exist_ok=True)
    official_x = np.memmap(
        work_dir / "official_x.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(expected_official, expected_features),
    )
    official_y = np.memmap(
        work_dir / "official_y.float32.mmap", dtype=np.float32, mode="w+", shape=expected_official
    )
    official_ids = np.memmap(
        work_dir / "official_ids.s64.mmap", dtype="S64", mode="w+", shape=expected_official
    )
    official_groups = np.memmap(
        work_dir / "official_groups.int32.mmap",
        dtype=np.int32,
        mode="w+",
        shape=expected_official,
    )
    group_labels: list[str] = []
    group_to_code: dict[str, int] = {}
    offset = 0
    for row in official_manifest.itertuples(index=False):
        shard_path = _cache_shard_path(official_manifest_path, str(row.path))
        if _sha256(shard_path) != str(row.file_sha256):
            raise AssertionError(f"official cache file SHA mismatch: {shard_path.name}")
        shard = pd.read_parquet(shard_path)
        if len(shard) != int(row.rows) or _row_content_sha256(shard) != str(row.row_content_sha256):
            raise AssertionError(f"official cache content mismatch: {shard_path.name}")
        missing = sorted({"id", "well", "target", *features} - set(shard.columns))
        if missing:
            raise AssertionError(f"official cache missing columns: {missing[:10]}")
        stop = offset + len(shard)
        official_x[offset:stop] = shard[features].to_numpy(np.float32, copy=False)
        official_y[offset:stop] = shard["target"].to_numpy(np.float32)
        ids = shard["id"].astype(str)
        if int(ids.str.len().max()) > 64:
            raise AssertionError("official id exceeds S64 cache contract")
        official_ids[offset:stop] = ids.to_numpy(dtype="S64")
        wells = shard["well"].astype(str)
        for well in pd.unique(wells):
            if well not in group_to_code:
                group_to_code[well] = len(group_labels)
                group_labels.append(well)
        official_groups[offset:stop] = wells.map(group_to_code).to_numpy(np.int32)
        offset = stop
        print(
            f"streamed official cache shard {int(row.batch_index) + 1}/{len(official_manifest)} "
            f"rows={len(shard)} peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}",
            flush=True,
        )
        del shard, ids, wells
        gc.collect()
    if offset != expected_official:
        raise AssertionError("streamed official cache row mismatch")
    if "last_known_tvt" not in features:
        raise AssertionError("last_known_tvt missing from cached feature schema")
    official_base = np.memmap(
        work_dir / "official_base.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=expected_official,
    )
    official_base[:] = official_x[:, features.index("last_known_tvt")]

    pseudo_x = np.memmap(
        work_dir / "pseudo_x.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(expected_pseudo, expected_features),
    )
    pseudo_y = np.memmap(
        work_dir / "pseudo_y.float32.mmap", dtype=np.float32, mode="w+", shape=expected_pseudo
    )
    pseudo_source = np.memmap(
        work_dir / "pseudo_source.int32.mmap",
        dtype=np.int32,
        mode="w+",
        shape=expected_pseudo,
    )
    offset = 0
    for row in pseudo_manifest.itertuples(index=False):
        shard_path = _cache_shard_path(pseudo_manifest_path, str(row.path))
        if _sha256(shard_path) != str(row.file_sha256):
            raise AssertionError(f"pseudo cache file SHA mismatch: {shard_path.name}")
        shard = pd.read_parquet(shard_path)
        if len(shard) != int(row.rows) or _row_content_sha256(shard) != str(row.row_content_sha256):
            raise AssertionError(f"pseudo cache content mismatch: {shard_path.name}")
        missing = sorted({"target", "source_well", *features} - set(shard.columns))
        if missing:
            raise AssertionError(f"pseudo cache missing columns: {missing[:10]}")
        stop = offset + len(shard)
        pseudo_x[offset:stop] = shard[features].to_numpy(np.float32, copy=False)
        pseudo_y[offset:stop] = shard["target"].to_numpy(np.float32)
        codes = shard["source_well"].astype(str).map(group_to_code)
        if codes.isna().any():
            raise AssertionError("pseudo cache contains unknown source well")
        pseudo_source[offset:stop] = codes.to_numpy(np.int32)
        offset = stop
        print(
            f"streamed pseudo cache shard {int(row.batch_index) + 1}/{len(pseudo_manifest)} "
            f"rows={len(shard)} peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}",
            flush=True,
        )
        del shard, codes
        gc.collect()
    if offset != expected_pseudo:
        raise AssertionError("streamed pseudo cache row mismatch")
    for array in [
        official_x,
        official_y,
        official_ids,
        official_groups,
        official_base,
        pseudo_x,
        pseudo_y,
        pseudo_source,
    ]:
        array.flush()
    return {
        "features": features,
        "official_x": official_x,
        "official_y": official_y,
        "official_ids": official_ids,
        "official_groups": official_groups,
        "official_group_labels": group_labels,
        "official_base": official_base,
        "pseudo_x": pseudo_x,
        "pseudo_y": pseudo_y,
        "pseudo_source": pseudo_source,
        "official_summary": official_summary,
        "pseudo_summary": pseudo_summary,
    }


def _train_augmented(
    *,
    official_ids: np.ndarray,
    official_groups: np.ndarray,
    official_y: np.ndarray,
    official_base: np.ndarray,
    official_x: np.ndarray,
    pseudo_y: np.ndarray,
    pseudo_source: np.ndarray,
    pseudo_x: np.ndarray,
    features: list[str],
    config: dict[str, Any],
    exp218_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    mode = dict(_nested(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8", {}))
    params_list = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
    folds = GroupKFold(n_splits=5)
    official_tvt = official_base + official_y
    official_weight = float(_nested(config, "model.exp218_augmentation.official_row_weight", 1.0))
    pseudo_weight = float(_nested(config, "model.exp218_augmentation.pseudo_row_weight", 0.5))
    oofs: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_augmentation_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_index, params in enumerate(params_list):
        oof = np.zeros(len(official_y), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(
            folds.split(np.arange(len(official_y)), official_y, groups=official_groups)
        ):
            valid_wells = set(official_groups[valid_idx])
            pseudo_mask = np.asarray([well not in valid_wells for well in pseudo_source])
            pseudo_idx = np.flatnonzero(pseudo_mask)
            x_train = np.empty((len(train_idx) + len(pseudo_idx), len(features)), dtype=np.float32)
            np.take(official_x, train_idx, axis=0, out=x_train[: len(train_idx)])
            np.take(pseudo_x, pseudo_idx, axis=0, out=x_train[len(train_idx) :])
            y_train = np.concatenate([official_y[train_idx], pseudo_y[pseudo_mask]])
            weights = np.concatenate(
                [
                    np.full(len(train_idx), official_weight, np.float32),
                    np.full(int(pseudo_mask.sum()), pseudo_weight, np.float32),
                ]
            )
            x_valid = np.empty((len(valid_idx), len(features)), dtype=np.float32)
            np.take(official_x, valid_idx, axis=0, out=x_valid)
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                y_train,
                sample_weight=weights,
                eval_set=[(x_valid, official_y[valid_idx])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best = int(model.best_iteration_ or params.get("n_estimators", 0))
            prediction = model.predict(x_valid, num_iteration=best).astype(np.float32)
            oof[valid_idx] = prediction
            model_path = model_dir / f"lgb{model_index}_fold{fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best)
            importance_rows.extend(
                {
                    "model": f"lgb{model_index}",
                    "fold": fold,
                    "feature": feature,
                    "gain": float(gain),
                }
                for feature, gain in zip(
                    features,
                    model.booster_.feature_importance(importance_type="gain"),
                    strict=True,
                )
            )
            metric_rows.append(
                {
                    "model": f"lgb{model_index}",
                    "fold": fold,
                    "official_train_rows": len(train_idx),
                    "pseudo_train_rows": int(pseudo_mask.sum()),
                    "valid_rows": len(valid_idx),
                    "best_iteration": best,
                    "rmse_tvt": float(
                        np.sqrt(
                            np.mean(
                                (official_tvt[valid_idx] - (official_base[valid_idx] + prediction))
                                ** 2
                            )
                        )
                    ),
                    "model_sha256": _sha256(model_path),
                }
            )
            del x_train, y_train, weights, x_valid, model, pseudo_idx
            gc.collect()
        oofs.append(oof)
        metric_rows.append(
            {
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rmse_tvt": float(np.sqrt(np.mean((official_tvt - (official_base + oof)) ** 2))),
            }
        )
    mean_oof = np.mean(np.vstack(oofs), axis=0).astype(np.float32)
    mean_tvt = official_base + mean_oof
    overall = float(np.sqrt(np.mean((official_tvt - mean_tvt) ** 2)))
    predictions = pd.DataFrame({"id": official_ids, "well": official_groups})
    predictions["target_tvt"] = official_tvt
    predictions["pred_tvt"] = mean_tvt
    predictions.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_metrics.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_feature_importance.csv",
        index=False,
    )
    return {
        "official_oof_rmse": overall,
        "reference_exp218_rmse": float(
            _nested(config, "model.exp218_augmentation.reference_exp218_lgb_mean_rmse")
        ),
        "delta_vs_exp218": overall
        - float(_nested(config, "model.exp218_augmentation.reference_exp218_lgb_mean_rmse")),
        "official_rows": len(official_y),
        "pseudo_rows": len(pseudo_y),
        "features": len(features),
        "prediction_content_sha256": _sha256(
            output_dir / f"{OUTPUT_PREFIX}_augmentation_predictions.csv.gz", decompressed=True
        ),
    }


def run_full_augmentation_evaluation(
    *,
    replay: pd.DataFrame,
    materialized: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    raw_train_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218
    import public_notebook_replay_audit as public

    started = time.time()
    exp218_config = yaml.safe_load(_find("exp218_config.yaml").read_text())
    work_dir = Path("/kaggle/working/exp239_augmentation_work")
    preflight = bool(_nested(config, "model.exp218_augmentation.preflight.enabled", False))
    if preflight:
        request_count = int(_nested(config, "model.exp218_augmentation.preflight.request_count", 2))
        selected_requests = replay.sort_values("request_id").head(request_count)["request_id"]
        replay = replay[replay["request_id"].isin(selected_requests)].copy()
        materialized = materialized[materialized["request_id"].isin(selected_requests)].copy()
    pseudo_base, request_train_dir, request_map = _build_pseudo_base(
        replay, materialized, frames, raw_train_dir, work_dir, config
    )
    pseudo_base_features = public.feature_columns_for_variant(
        pseudo_base, "pixiux_likpf_public_replay"
    )
    pseudo_base_features = [
        col
        for col in pseudo_base_features
        if col not in {"request_id", "source_well", "source_fold"}
    ]
    pseudo_learned = _generate_pseudo_learned(pseudo_base, request_train_dir, config)
    pseudo_full, pseudo_features = _assemble_full_features(
        pseudo_base,
        pseudo_learned,
        request_train_dir,
        exp218_config,
        base_feature_columns=pseudo_base_features,
    )
    expected_features = int(
        _nested(config, "model.exp218_augmentation.expected_feature_count", 380)
    )
    if preflight:
        if len(pseudo_features) != expected_features:
            raise AssertionError(
                f"preflight feature count {len(pseudo_features)} != {expected_features}"
            )
        result = {
            "preflight": True,
            "requests": int(replay["request_id"].nunique()),
            "pseudo_rows": int(len(pseudo_full)),
            "features": int(len(pseudo_features)),
            "feature_columns_sha256": hashlib.sha256(
                "\n".join(pseudo_features).encode()
            ).hexdigest(),
            "elapsed_seconds": time.time() - started,
            "lightgbm_configs": 0,
            "folds": 0,
            "boosters": 0,
            "parent_control_retrained": False,
        }
        summary_path = output_dir / f"{OUTPUT_PREFIX}_augmentation_preflight_summary.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    official_base, official_base_features, _ = exp218.load_exp072_full_replay_cache_frame(None)
    official_learned, _ = exp218.load_learned_likelihood_ml_features(None)
    official_full, official_features = _assemble_full_features(
        official_base,
        official_learned,
        raw_train_dir,
        exp218_config,
        base_feature_columns=official_base_features,
    )
    if official_features != pseudo_features:
        raise AssertionError("official/pseudo exp218 feature schema mismatch")
    if len(official_features) != expected_features:
        raise AssertionError(
            f"exp218 feature count {len(official_features)} != {expected_features}"
        )
    official_ids = official_full["id"].astype(str).to_numpy()
    official_groups = official_full["well"].astype(str).to_numpy()
    official_y = official_full["target"].to_numpy(np.float32)
    official_base_tvt = official_full["last_known_tvt"].to_numpy(np.float32)
    official_x = official_full[official_features].to_numpy(np.float32, copy=True)
    pseudo_y = pseudo_full["target"].to_numpy(np.float32)
    pseudo_source = pseudo_full["source_well"].astype(str).to_numpy()
    pseudo_x = pseudo_full[pseudo_features].to_numpy(np.float32, copy=True)
    del official_full, pseudo_full, official_base, official_learned, pseudo_base, pseudo_learned
    gc.collect()
    result = _train_augmented(
        official_ids=official_ids,
        official_groups=official_groups,
        official_y=official_y,
        official_base=official_base_tvt,
        official_x=official_x,
        pseudo_y=pseudo_y,
        pseudo_source=pseudo_source,
        pseudo_x=pseudo_x,
        features=official_features,
        config=config,
        exp218_config=exp218_config,
        output_dir=output_dir,
    )
    schema = pd.DataFrame(
        {"feature_index": range(len(official_features)), "feature": official_features}
    )
    schema.to_csv(output_dir / f"{OUTPUT_PREFIX}_augmentation_feature_schema.csv", index=False)
    request_map.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_request_manifest.csv", index=False
    )
    result.update(
        {
            "elapsed_seconds": time.time() - started,
            "feature_schema_sha256": _sha256(
                output_dir / f"{OUTPUT_PREFIX}_augmentation_feature_schema.csv"
            ),
            "parent_control_retrained": False,
            "lightgbm_configs": 3,
            "folds": 5,
            "boosters": 15,
        }
    )
    summary_path = output_dir / f"{OUTPUT_PREFIX}_augmentation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _write_streaming_predictions(
    *,
    path: Path,
    official_ids: np.ndarray,
    official_groups: np.ndarray,
    group_labels: list[str],
    target_tvt: np.ndarray,
    pred_tvt: np.ndarray,
    chunk_rows: int = 250_000,
) -> None:
    labels = np.asarray(group_labels)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for start in range(0, len(official_ids), chunk_rows):
                stop = min(start + chunk_rows, len(official_ids))
                frame = pd.DataFrame(
                    {
                        "id": np.char.decode(np.asarray(official_ids[start:stop]), "utf-8"),
                        "well": labels[np.asarray(official_groups[start:stop])],
                        "target_tvt": np.asarray(target_tvt[start:stop]),
                        "pred_tvt": np.asarray(pred_tvt[start:stop]),
                    }
                )
                frame.to_csv(compressed, index=False, header=start == 0, lineterminator="\n")
                del frame


def _train_augmented_memmaps(
    *,
    cache: dict[str, Any],
    config: dict[str, Any],
    exp218_config: dict[str, Any],
    output_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    import gr_wavelet_rotation_confidence_features_on_exp148 as exp218
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    features = cache["features"]
    official_x = cache["official_x"]
    official_y = cache["official_y"]
    official_base = cache["official_base"]
    official_groups = cache["official_groups"]
    pseudo_x = cache["pseudo_x"]
    pseudo_y = cache["pseudo_y"]
    pseudo_source = cache["pseudo_source"]
    mode = dict(_nested(exp218_config, "model.training.modes.gpu_repro_guard_dp_threads8", {}))
    params_list = exp218.apply_mode_overrides(exp218.exp063_lgb_config_family(fast=False), mode)
    expected_configs = int(_nested(config, "model.exp218_augmentation.lightgbm_configs", 3))
    expected_folds = int(_nested(config, "model.exp218_augmentation.folds", 5))
    if len(params_list) != expected_configs or expected_folds != 5:
        raise AssertionError("LightGBM config/fold count drift")
    folds = GroupKFold(n_splits=expected_folds)
    official_tvt = np.asarray(official_base) + np.asarray(official_y)
    official_weight = float(_nested(config, "model.exp218_augmentation.official_row_weight", 1.0))
    pseudo_weight = float(_nested(config, "model.exp218_augmentation.pseudo_row_weight", 0.5))
    oofs: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_augmentation_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_index, params in enumerate(params_list):
        oof = np.zeros(len(official_y), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(
            folds.split(official_y, official_y, groups=official_groups)
        ):
            valid_group_codes = np.unique(np.asarray(official_groups[valid_idx]))
            pseudo_mask = ~np.isin(pseudo_source, valid_group_codes)
            pseudo_idx = np.flatnonzero(pseudo_mask)
            train_rows = len(train_idx) + len(pseudo_idx)
            train_path = work_dir / f"fold_lgb{model_index}_{fold}_train.float32.mmap"
            valid_path = work_dir / f"fold_lgb{model_index}_{fold}_valid.float32.mmap"
            x_train = np.memmap(
                train_path,
                dtype=np.float32,
                mode="w+",
                shape=(train_rows, len(features)),
            )
            x_valid = np.memmap(
                valid_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(valid_idx), len(features)),
            )
            np.take(official_x, train_idx, axis=0, out=x_train[: len(train_idx)])
            np.take(pseudo_x, pseudo_idx, axis=0, out=x_train[len(train_idx) :])
            np.take(official_x, valid_idx, axis=0, out=x_valid)
            x_train.flush()
            x_valid.flush()
            y_train = np.concatenate([official_y[train_idx], pseudo_y[pseudo_idx]])
            weights = np.concatenate(
                [
                    np.full(len(train_idx), official_weight, np.float32),
                    np.full(len(pseudo_idx), pseudo_weight, np.float32),
                ]
            )
            print(
                f"training lgb{model_index} fold{fold}: official={len(train_idx)} "
                f"pseudo={len(pseudo_idx)} valid={len(valid_idx)} "
                f"peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}",
                flush=True,
            )
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                y_train,
                sample_weight=weights,
                eval_set=[(x_valid, official_y[valid_idx])],
                eval_metric="rmse",
                callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
            )
            best = int(model.best_iteration_ or params.get("n_estimators", 0))
            prediction = model.predict(x_valid, num_iteration=best).astype(np.float32)
            oof[valid_idx] = prediction
            model_path = model_dir / f"lgb{model_index}_fold{fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best)
            importance_rows.extend(
                {
                    "model": f"lgb{model_index}",
                    "fold": fold,
                    "feature": feature,
                    "gain": float(gain),
                }
                for feature, gain in zip(
                    features,
                    model.booster_.feature_importance(importance_type="gain"),
                    strict=True,
                )
            )
            metric_rows.append(
                {
                    "model": f"lgb{model_index}",
                    "fold": fold,
                    "official_train_rows": len(train_idx),
                    "pseudo_train_rows": len(pseudo_idx),
                    "valid_rows": len(valid_idx),
                    "best_iteration": best,
                    "rmse_tvt": float(
                        np.sqrt(
                            np.mean(
                                (official_tvt[valid_idx] - (official_base[valid_idx] + prediction))
                                ** 2
                            )
                        )
                    ),
                    "model_sha256": _sha256(model_path),
                }
            )
            del x_train, x_valid, y_train, weights, model, prediction, pseudo_idx, pseudo_mask
            gc.collect()
            train_path.unlink()
            valid_path.unlink()
        oofs.append(oof)
        metric_rows.append(
            {
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rmse_tvt": float(np.sqrt(np.mean((official_tvt - (official_base + oof)) ** 2))),
            }
        )
    mean_oof = np.mean(np.vstack(oofs), axis=0).astype(np.float32)
    mean_tvt = np.asarray(official_base) + mean_oof
    overall = float(np.sqrt(np.mean((official_tvt - mean_tvt) ** 2)))
    prediction_path = output_dir / f"{OUTPUT_PREFIX}_augmentation_predictions.csv.gz"
    _write_streaming_predictions(
        path=prediction_path,
        official_ids=cache["official_ids"],
        official_groups=official_groups,
        group_labels=cache["official_group_labels"],
        target_tvt=official_tvt,
        pred_tvt=mean_tvt,
    )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_metrics.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_augmentation_feature_importance.csv", index=False
    )
    return {
        "official_oof_rmse": overall,
        "reference_exp218_rmse": float(
            _nested(config, "model.exp218_augmentation.reference_exp218_lgb_mean_rmse")
        ),
        "delta_vs_exp218": overall
        - float(_nested(config, "model.exp218_augmentation.reference_exp218_lgb_mean_rmse")),
        "official_rows": len(official_y),
        "pseudo_rows": len(pseudo_y),
        "features": len(features),
        "prediction_content_sha256": _sha256(prediction_path, decompressed=True),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def run_cached_augmentation_evaluation(
    *,
    raw_train_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    exp218_config = yaml.safe_load(_find("exp218_config.yaml").read_text())
    work_dir = Path("/kaggle/working/exp239_dual_cache_memmaps")
    cache = _stream_dual_cache_to_memmaps(config, work_dir)
    print(
        f"training memmaps ready: official={cache['official_x'].shape} "
        f"pseudo={cache['pseudo_x'].shape} "
        f"peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.1f}",
        flush=True,
    )
    result = _train_augmented_memmaps(
        cache=cache,
        config=config,
        exp218_config=exp218_config,
        output_dir=output_dir,
        work_dir=work_dir,
    )
    schema_path = output_dir / f"{OUTPUT_PREFIX}_augmentation_feature_schema.csv"
    pd.DataFrame(
        {"feature_index": range(len(cache["features"])), "feature": cache["features"]}
    ).to_csv(schema_path, index=False, lineterminator="\n")
    result.update(
        {
            "stage": "gpu_dual_cache_streaming_training",
            "elapsed_seconds": time.time() - started,
            "feature_schema_sha256": _sha256(schema_path),
            "pseudo_feature_cache_manifest_sha256": cache["pseudo_summary"]["manifest_sha256"],
            "official_feature_cache_manifest_sha256": cache["official_summary"]["manifest_sha256"],
            "feature_cache_schema_sha256": cache["pseudo_summary"]["schema_sha256"],
            "parent_control_retrained": False,
            "lightgbm_configs": 3,
            "folds": 5,
            "boosters": 15,
        }
    )
    summary_path = output_dir / f"{OUTPUT_PREFIX}_augmentation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    for key in [
        "official_x",
        "official_y",
        "official_ids",
        "official_groups",
        "official_base",
        "pseudo_x",
        "pseudo_y",
        "pseudo_source",
    ]:
        array = cache.pop(key)
        array.flush()
        array._mmap.close()
    gc.collect()
    for mmap_path in work_dir.glob("*.mmap"):
        mmap_path.unlink()
    return result
