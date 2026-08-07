from __future__ import annotations

from pathlib import Path
from typing import Any

from exact_hmm_smoother import run_train_feature_cache
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config


def _hmm_config(config: dict[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.hmm") or {}
    keys = [
        "step",
        "n_rates",
        "rate_span",
        "sig_r",
        "sig_p",
        "df",
        "emission",
        "lam",
        "sigma_mode",
        "start_sig",
        "r0_sig",
        "band_pad",
        "mom",
        "rate_center",
    ]
    return {key: hmm[key] for key in keys if key in hmm}


def resolve_cluster_assignment(paths: ExperimentPaths, config: dict[str, Any]) -> Path:
    candidates = list(get_nested(config, "data.exp065_cluster_assignment_candidates") or [])
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        filename = "common_typewell_cluster_assignments.csv"
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob(filename)):
            checked.append(str(candidate))
            if candidate.stat().st_size > 0:
                return candidate
    raise FileNotFoundError("typewell cluster assignment was not found: " + " | ".join(checked))


def main() -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    feature_cache = get_nested(config, "feature_cache.hmm") or {}
    runtime = get_nested(config, "runtime") or {}
    peer_atlas_config = get_nested(config, "model.peer_atlas_emission") or {}
    validation = get_nested(config, "validation") or {}
    max_wells = feature_cache.get("max_wells")
    if max_wells is not None:
        max_wells = int(max_wells)
    target_wells = feature_cache.get("preflight_target_wells") or None
    if target_wells is not None and not isinstance(target_wells, list):
        raise ValueError("feature_cache.hmm.preflight_target_wells must be a list when set")
    return run_train_feature_cache(
        data_dir=paths.train_data_dir,
        output_dir=paths.artifacts_dir,
        hmm_config=_hmm_config(config),
        peer_atlas_config=peer_atlas_config,
        cluster_assignment_path=resolve_cluster_assignment(paths, config),
        n_folds=int(validation.get("n_folds", 5)),
        seed=int(validation.get("seed", 42)),
        output_prefix=str(feature_cache.get("output_prefix") or paths.experiment_name),
        max_wells=max_wells,
        target_wells=[str(well) for well in target_wells] if target_wells else None,
        fast=bool(feature_cache.get("fast", False)),
        numba_num_threads=runtime.get("numba_num_threads"),
        outer_workers=int(feature_cache.get("outer_workers") or 1),
    )


if __name__ == "__main__":
    main()
