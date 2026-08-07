from __future__ import annotations

from typing import Any

from exact_hmm_smoother import run_train_feature_cache
from settings import ExperimentPaths, get_nested, load_config


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


def main() -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    feature_cache = get_nested(config, "feature_cache.hmm") or {}
    self_gr_emission = get_nested(config, "self_gr_emission") or {}
    runtime = get_nested(config, "runtime") or {}
    max_wells = feature_cache.get("max_wells")
    if max_wells is not None:
        max_wells = int(max_wells)
    return run_train_feature_cache(
        data_dir=paths.train_data_dir,
        output_dir=paths.artifacts_dir,
        hmm_config=_hmm_config(config),
        self_gr_emission_config=self_gr_emission,
        output_prefix=str(feature_cache.get("output_prefix") or "exp225_state_known_tvt_self_gr_hmm_emission"),
        max_wells=max_wells,
        fast=bool(feature_cache.get("fast", False)),
        numba_num_threads=runtime.get("numba_num_threads"),
        outer_workers=int(feature_cache.get("outer_workers") or 1),
    )


if __name__ == "__main__":
    main()
