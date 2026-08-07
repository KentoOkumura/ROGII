import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # exp076_exp039_cv_reassessment train

    Train-side LightGBM reproducibility audit for the exp073 full replay model family evaluated on the exp039 CV surface. The notebook keeps low-level helpers in `exp073_exp039_cv_reassessment.py`, but performs the experiment orchestration in visible cells.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contents

    1. Setup and configuration
    2. Source artifact check
    3. Build exp039 CV training frame
    4. Configure audits and modes
    5. Fit LightGBM folds
    6. Save metrics and generated outputs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup and configuration
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import json
    import time

    import pandas as pd

    from settings import ExperimentPaths, get_nested, load_config
    from exp073_exp039_cv_reassessment import (
        EXP029_FEATURE_PATH,
        FULL_REPLAY_TRAIN_FEATURES,
        OUTPUT_PREFIX,
        build_exp039_cv_exp073_frame,
        exp039_cv_split_codes,
        find_artifact,
        find_path,
        _fit_one_mode,
    )


    def cfg_get(config, dotted_key, default=None):
        value = get_nested(config, dotted_key)
        return default if value is None else value

    return (
        EXP029_FEATURE_PATH,
        ExperimentPaths,
        FULL_REPLAY_TRAIN_FEATURES,
        OUTPUT_PREFIX,
        build_exp039_cv_exp073_frame,
        cfg_get,
        exp039_cv_split_codes,
        find_artifact,
        find_path,
        json,
        load_config,
        pd,
        time,
    )


@app.cell
def _(ExperimentPaths, OUTPUT_PREFIX, cfg_get, load_config, time):
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    train_started_at = time.time()

    print("Experiment:", config["experiment"]["name"])
    print("Route:", config["experiment"]["route"])
    print("Train mode:", cfg_get(config, "audit.mode"))
    print("Parent:", cfg_get(config, "lineage.parent"))
    print("Kernel sources:", cfg_get(config, "runtime.kaggle.kernel_sources"))
    print("Output prefix:", cfg_get(config, "audit.output_prefix", OUTPUT_PREFIX))
    return config, paths, train_started_at


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Source artifact check
    """)
    return


@app.cell
def _(
    EXP029_FEATURE_PATH,
    FULL_REPLAY_TRAIN_FEATURES,
    cfg_get,
    config,
    display,
    find_artifact,
    find_path,
    paths,
    pd,
):
    exp039_path = find_path(
        cfg_get(config, "data.exp039_cv_feature_path", EXP029_FEATURE_PATH),
        filename=EXP029_FEATURE_PATH.name,
    )
    cache_path = find_artifact(
        FULL_REPLAY_TRAIN_FEATURES,
        cfg_get(config, "data.exp072_train_feature_cache_local"),
    )

    print("exp039 CV surface:", exp039_path)
    print("exp073/exp072 full replay train cache:", cache_path)
    print("Artifacts dir:", paths.artifacts_dir)

    display(pd.read_csv(exp039_path, nrows=5, dtype={"id": str}))
    display(pd.read_csv(cache_path, nrows=5, dtype={"id": str, "well": str}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Build exp039 CV training frame
    """)
    return


@app.cell
def _(build_exp039_cv_exp073_frame, cfg_get, config, display, json):
    frame, feature_columns, join_stats = build_exp039_cv_exp073_frame(
        exp039_feature_path=cfg_get(config, "data.exp039_cv_feature_path"),
        cache_path=cfg_get(config, "data.exp072_train_feature_cache_local"),
        max_rows=cfg_get(config, "model.training.max_rows"),
    )

    print("Training rows:", len(frame))
    print("Wells:", frame["well_id"].nunique())
    print("Features:", len(feature_columns))
    print("Fold counts:")
    display(frame.groupby("fold").agg(rows=("id", "size"), wells=("well_id", "nunique")))
    print(json.dumps({k: v for k, v in join_stats.items() if k != "feature_source"}, indent=2, default=str))
    return feature_columns, frame, join_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Configure audits and modes
    """)
    return


@app.cell
def _(cfg_get, config, exp039_cv_split_codes, frame, json):
    mode_map = cfg_get(config, 'model.training.modes', {}) or {}
    selected_modes = list(cfg_get(config, 'model.training.active_modes', []) or mode_map)
    audits = tuple(cfg_get(config, 'validation.audits', ['leave_one_original_fold_out', 'well_hash_holdout']))
    well_hash_folds = int(cfg_get(config, 'validation.well_hash_folds', 5))
    split_map = exp039_cv_split_codes(frame, well_hash_folds=well_hash_folds)
    if not selected_modes:
        raise ValueError('No active LightGBM reproducibility modes configured')
    for _audit in audits:
        if _audit not in split_map:
            raise ValueError(f'unknown audit: {_audit}')
    for _mode_name in selected_modes:
        if _mode_name not in mode_map:
            raise ValueError(f'active mode is not defined under model.training.modes: {_mode_name}')
    run_options = {'audits': list(audits), 'selected_modes': selected_modes, 'well_hash_folds': well_hash_folds, 'fast': bool(cfg_get(config, 'audit.fast', False)), 'early_stopping_rounds': int(cfg_get(config, 'model.training.early_stopping_rounds', 250)), 'max_train_rows': cfg_get(config, 'model.training.max_train_rows'), 'save_models': bool(cfg_get(config, 'model.training.save_models', True)), 'save_predictions': bool(cfg_get(config, 'model.training.save_predictions', True))}
    print(json.dumps(run_options, indent=2, default=str))
    return audits, mode_map, run_options, selected_modes, split_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fit LightGBM folds
    """)
    return


@app.cell
def _(
    audits,
    display,
    feature_columns,
    frame,
    mode_map,
    paths,
    pd,
    run_options,
    selected_modes,
    split_map,
):
    metric_frames = []
    well_frames = []
    prediction_frames = []
    model_rows = []
    mode_summaries = []
    for _audit in audits:
        split_codes, split_labels = split_map[_audit]
        print(f'audit={_audit} splits={list(split_labels)}')
        for _mode_name in selected_modes:
            combined_mode_name = f'{_mode_name}__{_audit}'
            metrics_part, by_well_part, predictions_part, models_part, mode_summary = _fit_one_mode(mode_name=combined_mode_name, mode_config=mode_map[_mode_name], frame=frame, feature_columns=feature_columns, output_dir=paths.artifacts_dir, split_codes=split_codes, split_labels=split_labels, fast=run_options['fast'], early_stopping_rounds=run_options['early_stopping_rounds'], max_train_rows=run_options['max_train_rows'], save_models=run_options['save_models'])
            metrics_part.insert(0, 'audit', _audit)
            by_well_part.insert(0, 'audit', _audit)
            predictions_part.insert(0, 'audit', _audit)
            for model_row in models_part:
                model_row['audit'] = _audit
            mode_summary['audit'] = _audit
            mode_summary['base_mode'] = _mode_name
            metric_frames.append(metrics_part)
            well_frames.append(by_well_part)
            prediction_frames.append(predictions_part)
            model_rows.extend(models_part)
            mode_summaries.append(mode_summary)
    metrics = pd.concat(metric_frames, ignore_index=True)
    by_well = pd.concat(well_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = metrics[metrics['split'].astype(str).eq('pooled')].copy()
    display(pooled.sort_values('rmse_tvt'))
    return by_well, metrics, mode_summaries, model_rows, pooled, predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Save metrics and generated outputs
    """)
    return


@app.cell
def _(
    OUTPUT_PREFIX,
    audits,
    by_well,
    display,
    feature_columns,
    join_stats,
    json,
    metrics,
    mode_summaries,
    model_rows,
    paths,
    pd,
    pooled,
    predictions,
    run_options,
    selected_modes,
    time,
    train_started_at,
):
    metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    by_well_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_predictions.csv.gz"
    feature_schema_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    model_root = paths.artifacts_dir / f"{OUTPUT_PREFIX}_lgb_models"
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    manifest_path = model_root / "manifest.json"

    metrics.to_csv(metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    if run_options["save_predictions"]:
        predictions.to_csv(predictions_path, index=False, compression="gzip")
    pd.DataFrame({"feature": feature_columns}).to_csv(feature_schema_path, index=False)

    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "exp076_exp039_cv_reassessment",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "reference_branch": "exp039_ravaghi_single_lgbm_inference_submit",
        "mode": "exp073_full_replay_lgbm_retrained_on_exp039_cv_surface",
        "feature_source": join_stats["feature_source"],
        "cv_surface_source": join_stats["surface_source"],
        "join_stats": join_stats,
        "audits": list(audits),
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "experiment": "exp076_exp039_cv_reassessment",
        "status": "implemented_not_run" if metrics.empty else "train_completed",
        "mode": "exp073_full_replay_lgbm_retrained_on_exp039_cv_surface",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "reference_branch": "exp039_ravaghi_single_lgbm_inference_submit",
        "join_stats": join_stats,
        "audits": list(audits),
        "active_modes": selected_modes,
        "pooled_metrics": pooled.to_dict("records"),
        "artifacts": {
            "metrics": metrics_path.name,
            "by_well": by_well_path.name,
            "predictions": predictions_path.name if run_options["save_predictions"] else None,
            "feature_schema": feature_schema_path.name,
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - train_started_at, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    display(metrics.sort_values(["audit", "mode", "model", "split"]).head(20))
    print("Summary:", summary_path)
    print("Manifest:", manifest_path)
    return


if __name__ == "__main__":
    app.run()
