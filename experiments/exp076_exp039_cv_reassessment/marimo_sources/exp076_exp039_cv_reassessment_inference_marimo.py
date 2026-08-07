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
    # exp076_exp039_cv_reassessment inference

    Inference-side PF/Beam reproducibility audit. The notebook keeps PF/Beam feature generation helpers in `public_notebook_replay_audit.py`, but performs saved-booster inference, submission assembly, and SHA recording in visible cells.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contents

    1. Setup and configuration
    2. Source model check
    3. Test feature regeneration
    4. Saved booster prediction
    5. Submission and metrics
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

    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    from settings import ExperimentPaths, get_nested, load_config
    from exp073_exp039_cv_reassessment import (
        OUTPUT_PREFIX,
        find_model_manifest,
        generate_exp063_tracker_test_frame,
        load_exp063_tracker_test_frame,
        prediction_sha256,
        sha256_file,
    )


    def cfg_get(config, dotted_key, default=None):
        value = get_nested(config, dotted_key)
        return default if value is None else value

    return (
        ExperimentPaths,
        OUTPUT_PREFIX,
        Path,
        cfg_get,
        find_model_manifest,
        generate_exp063_tracker_test_frame,
        json,
        lgb,
        load_config,
        load_exp063_tracker_test_frame,
        np,
        pd,
        prediction_sha256,
        sha256_file,
        time,
    )


@app.cell
def _(ExperimentPaths, OUTPUT_PREFIX, cfg_get, load_config, time):
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    inference_started_at = time.time()

    print("Experiment:", config["experiment"]["name"])
    print("Route:", config["experiment"]["route"])
    print("Inference mode:", cfg_get(config, "inference.mode"))
    print("Train kernel sources:", cfg_get(config, "runtime.kaggle.inference_kernel_sources"))
    print("Raw data:", paths.raw_data_dir)
    print("Sample submission:", paths.sample_submission_path)
    print("Output prefix:", cfg_get(config, "audit.output_prefix", OUTPUT_PREFIX))
    return config, inference_started_at, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Source model check
    """)
    return


@app.cell
def _(Path, cfg_get, config, display, find_model_manifest, json, pd):
    manifest_path = find_model_manifest(cfg_get(config, "inference.model_manifest_path"))
    model_root = manifest_path.parent
    manifest = json.loads(Path(manifest_path).read_text())

    mode_name = cfg_get(config, "inference.selected_mode", "gpu_repro_guard_dp_threads8__leave_one_original_fold_out")
    model_name = cfg_get(config, "inference.selected_model", "lgb_mean")

    feature_columns = manifest.get("feature_source", {}).get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"{manifest_path} does not contain feature_source.feature_columns")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(f"No saved models for mode={mode_name} model={model_name}")

    print("Model manifest:", manifest_path)
    print("Selected mode:", mode_name)
    print("Selected model:", model_name)
    print("Model count:", len(model_rows))
    print("Feature count:", len(feature_columns))
    display(pd.DataFrame(model_rows).head())
    return (
        feature_columns,
        manifest_path,
        mode_name,
        model_name,
        model_root,
        model_rows,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Test feature regeneration
    """)
    return


@app.cell
def _(
    cfg_get,
    config,
    display,
    feature_columns,
    generate_exp063_tracker_test_frame,
    json,
    load_exp063_tracker_test_frame,
    paths,
):
    fg = cfg_get(config, "inference.feature_generation", {}) or {}
    regenerate_test_features = bool(cfg_get(config, "inference.regenerate_test_features", True))

    if regenerate_test_features:
        test_frame, test_meta = generate_exp063_tracker_test_frame(
            data_dir=paths.raw_data_dir,
            output_dir=paths.artifacts_dir,
            n_jobs=fg.get("n_jobs"),
            pf_seeds=fg.get("pf_seeds"),
            pf_particles=fg.get("pf_particles"),
            fast=bool(fg.get("fast", False)),
            use_gpu=str(fg.get("use_gpu", "auto")),
        )
    else:
        test_frame, test_meta = load_exp063_tracker_test_frame(
            cfg_get(config, "data.exp063_tracker_test_feature_path")
        )

    missing = sorted(set(feature_columns) - set(test_frame.columns))
    if missing:
        raise ValueError(f"test tracker frame is missing model features: {missing[:20]}")

    print(json.dumps(test_meta, indent=2, default=str))
    print("Test rows:", len(test_frame))
    print("Test wells:", test_frame["well"].nunique())
    display(test_frame[["id", "well", "last_known_tvt", *feature_columns[:5]]].head())
    return test_frame, test_meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Saved booster prediction
    """)
    return


@app.cell
def _(
    OUTPUT_PREFIX,
    display,
    feature_columns,
    lgb,
    mode_name,
    model_name,
    model_root,
    model_rows,
    np,
    paths,
    pd,
    test_frame,
):
    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows = []

    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "mode": item.get("mode"),
                "model": item.get("model"),
                "split": item.get("split"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
                "rows": int(len(pred)),
            }
        )

    base = test_frame["last_known_tvt"].to_numpy(np.float32)
    pred_tvt = (base + pred_delta).astype(np.float32)
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "mode": mode_name,
            "model": model_name,
            "last_known_tvt": base,
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
        }
    )
    predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression="gzip")

    print("Predictions:", predictions_path)
    display(predictions.head())
    display(pd.DataFrame(loaded_rows))
    return loaded_rows, pred_delta, predictions, predictions_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Submission and metrics
    """)
    return


@app.cell
def _(
    OUTPUT_PREFIX,
    cfg_get,
    config,
    display,
    inference_started_at,
    json,
    loaded_rows,
    manifest_path,
    mode_name,
    model_name,
    model_rows,
    paths,
    pd,
    pred_delta,
    prediction_sha256,
    predictions,
    predictions_path,
    sha256_file,
    test_frame,
    test_meta,
    time,
):
    submission_path = paths.submission_path
    sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
    submission_target_column = cfg_get(config, "data.submission_target_column", "tvt")
    target_column = submission_target_column if submission_target_column in sample.columns else str(sample.columns[1])

    pred_map = dict(zip(predictions["id"].astype(str), predictions["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()
    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    submission_sha = sha256_file(submission_path)
    prediction_sha = prediction_sha256(
        predictions["id"],
        pred_delta,
        label=f"{mode_name}/{model_name}/test",
    )
    metrics = {
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "prediction_sha256": prediction_sha,
        "submission_sha256": submission_sha,
    }
    metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv"
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_inference_summary.json"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    summary = {
        "experiment": "exp076_exp039_cv_reassessment",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_with_exp073_raw_test_pfbeam_regeneration",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "selected": {
            "mode": mode_name,
            "model": model_name,
            "model_count": int(len(model_rows)),
        },
        "metrics": metrics,
        "loaded_models": loaded_rows,
        "artifacts": {
            "predictions": predictions_path.name,
            "metrics": metrics_path.name,
            "summary": summary_path.name,
            "submission": str(submission_path),
        },
        "elapsed_seconds": round(time.time() - inference_started_at, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    display(pd.DataFrame([metrics]))
    display(sample.head())
    print("Submission:", submission_path)
    print("Summary:", summary_path)
    return


if __name__ == "__main__":
    app.run()
