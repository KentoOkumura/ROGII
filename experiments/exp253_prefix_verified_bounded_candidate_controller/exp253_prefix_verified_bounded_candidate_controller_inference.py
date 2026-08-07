# %% [markdown]
# # exp253 prefix-verified bounded candidate controller — inference
#
# This notebook is deliberately guarded by the train-side Stage 1 decision. It
# rebuilds only existing exp072 candidates from masked current-test prefixes and
# applies the frozen balanced bounded controller to the exp238 base submission.

# %% [markdown]
# ## Contents
# 1. Imports and configuration
# 2. Input and train-guard checks
# 3. Masked-prefix candidate replay
# 4. Prefix score and bounded move
# 5. Current-test orchestration
# 6. Submission and audit artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "exp253_prefix_verified_bounded_candidate_controller"
PACKAGE_DIR = Path.cwd()
IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else PACKAGE_DIR / "artifacts"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 1. Imports and configuration

# %%
def find_config() -> Path:
    for path in [PACKAGE_DIR / "config.yaml", Path("config.yaml")]:
        if path.exists() and path.stat().st_size:
            return path
    raise FileNotFoundError("config.yaml")


CONFIG_PATH = find_config()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())


def cfg(key: str, default: Any = None) -> Any:
    value: Any = CONFIG
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def stable_token(*parts: object, length: int = 20) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_raw_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("data/raw"),
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/sample_submission.csv"))
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir() and (root / "sample_submission.csv").exists():
            return root
    raise FileNotFoundError("ROGII raw data root")


def find_train_summary() -> Path:
    filename = f"{EXPERIMENT}_summary.json"
    matches = list(Path("/kaggle/input").glob(f"**/{filename}")) if Path("/kaggle/input").exists() else []
    valid = [path for path in matches if path.stat().st_size]
    if not valid:
        raise FileNotFoundError(filename)
    return max(valid, key=lambda path: path.stat().st_mtime)


def find_base_submission(sample: pd.DataFrame) -> Path:
    matches = list(Path("/kaggle/input").glob("**/submission.csv")) if Path("/kaggle/input").exists() else []
    preferred = [path for path in matches if "exp238-nested-rank-slot-exp218-inference" in str(path)]
    for path in [*preferred, *matches]:
        try:
            frame = pd.read_csv(path, dtype={"id": str})
        except Exception:
            continue
        if list(frame.columns) == ["id", "tvt"] and frame["id"].astype(str).equals(sample["id"].astype(str)):
            return path
    raise FileNotFoundError("exp238 base submission.csv")


# %% [markdown]
# ## 2. Input and train-guard checks

# %%
if not IS_KAGGLE and os.environ.get("ROGII_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError("Kaggle Notebook execution is canonical; local execution requires ROGII_ALLOW_LOCAL=1")

started = time.time()
raw_root = find_raw_root()
sample = pd.read_csv(raw_root / "sample_submission.csv", dtype={"id": str})[["id"]]
train_summary_path = find_train_summary()
train_summary = json.loads(train_summary_path.read_text())
if train_summary.get("active_mode") != "stage1_full_audit":
    raise RuntimeError("Inference forbidden: train summary is not Stage 1")
if not bool(train_summary.get("adoption_supported")) or not bool(train_summary.get("inference_allowed")):
    raise RuntimeError("Inference forbidden: Stage 1 safety guard did not pass")

base_submission_path = find_base_submission(sample)
base_submission = pd.read_csv(base_submission_path, dtype={"id": str})[["id", "tvt"]]
if not np.isfinite(base_submission["tvt"].to_numpy(float)).all():
    raise AssertionError("non-finite exp238 base submission")

print(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "route": cfg("experiment.route"),
            "train_summary": str(train_summary_path),
            "base_submission": str(base_submission_path),
            "cut_fractions": cfg("model.prefix.cut_fractions"),
            "alpha_cap": cfg("model.controller.alpha_cap"),
            "clip_max": cfg("model.controller.clip_max"),
            "training_during_inference": False,
        },
        indent=2,
    )
)


# %% [markdown]
# ## 3. Masked-prefix candidate replay

# %%
CANDIDATES = [str(value) for value in cfg("model.candidates.allowed")]
DEFAULT_CANDIDATE = str(cfg("model.candidates.default"))
CUT_FRACTIONS = [float(value) for value in cfg("model.prefix.cut_fractions")]


def prefix_requests(well: str, horizontal: pd.DataFrame) -> list[dict[str, Any]]:
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(visible))
    if len(known) < int(cfg("model.prefix.min_known_rows")):
        return []
    requests: list[dict[str, Any]] = []
    for fraction in CUT_FRACTIONS:
        cut_position = int(round(len(known) * fraction))
        cut_position = max(int(cfg("model.prefix.min_prefix_rows_at_cut")), cut_position)
        cut_position = min(cut_position, len(known) - int(cfg("model.prefix.min_holdout_rows")))
        if cut_position <= 0 or cut_position >= len(known):
            continue
        cutoff = int(known[cut_position - 1])
        holdout = known[cut_position:].astype(int)
        request_id = stable_token(EXPERIMENT, "test", well, f"{fraction:.2f}", cutoff)
        requests.append(
            {
                "request_id": request_id,
                "request_well": f"vp{request_id}",
                "source_well": well,
                "cut_fraction": fraction,
                "cutoff_index": cutoff,
                "holdout_indices": holdout,
            }
        )
    return requests


def write_masked_request(request: dict[str, Any], horizontal: pd.DataFrame, typewell_path: Path, output_dir: Path) -> tuple[Path, Path]:
    masked = horizontal.copy()
    cutoff = int(request["cutoff_index"])
    original_visible = pd.to_numeric(masked["TVT_input"], errors="coerce")
    masked.loc[masked.index > cutoff, "TVT_input"] = np.nan
    if masked.loc[masked.index > cutoff, "TVT_input"].notna().any():
        raise AssertionError("test TVT_input mask failed")
    if not np.allclose(
        pd.to_numeric(masked.loc[masked.index <= cutoff, "TVT_input"], errors="coerce").to_numpy(float),
        original_visible.loc[masked.index <= cutoff].to_numpy(float),
        equal_nan=True,
    ):
        raise AssertionError("visible prefix changed before cutoff")
    output_dir.mkdir(parents=True, exist_ok=True)
    horizontal_path = output_dir / f"{request['request_well']}__horizontal_well.csv"
    typewell_out = output_dir / f"{request['request_well']}__typewell.csv"
    masked.to_csv(horizontal_path, index=False)
    shutil.copyfile(typewell_path, typewell_out)
    return horizontal_path, typewell_out


def add_likelihood(public: Any, frame: pd.DataFrame, horizontal_path: Path, typewell_path: Path, seed_parts: tuple[object, ...]) -> pd.DataFrame:
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path).sort_values("TVT")
    likelihood, indices, _ = public.lik_pf(horizontal, typewell, seed_base=public.stable_seed(*seed_parts))
    values: dict[str, Any] = {"id": [f"{horizontal_path.stem.replace('__horizontal_well', '')}_{int(index)}" for index in indices]}
    for key, array in likelihood.items():
        column = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
        values[column] = array.astype(np.float32)
    return public.add_likpf_features(frame, pd.DataFrame(values))


def materialize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["row_index"] = pd.to_numeric(frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False))
    anchor = pd.to_numeric(frame["last_known_tvt"], errors="coerce")
    out["last_anchor_tvt"] = anchor
    for name in ("pf_ancc", "pf_z"):
        if name in frame:
            out[name] = pd.to_numeric(frame[name], errors="coerce")
    if "likpf_mean" in frame:
        out["likpf_mean"] = pd.to_numeric(frame["likpf_mean"], errors="coerce")
    elif "likpf_mean_d" in frame:
        out["likpf_mean"] = anchor + pd.to_numeric(frame["likpf_mean_d"], errors="coerce")
    for name, column in {
        "beam_mean": "beam_mean_d", "beam_med": "beam_med_d", "sc_ens": "sc_ens_d",
        "hyb": "hyb_d", "tvt_dense": "tvt_dense_d",
    }.items():
        if column in frame:
            out[name] = anchor + pd.to_numeric(frame[column], errors="coerce")
    return out


def replay_cut(public: Any, request: dict[str, Any], horizontal: pd.DataFrame, typewell_path: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    horizontal_path, typewell_out = write_masked_request(request, horizontal, typewell_path, output_dir)
    base = public.build_well(str(horizontal_path), str(typewell_out), False)
    if base is None or len(base) == 0:
        raise RuntimeError(f"test candidate replay failed: {request['request_id']}")
    base = add_likelihood(public, base, horizontal_path, typewell_out, (EXPERIMENT, "test", request["source_well"], request["cut_fraction"]))
    candidates = materialize_candidates(base)
    holdout = set(int(value) for value in request["holdout_indices"])
    candidates = candidates[candidates["row_index"].astype(int).isin(holdout)].copy()
    truth = pd.to_numeric(horizontal.loc[candidates["row_index"].astype(int), "TVT_input"], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        if candidate not in candidates:
            continue
        values = pd.to_numeric(candidates[candidate], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(truth)
        if valid.any():
            rows.append(
                {
                    "request_id": request["request_id"], "well": request["source_well"],
                    "cut_fraction": request["cut_fraction"], "candidate": candidate,
                    "holdout_rows": int(valid.sum()),
                    "rmse": float(np.sqrt(np.mean(np.square(values[valid] - truth[valid])))),
                }
            )
    return rows, {
        "request_id": request["request_id"], "source_well": request["source_well"],
        "cut_fraction": request["cut_fraction"], "cutoff_index": request["cutoff_index"],
        "holdout_rows": len(request["holdout_indices"]), "scored_candidates": len(rows),
    }


def official_candidates(public: Any, well: str, horizontal_path: Path, typewell_path: Path) -> pd.DataFrame:
    base = public.build_well(str(horizontal_path), str(typewell_path), False)
    if base is None or len(base) == 0:
        raise RuntimeError(f"official candidate generation failed: {well}")
    base = add_likelihood(public, base, horizontal_path, typewell_path, (EXPERIMENT, "test_official", well))
    return materialize_candidates(base)


# %% [markdown]
# ## 4. Prefix score and bounded move

# %%
def aggregate_scores(cut_scores: pd.DataFrame, well: str) -> dict[str, Any]:
    part = cut_scores[cut_scores["well"].eq(well)]
    cut_count = int(part["cut_fraction"].nunique())
    aggregate = part.groupby("candidate", as_index=False).agg(
        cuts=("cut_fraction", "nunique"),
        median_rmse=("rmse", "median"),
        std_rmse=("rmse", lambda values: float(np.nanstd(values.to_numpy(float)))),
    )
    aggregate["std_rmse"] = aggregate["std_rmse"].fillna(0.0)
    aggregate["score"] = aggregate["median_rmse"] + float(cfg("model.prefix.score_std_weight")) * aggregate["std_rmse"]
    aggregate = aggregate.sort_values(["score", "candidate"]).reset_index(drop=True)
    if aggregate.empty or DEFAULT_CANDIDATE not in set(aggregate["candidate"]):
        return {"well": well, "status": "skip_missing_default"}
    best = aggregate.iloc[0]
    second = float(aggregate.iloc[1]["score"]) if len(aggregate) > 1 else float(best["score"])
    default = float(aggregate.loc[aggregate["candidate"].eq(DEFAULT_CANDIDATE), "score"].iloc[0])
    wide = part.pivot(index="cut_fraction", columns="candidate", values="rmse")
    local_best = wide.min(axis=1, skipna=True)
    comparable = pd.DataFrame({"local_best": local_best, "default": wide[DEFAULT_CANDIDATE]}).dropna()
    consistency = float(
        (comparable["local_best"] <= comparable["default"] - float(cfg("model.prefix.consistency_min_improvement"))).mean()
    ) if len(comparable) else 0.0
    return {
        "well": well, "status": "ok", "cuts": cut_count, "candidate_count": int(len(aggregate)),
        "best_name": str(best["candidate"]), "best_score": float(best["score"]), "second_score": second,
        "default_name": DEFAULT_CANDIDATE, "default_score": default, "gain": default - float(best["score"]),
        "rank_margin": second - float(best["score"]), "consistency": consistency,
    }


def alpha_for(report: dict[str, Any], delta_rmse: float, delta_p95: float) -> float:
    if report.get("status") != "ok":
        return 0.0
    gain, best = float(report["gain"]), float(report["best_score"])
    margin, consistency = float(report["rank_margin"]), float(report["consistency"])
    if best > float(cfg("model.controller.max_best_score")) or gain < float(cfg("model.controller.min_gain")) or consistency < float(cfg("model.controller.min_consistency")):
        return 0.0
    alpha = float(cfg("model.controller.alpha_base"))
    alpha += float(cfg("model.controller.alpha_gain_scale")) * min(max(gain, 0.0), 5.0) / 5.0
    alpha += float(cfg("model.controller.alpha_margin_scale")) * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += float(cfg("model.controller.alpha_quality_bonus"))
    delta_soft = float(cfg("model.controller.delta_soft"))
    if delta_rmse > delta_soft:
        alpha *= max(0.20, delta_soft / max(delta_rmse, 1e-6))
    if delta_p95 > float(cfg("model.controller.delta_p95_hard")):
        return 0.0
    return float(np.clip(alpha, 0.0, float(cfg("model.controller.alpha_cap"))))


# %% [markdown]
# ## 5. Current-test orchestration

# %%
sys.path.insert(0, str(PACKAGE_DIR))
import public_notebook_replay_audit as public  # noqa: E402

train_wells = sorted(path.name.split("__horizontal_well.csv")[0] for path in (raw_root / "train").glob("*__horizontal_well.csv"))
test_wells = sorted(path.name.split("__horizontal_well.csv")[0] for path in (raw_root / "test").glob("*__horizontal_well.csv"))
public.configure_public_runtime(data_dir=raw_root, output_dir=WORK_DIR / "exp072_runtime", n_jobs=1, pf_seeds=128, pf_particles=500, fast=False, use_gpu="cpu", n_train_wells=None)
public.init_imputers(train_wells)

base_with_keys = base_submission.copy()
parts = base_with_keys["id"].str.rsplit("_", n=1, expand=True)
base_with_keys["well"] = parts[0]
base_with_keys["row_index"] = parts[1].astype(int)

all_cut_scores: list[dict[str, Any]] = []
all_manifests: list[dict[str, Any]] = []
reports: list[dict[str, Any]] = []
move_reports: list[dict[str, Any]] = []
prediction_parts: list[pd.DataFrame] = []
request_dir = WORK_DIR / "masked_prefix_test"
for index, well in enumerate(test_wells, 1):
    horizontal_path = raw_root / "test" / f"{well}__horizontal_well.csv"
    typewell_path = raw_root / "test" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    requests = prefix_requests(well, horizontal)
    well_cut_rows: list[dict[str, Any]] = []
    for request in requests:
        rows, manifest = replay_cut(public, request, horizontal, typewell_path, request_dir)
        well_cut_rows.extend(rows)
        all_cut_scores.extend(rows)
        all_manifests.append(manifest)
    report = aggregate_scores(pd.DataFrame(well_cut_rows), well) if well_cut_rows else {"well": well, "status": "skip_no_scores"}
    reports.append(report)
    candidates = official_candidates(public, well, horizontal_path, typewell_path)
    group = base_with_keys[base_with_keys["well"].eq(well)].sort_values("row_index").copy()
    best_name = str(report.get("best_name", ""))
    candidate_map = candidates.set_index(candidates["row_index"].astype(int))
    candidate = pd.to_numeric(candidate_map.reindex(group["row_index"])[best_name], errors="coerce").to_numpy(float) if best_name in candidate_map else np.full(len(group), np.nan)
    base = group["tvt"].to_numpy(float)
    if np.isfinite(candidate).all():
        diff = candidate - base
        delta_rmse = float(np.sqrt(np.mean(np.square(diff))))
        delta_p95 = float(np.quantile(np.abs(diff), 0.95))
        alpha = alpha_for(report, delta_rmse, delta_p95)
    else:
        diff = np.zeros(len(group), dtype=float)
        delta_rmse = delta_p95 = float("nan")
        alpha = 0.0
    gain = max(0.0, float(report.get("gain", 0.0)))
    clip = min(float(cfg("model.controller.clip_max")), float(cfg("model.controller.clip_base")) + float(cfg("model.controller.clip_gain")) * np.sqrt(gain + 1e-9))
    denominator = max(float(cfg("model.controller.ramp_min_rows")), float(cfg("model.controller.ramp_fraction")) * max(1, len(group)))
    ramp = 1.0 - np.exp(-np.arange(len(group), dtype=float) / denominator)
    move = np.clip(alpha * ramp * diff, -clip, clip)
    group["tvt"] = base + move
    prediction_parts.append(group[["id", "tvt"]])
    move_reports.append({"well": well, "best_name": best_name, "alpha": alpha, "gain": report.get("gain"), "consistency": report.get("consistency"), "delta_rmse_vs_base": delta_rmse, "delta_p95_vs_base": delta_p95, "max_move_clip": clip, "mean_abs_move": float(np.mean(np.abs(move))), "max_abs_move": float(np.max(np.abs(move)))})
    print(f"test well {index}/{len(test_wells)} {well} best={best_name} alpha={alpha:.4f}", flush=True)


# %% [markdown]
# ## 6. Submission and audit artifacts

# %%
submission = pd.concat(prediction_parts, ignore_index=True).set_index("id").reindex(sample["id"]).reset_index()
submission.columns = ["id", "tvt"]
if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise AssertionError("submission id contract failed")
if not np.isfinite(submission["tvt"].to_numpy(float)).all():
    raise AssertionError("submission contains non-finite predictions")

submission_path = WORK_DIR / "submission.csv"
cut_scores_path = WORK_DIR / f"{EXPERIMENT}_current_test_cut_scores.csv"
report_path = WORK_DIR / f"{EXPERIMENT}_current_test_well_report.csv"
move_path = WORK_DIR / f"{EXPERIMENT}_current_test_move_report.csv"
manifest_path = WORK_DIR / f"{EXPERIMENT}_current_test_request_manifest.csv"
summary_path = WORK_DIR / f"{EXPERIMENT}_current_test_summary.json"
submission.to_csv(submission_path, index=False)
pd.DataFrame(all_cut_scores).to_csv(cut_scores_path, index=False)
pd.DataFrame(reports).to_csv(report_path, index=False)
pd.DataFrame(move_reports).to_csv(move_path, index=False)
pd.DataFrame(all_manifests).to_csv(manifest_path, index=False)

summary = {
    "experiment": EXPERIMENT,
    "status": "current_test_inference_complete",
    "runtime_seconds": time.time() - started,
    "wells": len(test_wells),
    "rows": len(submission),
    "training_during_inference": False,
    "contact_reconstruction": False,
    "max_alpha": float(pd.DataFrame(move_reports)["alpha"].max()) if move_reports else 0.0,
    "max_abs_move": float(pd.DataFrame(move_reports)["max_abs_move"].max()) if move_reports else 0.0,
    "sha256": {
        "train_summary": sha256_file(train_summary_path),
        "base_submission": sha256_file(base_submission_path),
        "cut_scores": sha256_file(cut_scores_path),
        "well_report": sha256_file(report_path),
        "submission": sha256_file(submission_path),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(submission.head())
print(json.dumps(summary, indent=2, sort_keys=True))
