# %% [markdown]
# # exp253 prefix-verified bounded candidate controller — train audit
#
# Existing exp072 candidate paths are rebuilt from shortened legal prefixes. The
# controller then makes a bounded move from the frozen exp238 OOF base toward the
# prefix-verified candidate. No model is trained.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, and input helpers
# 3. Masked-prefix request and candidate replay helpers
# 4. Prefix scoring and bounded controller
# 5. Metrics, guards, and SHA helpers
# 6. Setup and input contract
# 7. Stage 0 / Stage 1 orchestration
# 8. Metrics and generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os

# Generated execution prelude; all scientific code below is copied verbatim.
os.environ["EXP253_ACTIVE_WELL_SHARD_INDEX"] = "0"
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "exp253_prefix_verified_bounded_candidate_controller"
PACKAGE_DIR = Path.cwd()
IS_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else PACKAGE_DIR / "artifacts"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 2. Runtime, configuration, and input helpers

# %%
def find_config() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml", Path("config.yaml")]
    candidates.extend(PACKAGE_DIR.parents[i] / "config.yaml" for i in range(min(4, len(PACKAGE_DIR.parents))))
    for path in candidates:
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


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_token(*parts: object, length: int = 20) -> str:
    value = "::".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def stable_fold(well: str, folds: int) -> int:
    return int(stable_token("fold", well, length=16), 16) % int(folds)


def stable_well_shard(well: str, shard_count: int) -> int:
    return int(stable_token(EXPERIMENT, "stage1_well_shard", well, length=16), 16) % int(shard_count)


def find_named(filename: str, *, local_candidates: list[Path] | None = None) -> Path:
    candidates = list(local_candidates or [])
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob(f"**/{filename}"))
    candidates.extend([PACKAGE_DIR / filename, WORK_DIR / filename])
    valid = [path for path in candidates if path.exists() and path.stat().st_size]
    if not valid:
        raise FileNotFoundError(filename)
    return max(valid, key=lambda path: path.stat().st_size)


def find_raw_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        PACKAGE_DIR / "data/raw",
        Path("data/raw"),
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/sample_submission.csv"))
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("ROGII raw data root")


def read_filtered_csv(path: Path, wells: set[str], usecols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in usecols if column not in header]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        part = chunk[chunk["well"].astype(str).isin(wells)].copy()
        if len(part):
            parts.append(part)
    if not parts:
        raise ValueError(f"No selected wells found in {path.name}")
    return pd.concat(parts, ignore_index=True)


def load_hidden_like() -> pd.DataFrame:
    filename = str(cfg("validation.hidden_like_assignment_file"))
    path = find_named(
        filename,
        local_candidates=[
            PACKAGE_DIR / "inputs" / filename,
            Path("experiments/exp115_hidden_like_spatial_holdout_from_ppt/artifacts") / filename,
        ],
    )
    return pd.read_csv(path, dtype={"well_id": str})


# %% [markdown]
# ## 3. Masked-prefix request and candidate replay helpers

# %%
CANDIDATES = [str(value) for value in cfg("model.candidates.allowed")]
DEFAULT_CANDIDATE = str(cfg("model.candidates.default"))
CUT_FRACTIONS = [float(value) for value in cfg("model.prefix.cut_fractions")]


def prefix_requests(well: str, horizontal: pd.DataFrame) -> list[dict[str, Any]]:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        return []
    first_hidden = int(np.flatnonzero(~np.isfinite(tvt_input))[0]) if (~np.isfinite(tvt_input)).any() else len(horizontal)
    known = known[known < first_hidden]
    if len(known) < int(cfg("model.prefix.min_known_rows")):
        return []
    requests: list[dict[str, Any]] = []
    for fraction in CUT_FRACTIONS:
        cut_position = int(round(len(known) * fraction))
        cut_position = max(int(cfg("model.prefix.min_prefix_rows_at_cut")), cut_position)
        cut_position = min(cut_position, len(known) - int(cfg("model.prefix.min_holdout_rows")))
        if cut_position <= 0 or cut_position >= len(known):
            continue
        cutoff_index = int(known[cut_position - 1])
        holdout_indices = known[cut_position:].astype(int)
        if len(holdout_indices) < int(cfg("model.prefix.min_holdout_rows")):
            continue
        request_id = stable_token(EXPERIMENT, well, f"{fraction:.2f}", cutoff_index)
        requests.append(
            {
                "request_id": request_id,
                "request_well": f"vp{request_id}",
                "source_well": well,
                "cut_fraction": fraction,
                "cutoff_index": cutoff_index,
                "holdout_indices": holdout_indices,
            }
        )
    return requests


def write_masked_request(
    request: dict[str, Any], horizontal: pd.DataFrame, typewell_path: Path, request_train_dir: Path
) -> tuple[Path, Path]:
    masked = horizontal.copy()
    cutoff = int(request["cutoff_index"])
    truth = pd.to_numeric(masked["TVT"], errors="coerce")
    masked["TVT_input"] = np.nan
    masked.loc[masked.index <= cutoff, "TVT_input"] = truth.loc[masked.index <= cutoff]
    # Feature generation never receives truth after the synthetic cut.
    masked.loc[masked.index > cutoff, "TVT"] = np.nan
    if masked.loc[masked.index > cutoff, "TVT_input"].notna().any():
        raise AssertionError("TVT_input mask failed")
    if masked.loc[masked.index > cutoff, "TVT"].notna().any():
        raise AssertionError("TVT target mask failed")
    request_train_dir.mkdir(parents=True, exist_ok=True)
    horizontal_path = request_train_dir / f"{request['request_well']}__horizontal_well.csv"
    typewell_out = request_train_dir / f"{request['request_well']}__typewell.csv"
    masked.to_csv(horizontal_path, index=False)
    shutil.copyfile(typewell_path, typewell_out)
    return horizontal_path, typewell_out


def install_source_well_exclusion(public: Any, mapping: dict[str, str]) -> None:
    for imputer in (public._FI, public._DI):
        original = imputer.impute

        def translated(xy: np.ndarray, self_wid: str | None = None, *, _original=original):
            return _original(xy, self_wid=mapping.get(str(self_wid), self_wid))

        imputer.impute = translated


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
    delta_map = {
        "beam_mean": "beam_mean_d",
        "beam_med": "beam_med_d",
        "sc_ens": "sc_ens_d",
        "hyb": "hyb_d",
        "tvt_dense": "tvt_dense_d",
    }
    for name, column in delta_map.items():
        if column in frame:
            out[name] = anchor + pd.to_numeric(frame[column], errors="coerce")
    return out


def replay_request(
    public: Any,
    request: dict[str, Any],
    horizontal: pd.DataFrame,
    typewell_path: Path,
    request_train_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    horizontal_path, typewell_out = write_masked_request(request, horizontal, typewell_path, request_train_dir)
    base = public.build_well(str(horizontal_path), str(typewell_out), True)
    if base is None or len(base) == 0:
        raise RuntimeError(f"candidate replay failed: {request['request_id']}")
    request_hw = pd.read_csv(horizontal_path)
    request_tw = pd.read_csv(typewell_out).sort_values("TVT")
    likelihood, indices, _ = public.lik_pf(
        request_hw,
        request_tw,
        seed_base=public.stable_seed(EXPERIMENT, request["source_well"], request["cut_fraction"]),
    )
    likelihood_frame: dict[str, Any] = {"id": [f"{request['request_well']}_{int(index)}" for index in indices]}
    for key, values in likelihood.items():
        column = "likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")
        likelihood_frame[column] = values.astype(np.float32)
    base = public.add_likpf_features(base, pd.DataFrame(likelihood_frame))
    candidates = materialize_candidates(base)
    holdout = set(int(value) for value in request["holdout_indices"])
    candidates = candidates[candidates["row_index"].astype(int).isin(holdout)].copy()
    truth = pd.to_numeric(horizontal.loc[candidates["row_index"].astype(int), "TVT"], errors="coerce").to_numpy(float)
    score_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        if candidate not in candidates:
            continue
        values = pd.to_numeric(candidates[candidate], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(truth)
        if not valid.any():
            continue
        score_rows.append(
            {
                "request_id": request["request_id"],
                "well": request["source_well"],
                "cut_fraction": request["cut_fraction"],
                "candidate": candidate,
                "holdout_rows": int(valid.sum()),
                "rmse": float(np.sqrt(np.mean(np.square(values[valid] - truth[valid])))),
            }
        )
    manifest = {
        "request_id": request["request_id"],
        "request_well": request["request_well"],
        "source_well": request["source_well"],
        "cut_fraction": request["cut_fraction"],
        "cutoff_index": request["cutoff_index"],
        "holdout_rows": len(request["holdout_indices"]),
        "scored_candidates": len(score_rows),
    }
    return score_rows, manifest


# %% [markdown]
# ## 4. Prefix scoring and bounded controller

# %%
def aggregate_prefix_scores(cut_scores: pd.DataFrame, selected_wells: list[str]) -> pd.DataFrame:
    std_weight = float(cfg("model.prefix.score_std_weight"))
    improvement = float(cfg("model.prefix.consistency_min_improvement"))
    rows: list[dict[str, Any]] = []
    for well in selected_wells:
        part = cut_scores[cut_scores["well"].eq(well)].copy()
        if part.empty:
            rows.append({"well": well, "status": "skip_no_scores"})
            continue
        cut_count = int(part["cut_fraction"].nunique())
        aggregate = (
            part.groupby("candidate", as_index=False)
            .agg(
                cuts=("cut_fraction", "nunique"),
                median_rmse=("rmse", "median"),
                std_rmse=("rmse", lambda values: float(np.nanstd(values.to_numpy(float)))),
            )
        )
        aggregate["std_rmse"] = aggregate["std_rmse"].fillna(0.0)
        aggregate["score"] = aggregate["median_rmse"] + std_weight * aggregate["std_rmse"]
        aggregate = aggregate.sort_values(["score", "candidate"]).reset_index(drop=True)
        if aggregate.empty or DEFAULT_CANDIDATE not in set(aggregate["candidate"]):
            rows.append({"well": well, "status": "skip_missing_default", "cuts": cut_count})
            continue
        best = aggregate.iloc[0]
        second_score = float(aggregate.iloc[1]["score"]) if len(aggregate) > 1 else float(best["score"])
        default_score = float(aggregate.loc[aggregate["candidate"].eq(DEFAULT_CANDIDATE), "score"].iloc[0])
        wide = part.pivot(index="cut_fraction", columns="candidate", values="rmse")
        local_best = wide.min(axis=1, skipna=True)
        comparable = pd.DataFrame({"local_best": local_best, "default": wide[DEFAULT_CANDIDATE]}).dropna()
        consistency = float((comparable["local_best"] <= comparable["default"] - improvement).mean()) if len(comparable) else 0.0
        rows.append(
            {
                "well": well,
                "status": "ok",
                "cuts": cut_count,
                "candidate_count": int(len(aggregate)),
                "best_name": str(best["candidate"]),
                "best_score": float(best["score"]),
                "second_score": second_score,
                "default_name": DEFAULT_CANDIDATE,
                "default_score": default_score,
                "gain": default_score - float(best["score"]),
                "rank_margin": second_score - float(best["score"]),
                "consistency": consistency,
            }
        )
    return pd.DataFrame(rows)


def controller_alpha(report: pd.Series, delta_rmse: float, delta_p95: float) -> float:
    if report.get("status") != "ok":
        return 0.0
    gain = float(report.get("gain", 0.0))
    best = float(report.get("best_score", np.inf))
    margin = float(report.get("rank_margin", 0.0))
    consistency = float(report.get("consistency", 0.0))
    if (
        not np.isfinite(best)
        or best > float(cfg("model.controller.max_best_score"))
        or gain < float(cfg("model.controller.min_gain"))
        or consistency < float(cfg("model.controller.min_consistency"))
    ):
        return 0.0
    alpha = float(cfg("model.controller.alpha_base"))
    alpha += float(cfg("model.controller.alpha_gain_scale")) * min(max(gain, 0.0), 5.0) / 5.0
    alpha += float(cfg("model.controller.alpha_margin_scale")) * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += float(cfg("model.controller.alpha_quality_bonus"))
    delta_soft = float(cfg("model.controller.delta_soft"))
    if np.isfinite(delta_rmse) and delta_rmse > delta_soft:
        alpha *= max(0.20, delta_soft / max(delta_rmse, 1e-6))
    if np.isfinite(delta_p95) and delta_p95 > float(cfg("model.controller.delta_p95_hard")):
        return 0.0
    return float(np.clip(alpha, 0.0, float(cfg("model.controller.alpha_cap"))))


def apply_controller(official: pd.DataFrame, well_report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = well_report.set_index("well", drop=False)
    output_parts: list[pd.DataFrame] = []
    move_rows: list[dict[str, Any]] = []
    for well, group in official.groupby("well", sort=True):
        group = group.sort_values("md_since").copy()
        report = reports.loc[well] if well in reports.index else pd.Series({"status": "skip_no_report"})
        best_name = str(report.get("best_name", ""))
        base = group["base_pred"].to_numpy(float)
        candidate = group[best_name].to_numpy(float) if best_name in group else np.full(len(group), np.nan)
        valid = np.isfinite(base) & np.isfinite(candidate)
        if not valid.all():
            alpha = 0.0
            diff = np.zeros(len(group), dtype=float)
            apply_status = "kept_base_nonfinite_candidate"
        else:
            diff = candidate - base
            delta_rmse = float(np.sqrt(np.mean(np.square(diff))))
            delta_p95 = float(np.quantile(np.abs(diff), 0.95))
            alpha = controller_alpha(report, delta_rmse, delta_p95)
            apply_status = "applied" if alpha > 0 else "kept_base"
        gain = max(0.0, float(report.get("gain", 0.0)))
        max_move = min(
            float(cfg("model.controller.clip_max")),
            float(cfg("model.controller.clip_base")) + float(cfg("model.controller.clip_gain")) * np.sqrt(gain + 1e-9),
        )
        ramp_denominator = max(
            float(cfg("model.controller.ramp_min_rows")),
            float(cfg("model.controller.ramp_fraction")) * max(1, len(group)),
        )
        ramp = 1.0 - np.exp(-np.arange(len(group), dtype=float) / ramp_denominator)
        move = np.clip(alpha * ramp * diff, -max_move, max_move)
        group["controller_pred"] = base + move
        group["selected_candidate"] = best_name
        group["controller_alpha"] = alpha
        group["controller_move"] = move
        output_parts.append(group)
        move_rows.append(
            {
                "well": well,
                "status": report.get("status", "missing"),
                "best_name": best_name,
                "gain": float(report.get("gain", np.nan)),
                "rank_margin": float(report.get("rank_margin", np.nan)),
                "consistency": float(report.get("consistency", np.nan)),
                "alpha": alpha,
                "max_move_clip": max_move,
                "mean_abs_move": float(np.mean(np.abs(move))) if len(move) else 0.0,
                "max_abs_move": float(np.max(np.abs(move))) if len(move) else 0.0,
                "apply_status": apply_status,
            }
        )
    return pd.concat(output_parts, ignore_index=True), pd.DataFrame(move_rows)


# %% [markdown]
# ## 5. Metrics, guards, and SHA helpers

# %%
def rmse(values: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(truth)
    return float(np.sqrt(np.mean(np.square(values[valid] - truth[valid])))) if valid.any() else float("nan")


def metric_row(surface: str, frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["true_tvt"].to_numpy(float)
    base_score = rmse(frame["base_pred"].to_numpy(float), truth)
    controller_score = rmse(frame["controller_pred"].to_numpy(float), truth)
    return {
        "surface": surface,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "base_rmse": base_score,
        "controller_rmse": controller_score,
        "delta_rmse": controller_score - base_score,
    }


def evaluate_controller(oof: pd.DataFrame, hidden_like: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = oof.merge(hidden_like, left_on="well", right_on="well_id", how="left")
    rows = [metric_row("overall", frame)]
    for bucket in cfg("validation.distance_buckets"):
        lower = float(bucket["min_md_since"])
        upper = bucket.get("max_md_since")
        mask = frame["md_since"].ge(lower)
        if upper is not None:
            mask &= frame["md_since"].lt(float(upper))
        rows.append(metric_row(str(bucket["name"]), frame[mask]))
    for column, surface in [
        ("verification_like_spatial_role", "hidden_like_spatial"),
        ("verification_like_typewell_purged_role", "hidden_like_typewell_purged"),
    ]:
        rows.append(metric_row(surface, frame[frame[column].eq("valid")]))
    for fold in sorted(frame["diagnostic_fold"].unique()):
        rows.append(metric_row(f"fold_{int(fold)}", frame[frame["diagnostic_fold"].eq(fold)]))
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=True):
        row = metric_row("well", group)
        row["well"] = well
        by_well_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(by_well_rows)


def adoption_guard(metrics: pd.DataFrame, by_well: pd.DataFrame) -> dict[str, Any]:
    delta = metrics.set_index("surface")["delta_rmse"].to_dict()
    improved_folds = sum(float(delta.get(f"fold_{fold}", np.inf)) < 0 for fold in range(int(cfg("validation.diagnostic_folds"))))
    worst = float(by_well["delta_rmse"].max()) if len(by_well) else float("inf")
    checks = {
        "global_improved": float(delta.get("overall", np.inf)) < 0,
        "near_nonworse": float(delta.get("000_050", np.inf)) <= 0,
        "longtail_nonworse": float(delta.get("1000_plus", np.inf)) <= 0,
        "hidden_like_spatial_nonworse": float(delta.get("hidden_like_spatial", np.inf)) <= 0,
        "hidden_like_typewell_purged_nonworse": float(delta.get("hidden_like_typewell_purged", np.inf)) <= 0,
        "fold_stability": improved_folds >= int(cfg("validation.adoption_guard.min_improved_folds")),
        "worst_well": worst <= float(cfg("validation.adoption_guard.max_worst_well_regression")),
    }
    required = {
        "global_improved": bool(cfg("validation.adoption_guard.require_global_improvement", True)),
        "near_nonworse": bool(cfg("validation.adoption_guard.require_near_nonworse", True)),
        "longtail_nonworse": bool(cfg("validation.adoption_guard.require_longtail_nonworse", True)),
        "hidden_like_spatial_nonworse": bool(cfg("validation.adoption_guard.require_hidden_like_spatial_nonworse", True)),
        "hidden_like_typewell_purged_nonworse": bool(
            cfg("validation.adoption_guard.require_hidden_like_typewell_purged_nonworse", True)
        ),
        "fold_stability": True,
        "worst_well": bool(cfg("validation.adoption_guard.require_worst_well", True)),
    }
    required_checks = [name for name, is_required in required.items() if is_required]
    return {
        "checks": checks,
        "required": required,
        "required_checks": required_checks,
        "improved_folds": improved_folds,
        "worst_well_regression": worst,
        "pass": all(checks[name] for name in required_checks),
    }


# %% [markdown]
# ## 6. Setup and input contract

# %%
if not IS_KAGGLE and os.environ.get("ROGII_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError("Kaggle Notebook execution is canonical; local execution requires ROGII_ALLOW_LOCAL=1")

started = time.time()
raw_root = find_raw_root()
raw_train = raw_root / "train"
raw_wells = sorted(path.name.split("__horizontal_well.csv")[0] for path in raw_train.glob("*__horizontal_well.csv"))
active_mode = str(cfg("audit.active_mode"))
shard_count = int(cfg("audit.stage1_shard.count", 1))
active_shard_index = int(
    os.environ.get(
        "EXP253_ACTIVE_WELL_SHARD_INDEX",
        str(cfg("audit.stage1_shard.active_index", 0)),
    )
)
if active_mode == "stage0_preview":
    selected_wells = raw_wells[: int(cfg("audit.stage0_max_wells"))]
elif active_mode == "stage1_full_audit" and bool(cfg("audit.stage1_enabled")):
    if shard_count < 1 or active_shard_index < 0 or active_shard_index >= shard_count:
        raise ValueError(f"invalid Stage 1 shard index={active_shard_index}, count={shard_count}")
    selected_wells = [
        well for well in raw_wells if stable_well_shard(well, shard_count) == active_shard_index
    ]
    if not selected_wells:
        raise RuntimeError(f"Stage 1 shard {active_shard_index}/{shard_count} selected no wells")
else:
    raise ValueError(f"unsupported or disabled active_mode={active_mode}")
selected_well_set = set(selected_wells)

candidate_cache_path = find_named(
    str(cfg("data.official_candidate_cache.filename")),
    local_candidates=[
        Path("experiments/exp072_exp063_full_replay_feature_cache/artifacts") / str(cfg("data.official_candidate_cache.filename"))
    ],
)
base_oof_path = find_named(str(cfg("data.base_oof.filename")))

# Validate mounted parent schemas before the expensive 96-request Stage 0 replay.
candidate_cache_header = set(pd.read_csv(candidate_cache_path, nrows=0).columns)
official_likpf_column = "likpf_mean" if "likpf_mean" in candidate_cache_header else "likpf_mean_d"
official_required = {
    "id", "well", "last_known_tvt", "target", "md_since", "pf_ancc", "pf_z",
    "beam_mean_d", "beam_med_d", "sc_ens_d", "hyb_d", official_likpf_column, "tvt_dense_d",
}
missing_official = sorted(official_required - candidate_cache_header)
if missing_official:
    raise ValueError(f"{candidate_cache_path.name} missing required columns: {missing_official}")
base_header = set(pd.read_csv(base_oof_path, nrows=0).columns)
base_required = {"id", "well", "last_known_tvt", "target", str(cfg("data.base_oof.prediction_column"))}
missing_base = sorted(base_required - base_header)
if missing_base:
    raise ValueError(f"{base_oof_path.name} missing required columns: {missing_base}")

print(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "route": cfg("experiment.route"),
            "active_mode": active_mode,
            "selected_wells": len(selected_wells),
            "all_stage1_wells": len(raw_wells),
            "well_shard": {
                "index": active_shard_index,
                "count": shard_count,
                "policy": cfg("audit.stage1_shard.policy"),
            },
            "cut_fractions": CUT_FRACTIONS,
            "candidates": CANDIDATES,
            "default_candidate": DEFAULT_CANDIDATE,
            "variant_count": len(cfg("model.active_variants")),
            "model_configs": cfg("model.model_configs"),
            "folds_trained": cfg("model.folds_trained"),
            "boosters": cfg("model.boosters"),
            "parent_control_retraining": cfg("model.parent_control_retraining"),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 7. Stage 0 / Stage 1 orchestration

# %%
sys.path.insert(0, str(PACKAGE_DIR))
import public_notebook_replay_audit as public  # noqa: E402

public.configure_public_runtime(
    data_dir=raw_root,
    output_dir=WORK_DIR / "exp072_runtime",
    n_jobs=1,
    pf_seeds=128,
    pf_particles=500,
    fast=False,
    use_gpu="cpu",
    n_train_wells=None,
)
public.init_imputers(raw_wells)

request_train_dir = WORK_DIR / "masked_prefix_raw" / "train"
all_requests: list[dict[str, Any]] = []
horizontal_by_well: dict[str, pd.DataFrame] = {}
for well in selected_wells:
    horizontal = pd.read_csv(raw_train / f"{well}__horizontal_well.csv")
    horizontal_by_well[well] = horizontal
    all_requests.extend(prefix_requests(well, horizontal))
request_mapping = {str(request["request_well"]): str(request["source_well"]) for request in all_requests}
install_source_well_exclusion(public, request_mapping)

cut_score_rows: list[dict[str, Any]] = []
manifest_rows: list[dict[str, Any]] = []
error_rows: list[dict[str, Any]] = []
for request_index, request in enumerate(all_requests, 1):
    well = str(request["source_well"])
    try:
        scores, manifest = replay_request(
            public,
            request,
            horizontal_by_well[well],
            raw_train / f"{well}__typewell.csv",
            request_train_dir,
        )
        cut_score_rows.extend(scores)
        manifest_rows.append(manifest)
    except Exception as error:
        error_rows.append({"request_id": request["request_id"], "well": well, "error": repr(error)})
    if request_index % 10 == 0 or request_index == len(all_requests):
        print(f"prefix replay {request_index}/{len(all_requests)} errors={len(error_rows)}", flush=True)

request_manifest = pd.DataFrame(manifest_rows)
cut_scores = pd.DataFrame(cut_score_rows)
well_report = aggregate_prefix_scores(cut_scores, selected_wells)

official_usecols = [
    "id", "well", "last_known_tvt", "target", "md_since", "pf_ancc", "pf_z",
    "beam_mean_d", "beam_med_d", "sc_ens_d", "hyb_d", official_likpf_column, "tvt_dense_d",
]
official_raw = read_filtered_csv(candidate_cache_path, selected_well_set, official_usecols)
official_candidates = materialize_candidates(official_raw)
official = official_raw[["id", "well", "last_known_tvt", "target", "md_since"]].copy()
for candidate in CANDIDATES:
    if candidate not in official_candidates:
        raise ValueError(f"official candidate cache missing materialized {candidate}")
    official[candidate] = official_candidates[candidate].to_numpy(float)
official["true_tvt"] = official["last_known_tvt"] + official["target"]

base_usecols = ["id", "well", "last_known_tvt", "target", str(cfg("data.base_oof.prediction_column"))]
base = read_filtered_csv(base_oof_path, selected_well_set, base_usecols)
base = base[["id", str(cfg("data.base_oof.prediction_column"))]].rename(columns={str(cfg("data.base_oof.prediction_column")): "base_pred"})
official = official.merge(base, on="id", how="left", validate="one_to_one")
if official["base_pred"].isna().any():
    raise AssertionError("exp238 base OOF join contains missing predictions")
official["diagnostic_fold"] = official["well"].astype(str).map(lambda well: stable_fold(well, int(cfg("validation.diagnostic_folds"))))

controller_oof, move_report = apply_controller(official, well_report)
hidden_like = load_hidden_like()
metrics, by_well = evaluate_controller(controller_oof, hidden_like)
guard = adoption_guard(metrics, by_well)

scored_ok = well_report["status"].eq("ok") if "status" in well_report else pd.Series(False, index=well_report.index)
scored_fraction = float(scored_ok.mean()) if len(scored_ok) else 0.0
three_cut_fraction = float(well_report.loc[scored_ok, "cuts"].eq(len(CUT_FRACTIONS)).mean()) if scored_ok.any() else 0.0
min_candidate_count = int(well_report.loc[scored_ok, "candidate_count"].min()) if scored_ok.any() else 0
nonfinite_predictions = int((~np.isfinite(controller_oof["controller_pred"].to_numpy(float))).sum())
max_alpha = float(move_report["alpha"].max()) if len(move_report) else 0.0
max_move = float(move_report["max_abs_move"].max()) if len(move_report) else 0.0
stage0_checks = {
    "request_errors_zero": len(error_rows) == 0,
    "scored_well_fraction": scored_fraction >= float(cfg("validation.stage0_contract.min_scored_well_fraction")),
    "three_cuts": three_cut_fraction == 1.0,
    "candidate_count": min_candidate_count >= int(cfg("validation.stage0_contract.min_candidates_per_well")),
    "nonfinite_zero": nonfinite_predictions == 0,
    "alpha_cap": max_alpha <= float(cfg("model.controller.alpha_cap")) + 1e-9,
    "move_cap": max_move <= float(cfg("model.controller.clip_max")) + 1e-9,
}
stage0_pass = all(stage0_checks.values())
is_partial_stage1 = active_mode == "stage1_full_audit" and shard_count > 1
adoption_supported = bool(
    active_mode == "stage1_full_audit" and not is_partial_stage1 and stage0_pass and guard["pass"]
)

# %% [markdown]
# ## 8. Metrics and generated artifacts

# %%
prefix = str(cfg("audit.output_prefix"))
paths = {
    "request_manifest": WORK_DIR / f"{prefix}_request_manifest.csv",
    "cut_scores": WORK_DIR / f"{prefix}_cut_scores.csv",
    "well_report": WORK_DIR / f"{prefix}_well_report.csv",
    "move_report": WORK_DIR / f"{prefix}_move_report.csv",
    "metrics": WORK_DIR / f"{prefix}_metrics.csv",
    "by_well": WORK_DIR / f"{prefix}_by_well.csv",
    "oof": WORK_DIR / f"{prefix}_oof.csv.gz",
    "summary": WORK_DIR / f"{prefix}_summary.json",
    "plot": WORK_DIR / f"{prefix}_distance_metrics.png",
}
request_manifest.to_csv(paths["request_manifest"], index=False)
cut_scores.to_csv(paths["cut_scores"], index=False)
well_report.to_csv(paths["well_report"], index=False)
move_report.to_csv(paths["move_report"], index=False)
metrics.to_csv(paths["metrics"], index=False)
by_well.to_csv(paths["by_well"], index=False)
controller_oof[["id", "well", "true_tvt", "base_pred", "controller_pred", "selected_candidate", "controller_alpha", "controller_move", "md_since", "diagnostic_fold"]].to_csv(paths["oof"], index=False, compression="gzip")

plot_frame = metrics[metrics["surface"].isin([item["name"] for item in cfg("validation.distance_buckets")])].copy()
ax = plot_frame.set_index("surface")[["base_rmse", "controller_rmse"]].plot(kind="bar", figsize=(11, 4), title="Prefix-verified bounded controller")
ax.set_ylabel("RMSE")
plt.tight_layout()
plt.savefig(paths["plot"], dpi=140)
plt.close()

summary = {
    "experiment": EXPERIMENT,
    "status": "stage0_complete" if active_mode == "stage0_preview" else "stage1_shard_complete",
    "active_mode": active_mode,
    "route": cfg("experiment.route"),
    "runtime_seconds": time.time() - started,
    "selected_wells": len(selected_wells),
    "all_stage1_wells": len(raw_wells),
    "well_shard": {
        "index": active_shard_index,
        "count": shard_count,
        "policy": cfg("audit.stage1_shard.policy"),
        "selected_wells": len(selected_wells),
    },
    "partial_stage1": is_partial_stage1,
    "aggregate_required": is_partial_stage1,
    "request_count": len(all_requests),
    "request_errors": error_rows,
    "scored_well_fraction": scored_fraction,
    "three_cut_fraction": three_cut_fraction,
    "min_candidate_count": min_candidate_count,
    "max_alpha": max_alpha,
    "max_abs_move": max_move,
    "stage0_checks": stage0_checks,
    "stage0_pass": stage0_pass,
    "adoption_guard": guard,
    "adoption_supported": adoption_supported,
    "inference_allowed": adoption_supported,
    "training_cost": {"variants": 1, "model_configs": 0, "folds": 0, "boosters": 0, "parent_control_retraining": False},
    "sha256": {
        "config": sha256_file(CONFIG_PATH),
        "official_candidate_cache_decompressed": sha256_file(candidate_cache_path, decompressed=candidate_cache_path.suffix == ".gz"),
        "base_oof_decompressed": sha256_file(base_oof_path, decompressed=base_oof_path.suffix == ".gz"),
        "request_manifest": sha256_file(paths["request_manifest"]),
        "cut_scores": sha256_file(paths["cut_scores"]),
        "well_report": sha256_file(paths["well_report"]),
        "controller_oof_decompressed": sha256_file(paths["oof"], decompressed=True),
    },
}
paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

print(metrics.to_string(index=False))
print(json.dumps(summary, indent=2, sort_keys=True))
