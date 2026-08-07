# %% [markdown]
# # exp288_known_tvt_typewell_horizontal_gr_visualization train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and interpolation helpers
# 4. Plot and index helpers
# 5. Setup and input checks
# 6. Generate one comparison PNG per well
# 7. Summary and generated artifacts

# %%
from __future__ import annotations

import hashlib
import html as html_lib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp288_known_tvt_typewell_horizontal_gr_visualization"
OUTPUT_PREFIX = "exp288_known_tvt_typewell_horizontal_gr_visualization"
COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_project_root() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return cwd


def find_experiment_config(root: Path) -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / EXPERIMENT_NAME / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config.yaml for {EXPERIMENT_NAME}")


def load_config() -> dict[str, Any]:
    root = find_project_root()
    project = read_yaml(root / "project.yml")
    experiment = read_yaml(find_experiment_config(root))
    defaults = {
        "data": {"train_dir": get_nested(project, "data.train_dir", "data/raw/train")},
        "runtime": {"kaggle": get_nested(project, "runtime.kaggle", {})},
    }
    return deep_merge(defaults, experiment)


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def contains_horizontal_files(path: Path) -> bool:
    return path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    root = find_project_root()
    local_candidates = [configured, root / configured]
    for candidate in local_candidates:
        if contains_horizontal_files(candidate):
            return candidate

    kaggle_candidates = [
        KAGGLE_INPUT_ROOT / "competitions" / COMPETITION_SLUG / "train",
        KAGGLE_INPUT_ROOT / COMPETITION_SLUG / "train",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        kaggle_candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("*/train")))
        kaggle_candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("*/*/train")))
    for candidate in kaggle_candidates:
        if contains_horizontal_files(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a train directory containing horizontal well CSVs from {configured}"
    )


def resolve_output_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    if is_kaggle_runtime():
        artifact_dir = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        artifact_dir = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    figure_subdir = str(
        get_nested(config, "audit.outputs.figure_subdir", "reference_vs_horizontal_gr")
    )
    figure_dir = artifact_dir / figure_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, figure_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(float(value)) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %% [markdown]
# ## 3. Input and interpolation helpers

# %%
class WellSkipError(ValueError):
    """Expected non-fatal reason why a well cannot produce a comparison figure."""


def well_from_horizontal_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def typewell_path_for(horizontal_path: Path) -> Path:
    well = well_from_horizontal_path(horizontal_path)
    return horizontal_path.with_name(f"{well}__typewell.csv")


def require_columns(path: Path, required: list[str]) -> None:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")


def read_well_inputs(
    horizontal_path: Path,
    typewell_path: Path,
    horizontal_columns: list[str],
    typewell_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not typewell_path.exists():
        raise WellSkipError("missing_typewell_file")
    require_columns(horizontal_path, horizontal_columns)
    require_columns(typewell_path, typewell_columns)
    horizontal = pd.read_csv(horizontal_path, usecols=horizontal_columns)
    typewell = pd.read_csv(typewell_path, usecols=typewell_columns)
    return horizontal, typewell


def prepare_typewell_curve(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna(subset=["TVT", "GR"])
    if clean.empty:
        raise WellSkipError("no_finite_typewell_tvt_gr_pairs")
    clean = clean.groupby("TVT", as_index=False, sort=True)["GR"].median()
    if len(clean) < 2:
        raise WellSkipError("fewer_than_two_unique_typewell_tvt_points")
    tvt = clean["TVT"].to_numpy(np.float64)
    gr = clean["GR"].to_numpy(np.float64)
    if not np.all(np.diff(tvt) > 0):
        raise ValueError("Type Well TVT must be strictly increasing after duplicate aggregation")
    return tvt, gr


def build_reference_frame(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    true_tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    type_tvt, type_gr = prepare_typewell_curve(typewell)

    known_mask = np.isfinite(tvt_input)
    if not known_mask.any():
        raise WellSkipError("no_finite_tvt_input_rows")
    target_mask = ~known_mask
    alignment_tvt = np.where(known_mask, tvt_input, true_tvt)
    finite_plot_md = np.isfinite(md)
    if not finite_plot_md.any():
        raise WellSkipError("no_finite_md_rows")

    reference_gr = np.full(len(horizontal), np.nan, dtype=np.float64)
    inside_typewell = (
        np.isfinite(alignment_tvt)
        & (alignment_tvt >= type_tvt[0])
        & (alignment_tvt <= type_tvt[-1])
    )
    if inside_typewell.any():
        reference_gr[inside_typewell] = np.interp(
            alignment_tvt[inside_typewell], type_tvt, type_gr
        )
    if not np.isfinite(reference_gr).any():
        raise WellSkipError("alignment_tvt_rows_outside_typewell_tvt_range")

    frame = pd.DataFrame(
        {
            "MD": md,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
            "alignment_tvt": alignment_tvt,
            "is_prediction_target": target_mask,
            "reference_gr": reference_gr,
            "horizontal_gr": horizontal_gr,
        }
    )
    finite_reference = np.isfinite(reference_gr)
    paired_mask = finite_reference & np.isfinite(horizontal_gr)
    metadata = {
        "horizontal_rows": int(len(horizontal)),
        "known_tvt_rows": int(known_mask.sum()),
        "prediction_target_rows": int(target_mask.sum()),
        "reference_rows": int(finite_reference.sum()),
        "known_reference_rows": int((finite_reference & known_mask).sum()),
        "prediction_target_reference_rows": int((finite_reference & target_mask).sum()),
        "paired_gr_rows": int(paired_mask.sum()),
        "known_paired_gr_rows": int((paired_mask & known_mask).sum()),
        "prediction_target_paired_gr_rows": int((paired_mask & target_mask).sum()),
        "typewell_unique_points": int(len(type_tvt)),
        "typewell_tvt_min": float(type_tvt[0]),
        "typewell_tvt_max": float(type_tvt[-1]),
        "plot_md_min": float(np.nanmin(md[finite_plot_md])),
        "plot_md_max": float(np.nanmax(md[finite_plot_md])),
        "uses_true_tvt_for_prediction_target_eda": True,
    }
    return frame, metadata


# %% [markdown]
# ## 4. Plot and index helpers

# %%
def shared_gr_limits(frame: pd.DataFrame, padding_fraction: float) -> tuple[float, float]:
    values = np.concatenate(
        [
            frame["reference_gr"].to_numpy(np.float64),
            frame["horizontal_gr"].to_numpy(np.float64),
        ]
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise WellSkipError("no_finite_reference_or_horizontal_gr")
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = max(span * float(padding_fraction), 1.0)
    if span == 0:
        return lower - padding, upper + padding
    return lower - padding, upper + padding


def prediction_target_md_spans(frame: pd.DataFrame) -> list[tuple[float, float]]:
    md = frame["MD"].to_numpy(np.float64)
    target = frame["is_prediction_target"].to_numpy(bool) & np.isfinite(md)
    positions = np.flatnonzero(target)
    if len(positions) == 0:
        return []
    runs = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    finite_md = md[np.isfinite(md)]
    md_step = float(np.nanmedian(np.abs(np.diff(finite_md)))) if len(finite_md) > 1 else 1.0
    if not np.isfinite(md_step) or md_step <= 0:
        md_step = 1.0
    spans: list[tuple[float, float]] = []
    for run in runs:
        start = float(np.nanmin(md[run]))
        end = float(np.nanmax(md[run]))
        if start == end:
            start -= md_step / 2.0
            end += md_step / 2.0
        spans.append((start, end))
    return spans


def plot_well_comparison(
    well: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    plot_config: dict[str, Any],
    output_path: Path,
) -> None:
    figsize = tuple(float(value) for value in plot_config.get("figsize", [14.0, 7.0]))
    dpi = int(plot_config.get("dpi", 120))
    line_width = float(plot_config.get("line_width", 0.9))
    grid_alpha = float(plot_config.get("grid_alpha", 0.25))
    gr_padding_fraction = float(plot_config.get("gr_padding_fraction", 0.05))
    y_min, y_max = shared_gr_limits(frame, gr_padding_fraction)
    x_min = float(metadata["plot_md_min"])
    x_max = float(metadata["plot_md_max"])
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    figure, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    target_spans = prediction_target_md_spans(frame)
    for axis in axes:
        for span_index, (span_start, span_end) in enumerate(target_spans):
            axis.axvspan(
                span_start,
                span_end,
                color=str(plot_config.get("prediction_target_color", "#ffbf00")),
                alpha=float(plot_config.get("prediction_target_alpha", 0.12)),
                label="Prediction target (true TVT used for train EDA)"
                if span_index == 0
                else None,
                zorder=0,
            )
    axes[0].plot(
        frame["MD"],
        frame["reference_gr"],
        color=str(plot_config.get("reference_color", "#d62728")),
        linewidth=line_width,
        label="Type Well reference GR",
        zorder=2,
    )
    axes[1].plot(
        frame["MD"],
        frame["horizontal_gr"],
        color=str(plot_config.get("horizontal_color", "#1f77b4")),
        linewidth=line_width,
        label="Horizontal GR",
        zorder=2,
    )
    axes[0].set_title(
        "Type Well reference GR: TVT_input on known rows; true TVT on target rows (train EDA)",
        loc="left",
        fontsize=10,
    )
    axes[1].set_title("Horizontal GR (observed, no fill)", loc="left", fontsize=10)
    for axis in axes:
        axis.set_ylabel("GR")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, alpha=grid_alpha)
        axis.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("Horizontal well MD [ft]")
    figure.suptitle(
        f"well={well} | known rows={metadata['known_tvt_rows']:,} "
        f"(paired={metadata['known_paired_gr_rows']:,}) | target rows={metadata['prediction_target_rows']:,} "
        f"(paired={metadata['prediction_target_paired_gr_rows']:,})",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def write_html_index(manifest: pd.DataFrame, index_path: Path, artifact_dir: Path) -> None:
    rows = [
        "<html><head><meta charset='utf-8'>",
        "<title>Known-TVT Type Well reference GR vs horizontal GR</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} "
        ".card{border:1px solid #ccc;padding:12px;margin:0 0 20px 0;} "
        ".meta{font-size:13px;line-height:1.5;margin-bottom:8px;} "
        "img{width:100%;height:auto;border:1px solid #ddd;} "
        "table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 6px;}"
        "</style></head><body>",
        "<h1>Known-TVT Type Well reference GR vs horizontal GR</h1>",
        "<p>Top: Type Well GR interpolated at TVT_input on known rows and train true TVT "
        "on prediction-target rows. Bottom: observed horizontal GR over the full well. "
        "Prediction-target intervals are shaded. Both panels share MD and GR display ranges. "
        "No smoothing, calibration, offset search, or GR filling is applied.</p>",
    ]
    saved = manifest.loc[manifest["status"] == "saved"].copy()
    rows.append(f"<p>Saved figures: <b>{len(saved):,}</b></p>")
    for _, row in saved.iterrows():
        relative = Path(str(row["png_path"]))
        try:
            relative = relative.relative_to(artifact_dir)
        except ValueError:
            pass
        rows.extend(
            [
                "<div class='card'>",
                "<div class='meta'>"
                f"<b>{html_lib.escape(str(row['well']))}</b> | "
                f"known={int(row['known_tvt_rows']):,} | "
                f"target={int(row['prediction_target_rows']):,} | "
                f"known paired={int(row['known_paired_gr_rows']):,} | "
                f"target paired={int(row['prediction_target_paired_gr_rows']):,}</div>",
                f"<img loading='lazy' src='{html_lib.escape(relative.as_posix())}' "
                f"alt='{html_lib.escape(str(row['well']))} GR comparison'>",
                "</div>",
            ]
        )
    skipped = manifest.loc[manifest["status"] != "saved", ["well", "reason"]]
    if not skipped.empty:
        rows.append("<h2>Skipped wells</h2><table><tr><th>well</th><th>reason</th></tr>")
        for _, row in skipped.iterrows():
            rows.append(
                f"<tr><td>{html_lib.escape(str(row['well']))}</td>"
                f"<td>{html_lib.escape(str(row['reason']))}</td></tr>"
            )
        rows.append("</table>")
    rows.append("</body></html>")
    index_path.write_text("\n".join(rows))


# %% [markdown]
# ## 5. Setup and input checks

# %%
start_time = time.time()
config = load_config()
audit_config = get_nested(config, "audit", {})
plot_config = get_nested(config, "audit.plot", {})
artifact_dir, figure_dir = resolve_output_dirs(config)
train_dir = resolve_train_dir(config)

print("experiment:", get_nested(config, "experiment.name", EXPERIMENT_NAME))
print("route:", get_nested(config, "experiment.route"))
print("status:", get_nested(config, "experiment.status"))
print("audit mode:", get_nested(config, "audit.mode"))
print("train_dir:", train_dir)
print("artifact_dir:", artifact_dir)
print("figure_dir:", figure_dir)
print("interpolation:", get_nested(config, "audit.interpolation"))
print(
    "use_true_tvt_for_prediction_target_eda:",
    get_nested(config, "audit.use_true_tvt_for_prediction_target_eda"),
)
print("max_wells:", get_nested(config, "audit.max_wells"))
print("well_include:", get_nested(config, "audit.well_include", []))
print("active variants / model configs / folds / boosters: 0 / 0 / 0 / 0")
if get_nested(config, "audit.use_true_tvt_for_prediction_target_eda") is not True:
    raise ValueError("This train-only EDA notebook requires true TVT on prediction-target rows")

horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
if not horizontal_paths:
    raise FileNotFoundError(f"No horizontal well CSV files found in {train_dir}")
well_include = {str(value) for value in get_nested(config, "audit.well_include", [])}
if well_include:
    horizontal_paths = [
        path for path in horizontal_paths if well_from_horizontal_path(path) in well_include
    ]
    missing_requested = sorted(
        well_include - {well_from_horizontal_path(path) for path in horizontal_paths}
    )
    if missing_requested:
        raise FileNotFoundError(f"Requested wells were not found: {missing_requested}")
max_wells = get_nested(config, "audit.max_wells")
if max_wells is not None:
    horizontal_paths = horizontal_paths[: int(max_wells)]

print("selected horizontal wells:", len(horizontal_paths))
print("first files:", [path.name for path in horizontal_paths[:5]])


# %% [markdown]
# ## 6. Generate one comparison PNG per well

# %%
horizontal_columns = [str(value) for value in get_nested(config, "audit.horizontal_columns")]
typewell_columns = [str(value) for value in get_nested(config, "audit.typewell_columns")]
manifest_rows: list[dict[str, Any]] = []

for well_index, horizontal_path in enumerate(horizontal_paths, start=1):
    well = well_from_horizontal_path(horizontal_path)
    typewell_path = typewell_path_for(horizontal_path)
    base_row: dict[str, Any] = {
        "well": well,
        "status": "skipped",
        "reason": "",
        "horizontal_path": str(horizontal_path),
        "typewell_path": str(typewell_path),
        "horizontal_bytes": int(horizontal_path.stat().st_size),
        "typewell_bytes": int(typewell_path.stat().st_size) if typewell_path.exists() else None,
    }
    try:
        horizontal, typewell = read_well_inputs(
            horizontal_path,
            typewell_path,
            horizontal_columns,
            typewell_columns,
        )
        reference_frame, well_metadata = build_reference_frame(horizontal, typewell)
        png_path = figure_dir / f"{well}.png"
        plot_well_comparison(
            well=well,
            frame=reference_frame,
            metadata=well_metadata,
            plot_config=plot_config,
            output_path=png_path,
        )
        base_row.update(well_metadata)
        base_row.update(
            {
                "status": "saved",
                "reason": "",
                "png_path": str(png_path),
                "png_bytes": int(png_path.stat().st_size),
                "png_sha256": sha256_file(png_path),
            }
        )
    except WellSkipError as exc:
        base_row.update(
            {
                "reason": str(exc),
                "png_path": None,
                "png_bytes": None,
                "png_sha256": None,
            }
        )
    manifest_rows.append(base_row)
    if well_index % 50 == 0 or well_index == len(horizontal_paths):
        saved_so_far = sum(row["status"] == "saved" for row in manifest_rows)
        print(f"processed {well_index:,}/{len(horizontal_paths):,}; saved={saved_so_far:,}")

manifest = pd.DataFrame(manifest_rows).sort_values("well").reset_index(drop=True)
saved_count = int((manifest["status"] == "saved").sum())
skipped_count = int(len(manifest) - saved_count)
if saved_count == 0:
    raise RuntimeError("No comparison PNGs were generated")

display(
    manifest[
        [
            "well",
            "status",
            "reason",
            "known_tvt_rows",
            "prediction_target_rows",
            "known_reference_rows",
            "prediction_target_reference_rows",
            "known_paired_gr_rows",
            "prediction_target_paired_gr_rows",
            "png_path",
        ]
    ].head(10)
)
print("saved / skipped:", saved_count, "/", skipped_count)


# %% [markdown]
# ## 7. Summary and generated artifacts

# %%
manifest_name = str(
    get_nested(
        config,
        "audit.outputs.manifest_csv",
        f"{OUTPUT_PREFIX}_manifest.csv",
    )
)
index_name = str(
    get_nested(
        config,
        "audit.outputs.index_html",
        f"{OUTPUT_PREFIX}_index.html",
    )
)
summary_name = str(
    get_nested(
        config,
        "audit.outputs.summary_json",
        f"{OUTPUT_PREFIX}_summary.json",
    )
)
manifest_path = artifact_dir / manifest_name
index_path = artifact_dir / index_name
summary_path = artifact_dir / summary_name

manifest.to_csv(manifest_path, index=False)
write_html_index(manifest, index_path, artifact_dir)

reason_counts = Counter(
    str(value) for value in manifest.loc[manifest["status"] != "saved", "reason"].tolist()
)
summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed" if skipped_count == 0 else "completed_with_skips",
    "route": get_nested(config, "experiment.route", "pf_beam"),
    "audit_mode": get_nested(config, "audit.mode"),
    "train_dir": str(train_dir),
    "input_wells": int(len(horizontal_paths)),
    "saved_pngs": saved_count,
    "skipped_wells": skipped_count,
    "skipped_by_reason": dict(sorted(reason_counts.items())),
    "interpolation": get_nested(config, "audit.interpolation"),
    "uses_true_tvt_for_prediction_target_eda": True,
    "prediction_target_mask": "TVT_input_isna",
    "x_axis": get_nested(config, "audit.x_axis"),
    "quality_metrics_estimated": [],
    "runtime_seconds": float(time.time() - start_time),
    "artifacts": {
        "figure_dir": str(figure_dir),
        "manifest_csv": str(manifest_path),
        "index_html": str(index_path),
        "summary_json": str(summary_path),
    },
    "sha256": {
        "manifest_csv": sha256_file(manifest_path),
        "index_html": sha256_file(index_path),
    },
    "execution_contract": {
        "active_variants": 0,
        "model_configs": 0,
        "folds": 0,
        "boosters": 0,
        "pf_beam_generation": False,
        "inference": False,
        "submission": False,
    },
}
write_json(summary_path, summary)

runtime_metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "route": "pf_beam",
    "cv": None,
    "public_lb": None,
    "private_lb": None,
    "metric": None,
    "input_wells": int(len(horizontal_paths)),
    "saved_pngs": saved_count,
    "skipped_wells": skipped_count,
    "uses_true_tvt_for_prediction_target_eda": True,
    "key_idea": "Known-TVT Type Well reference GR versus horizontal GR visualization.",
}
runtime_metrics_path = (
    KAGGLE_WORKING_ROOT / "metrics.json"
    if is_kaggle_runtime()
    else artifact_dir / f"{OUTPUT_PREFIX}_metrics.json"
)
write_json(runtime_metrics_path, runtime_metrics)

print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
print("generated artifacts:")
for path in [figure_dir, manifest_path, index_path, summary_path, runtime_metrics_path]:
    print(" -", path)
