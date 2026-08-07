from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.pf_beam_disagreement_error_map import build_error_maps


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    well_summary_path = tmp_path / "well_summary.csv"
    plot_manifest_path = tmp_path / "plot_manifest.csv"
    ml_metrics_path = tmp_path / "ml_by_well.csv"

    pd.DataFrame(
        {
            "well_id": ["well_a", "well_b", "well_c"],
            "rows": [100, 200, 300],
            "primary_pf_rmse": [3.0, 8.0, 5.0],
            "primary_beam_rmse": [4.0, 6.0, 9.0],
            "anchor_rmse": [5.0, 7.0, 6.0],
            "pf_beam_abs_diff_mean": [1.0, 10.0, 4.0],
            "pf_beam_abs_diff_p95": [2.0, 12.0, 5.0],
            "pf_ancc_std_mean": [0.1, 0.3, 0.2],
            "beam_std_d_mean": [0.5, 1.0, 0.7],
            "known_len_mean": [800, 1600, 2600],
            "eval_len_mean": [900, 3000, 5500],
            "pfx_rmse_mean": [2.0, 5.0, 3.0],
            "likpf_mean_d_mean": [1.0, -2.0, 4.0],
        }
    ).to_csv(well_summary_path, index=False)
    pd.DataFrame(
        {
            "well_id": ["well_a", "well_b", "well_c"],
            "reason": ["all_wells", "all_wells", "all_wells"],
            "plot_path": ["a.png", "b.png", "c.png"],
        }
    ).to_csv(plot_manifest_path, index=False)
    pd.DataFrame(
        {
            "mode": ["gpu", "gpu", "gpu", "gpu"],
            "model": ["lgb0", "lgb_mean", "lgb_mean", "lgb_mean"],
            "well": ["well_a", "well_a", "well_b", "well_c"],
            "rows": [100, 100, 200, 300],
            "rmse_tvt": [10.0, 2.0, 9.0, 4.0],
            "error_mean": [1.0, 0.1, -0.2, 0.3],
            "error_abs_mean": [1.0, 1.5, 7.0, 3.0],
        }
    ).to_csv(ml_metrics_path, index=False)
    return well_summary_path, plot_manifest_path, ml_metrics_path


def test_build_error_maps_joins_ml_and_buckets(tmp_path: Path) -> None:
    well_summary_path, plot_manifest_path, ml_metrics_path = write_fixture(tmp_path)

    outputs = build_error_maps(
        well_summary_path=well_summary_path,
        plot_manifest_path=plot_manifest_path,
        ml_well_metrics_path=ml_metrics_path,
        ml_model="lgb_mean",
    )

    overall = outputs.overall.iloc[0]
    assert overall["wells"] == 3
    assert overall["rows"] == 600
    assert math.isclose(overall["pf_pooled_rmse"], math.sqrt((9 * 100 + 64 * 200 + 25 * 300) / 600))
    assert math.isclose(overall["ml_pooled_rmse"], math.sqrt((4 * 100 + 81 * 200 + 16 * 300) / 600))
    assert overall["pf_beats_ml_wells"] == 1
    assert overall["pf_beats_beam_wells"] == 2

    well_b = outputs.well_map.loc[outputs.well_map["well_id"] == "well_b"].iloc[0]
    assert well_b["plot_path"] == "b.png"
    assert well_b["ml_model"] == "lgb_mean"
    assert math.isclose(well_b["pf_minus_ml_rmse"], -1.0)

    segment_types = set(outputs.bucket_metrics["segment_type"])
    assert "pf_beam_disagreement_bucket" in segment_types
    assert "tail_length_bucket" in segment_types
    assert "best_engine" in segment_types
