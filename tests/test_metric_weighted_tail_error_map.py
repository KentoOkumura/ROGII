from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.metric_weighted_tail_error_map import (
    CandidateSpec,
    build_error_maps,
    parse_candidate_spec,
)


def write_visible_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    sample_path = tmp_path / "sample_submission.csv"
    candidate_path = tmp_path / "candidate.csv"
    anchor_path = tmp_path / "anchor.csv"

    pd.DataFrame({"id": ["well_a_2", "well_a_3", "well_b_1"], "tvt": [0.0, 0.0, 0.0]}).to_csv(
        sample_path, index=False
    )
    pd.DataFrame(
        {
            "MD": [100.0, 101.0, 102.0, 103.0],
            "TVT": [10.0, 11.0, 13.0, 16.0],
            "TVT_input": [10.0, 11.0, None, None],
        }
    ).to_csv(train_dir / "well_a__horizontal_well.csv", index=False)
    pd.DataFrame(
        {
            "MD": [200.0, 201.0],
            "TVT": [20.0, 24.0],
            "TVT_input": [20.0, None],
        }
    ).to_csv(train_dir / "well_b__horizontal_well.csv", index=False)
    pd.DataFrame(
        {"id": ["well_a_2", "well_a_3", "well_b_1"], "tvt": [14.0, 15.0, 22.0]}
    ).to_csv(candidate_path, index=False)
    pd.DataFrame(
        {"id": ["well_a_2", "well_a_3", "well_b_1"], "tvt": [11.0, 11.0, 20.0]}
    ).to_csv(anchor_path, index=False)
    return sample_path, train_dir, candidate_path, anchor_path


def test_parse_candidate_spec_uses_name_prefix() -> None:
    spec = parse_candidate_spec("exp027=experiments/exp027/artifacts/submission.csv")

    assert spec.name == "exp027"
    assert spec.path.name == "submission.csv"


def test_build_error_maps_scores_candidates_against_anchor(tmp_path: Path) -> None:
    sample_path, train_dir, candidate_path, anchor_path = write_visible_fixture(tmp_path)

    overall, well_map, bucket_map, row_map = build_error_maps(
        candidate_specs=[CandidateSpec("candidate", candidate_path)],
        sample_path=sample_path,
        train_dir=train_dir,
        anchor_spec=CandidateSpec("anchor", anchor_path),
    )

    assert overall.loc[0, "rows"] == 3
    assert math.isclose(overall.loc[0, "rmse"], math.sqrt(6.0 / 3.0))
    assert math.isclose(overall.loc[0, "anchor_rmse"], math.sqrt(45.0 / 3.0))
    assert math.isclose(overall.loc[0, "delta_sse_vs_anchor"], -39.0)

    well_a = well_map.loc[well_map["well_id"] == "well_a"].iloc[0]
    assert well_a["tail_rows"] == 2
    assert math.isclose(well_a["sse"], 2.0)
    assert math.isclose(well_a["anchor_sse"], 29.0)
    assert not bool(well_a["worse_than_anchor"])

    assert set(bucket_map["segment_type"]) == {"distance_bucket", "tail_length_bucket"}
    assert len(row_map) == 3
    assert row_map["weighted_sse_share"].sum() == 1.0
