# Public notebook catch-up after self improvements

調査日: 2026-06-06

## Context

- Self anchor: exp026_pseudo_tail_bucket_shrink_inference_submit Public LB 12.102
- Trigger: high-priority self improvements have produced a new public LB anchor, so public top-notebook replay can start as a separate, audited route.
- Listings scanned: score_ascending_latest, vote_top, date_run_recent
- Inventory CSV: `docs/notebooks/rogii-wellbore-geology-prediction/public_notebook_catchup_inventory_2026-06-06.csv`

## Refresh commands

Run these before regenerating this report when network/Kaggle credentials are available:

```bash
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/vote_top --sort-by voteCount --force
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest --sort-by scoreAscending --force
python3 .agents/skills/kaggle-notebook-fetch/scripts/fetch_top_notebooks.py --competition rogii-wellbore-geology-prediction --limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent --sort-by dateRun --force --retries 3
uv run python scripts/public_notebook_catchup.py --as-of 2026-06-06
```

## Replay Queue

| Priority | Ref | Score | Family | Risks | Replay note |
| --- | --- | ---: | --- | --- | --- |
| target | `kojimar/rogii-physical-pf-signal-meets-artifact-stack` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `needless090/lb-8-860-rogii-sel15-256seeds` | 8.860 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `safar1/lb-score-8-863` | 8.863 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `aidensong123/rogii-sel15-rerun` | - | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` | 9.251 | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `svanikkolli/aeroridge-engine-v2` | - | aeroridge | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| high | `qamarmath/ml-physics-wellbore-geology-prediction` | - | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `alisalmanrana/wellbore-geology-prediction-ridge-8d98aa` | - | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `pilkwang/rogii-eda-target-free-alignment-for-tvt` | - | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `misakamikoto66/rogii-tree-models-and-physics-blend` | - | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `ravaghi/wellbore-geology-prediction-ridge` | - | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188` | 8.188 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `needless090/lb8-781-rogii-sel15-spread3` | 8.781 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |

## Target Notebook Inventory

| Priority | Ref | Score | Family | Risks | Replay note |
| --- | --- | ---: | --- | --- | --- |
| target | `kojimar/rogii-physical-pf-signal-meets-artifact-stack` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `needless090/lb-8-860-rogii-sel15-256seeds` | 8.860 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `safar1/lb-score-8-863` | 8.863 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `aidensong123/rogii-sel15-rerun` | - | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` | 9.251 | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `svanikkolli/aeroridge-engine-v2` | - | aeroridge | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |

## Family Counts

- `pf_physical_sel15`: 28
- `unknown`: 9
- `dwt_alignment`: 6
- `aeroridge`: 4
- `artifact_stack`: 3
- `postprocess`: 1

## Risk Counts

- `formation_or_geology_boundary_check`: 50
- `static_submission_or_blend_check`: 37
- `public_visible_branch_check`: 31
- `gpu_required_or_enabled`: 28
- `external_artifact_dependency`: 25
- `internet_enabled`: 3

## Implementation Handoff

1. Start with `needless090/lb8-781-rogii-sel15-spread3` as `exp027_public_replay_needless090_sel15_spread3` if refreshed metadata still shows no external dataset/model/kernel dependency.
2. Replay the selected public notebook on Kaggle without code changes first; record kernel version, output hash, runtime, `submission.csv` checks, and LB.
3. Keep replay output separate from self CV until dependency and hidden-safety checks pass.
4. Treat `kojimar/*TabICL*` and AeroRidge routes as artifact-stack audits first, because they require dataset/model input inventory before replay.

## Notes

- Kaggle listing metadata does not expose public score directly; `known_score` is parsed only from notebook titles.
- `formation_or_geology_boundary_check` is a review flag, not an automatic rejection. The replay audit must distinguish hidden-safe runtime inputs from train-only leakage.
- Static public CSV blends remain unsafe until rerun on hidden test inside the submitted Kaggle notebook.
