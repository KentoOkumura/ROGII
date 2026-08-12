# Public notebook catch-up after self improvements

調査日: 2026-06-11

## Context

- Self anchor: exp026_pseudo_tail_bucket_shrink_inference_submit Public LB 12.102
- Trigger: high-priority self improvements have produced a new public LB anchor, so public top-notebook replay can start as a separate, audited route.
- Listings scanned: score_ascending_20260611, vote_top_20260611, date_run_recent_20260611
- Inventory CSV: `docs/notebooks/rogii-wellbore-geology-prediction/public_notebook_catchup_inventory_2026-06-11.csv`

## Refresh commands

Run these before regenerating this report when network/Kaggle credentials are available:

```bash
task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction EXTRA_ARGS="--limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/vote_top --sort-by voteCount --force"
task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction EXTRA_ARGS="--limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest --sort-by scoreAscending --force"
task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction EXTRA_ARGS="--limit 20 --output-dir docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent --sort-by dateRun --force --retries 3"
uv run python studies/public_notebook_catchup.py --as-of 2026-06-11
```

`task`が利用できない環境では、同じ変数と引数で`make fetch-kaggle-notebooks`を使う。

## Replay Queue

| Priority | Ref | Score | Family | Risks | Replay note |
| --- | --- | ---: | --- | --- | --- |
| target | `aidensong123/rogii-sel15-rerun` | - | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `kojimar/rogii-physical-pf-signal-meets-artifact-stack` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `needless090/lb-8-860-rogii-sel15-256seeds` | 8.860 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` | 9.251 | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `safar1/lb-score-8-863` | 8.863 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `svanikkolli/aeroridge-engine-v2` | - | aeroridge | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `afr1ste/rogii-public-koolbox-truth-probe-8-107` | 8.107 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188` | 8.188 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `lightningv08/lb-7-776-rogii-ridge-sp` | 7.776 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| high | `needless090/lb8-781-rogii-sel15-spread3` | 8.781 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| high | `tasmim/lb-9-259-rogii-wellbore-geology-prediction` | 9.259 | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check | Inventory external inputs before replay. |

## Target Notebook Inventory

| Priority | Ref | Score | Family | Risks | Replay note |
| --- | --- | ---: | --- | --- | --- |
| target | `aidensong123/rogii-sel15-rerun` | - | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `kojimar/rogii-inference-stack-with-pf-beam-and-tabicl` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `kojimar/rogii-physical-pf-signal-meets-artifact-stack` | - | artifact_stack | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, public_visible_branch_check | Inventory external inputs before replay. |
| target | `needless090/lb-8-860-rogii-sel15-256seeds` | 8.860 | pf_physical_sel15 | formation_or_geology_boundary_check, public_visible_branch_check, static_submission_or_blend_check | Replay candidate: no metadata external dataset/model/kernel sources. |
| target | `nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based` | 9.251 | dwt_alignment | external_artifact_dependency, formation_or_geology_boundary_check, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `safar1/lb-score-8-863` | 8.863 | pf_physical_sel15 | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |
| target | `svanikkolli/aeroridge-engine-v2` | - | aeroridge | external_artifact_dependency, formation_or_geology_boundary_check, gpu_required_or_enabled, static_submission_or_blend_check | Inventory external inputs before replay. |

## Family Counts

- `pf_physical_sel15`: 69
- `unknown`: 14
- `dwt_alignment`: 9
- `artifact_stack`: 6
- `aeroridge`: 6
- `postprocess`: 6

## Risk Counts

- `formation_or_geology_boundary_check`: 106
- `static_submission_or_blend_check`: 78
- `public_visible_branch_check`: 70
- `external_artifact_dependency`: 64
- `gpu_required_or_enabled`: 46
- `internet_enabled`: 5
- `none`: 1

## Implementation Handoff

1. Start with `needless090/lb8-781-rogii-sel15-spread3` as `exp027_public_replay_needless090_sel15_spread3` if refreshed metadata still shows no external dataset/model/kernel dependency.
2. Replay the selected public notebook on Kaggle without code changes first; record kernel version, output hash, runtime, `submission.csv` checks, and LB.
3. Keep replay output separate from self CV until dependency and hidden-safety checks pass.
4. Treat `kojimar/*TabICL*` and AeroRidge routes as artifact-stack audits first, because they require dataset/model input inventory before replay.

## Notes

- Kaggle listing metadata does not expose public score directly; `known_score` is parsed only from notebook titles.
- `formation_or_geology_boundary_check` is a review flag, not an automatic rejection. The replay audit must distinguish hidden-safe runtime inputs from train-only leakage.
- Static public CSV blends remain unsafe until rerun on hidden test inside the submitted Kaggle notebook.
