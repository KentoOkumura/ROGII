# exp147_exp092_exp098_rank_slot_replacement_only

## Status

Kaggle train v1 完了。OOF が exp092 / exp139 を更新できなかったため rejected。inference / submit は行わない。

## Hypothesis

exp139 では exp092 に exp098 rank-slot 代表 15 列を add-only したが、best `lgb1` は exp092 `lgb1` から +0.002428 悪化した。一方で rank-slot U-shape features は feature importance 上位に入り、信号自体はある。

exp147 では add-only ではなく replacement-only として、exp092 の generated U-projection / disagreement features のうち rank-slot と重複する列を限定的に落とし、rank1/rank2/rank3 の rank-slot signals で置換する。

## Validation Strategy

exp072 deterministic full replay train feature cache、base 196 features、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM family を固定する。rank-slot features は exp098 と同じ target-free score で作り、exp098 prediction / OOF / blend / direct selector は使わない。

比較対象は exp092 `lgb1` 9.322479896 / `lgb_mean` 9.343064066、exp098 `lgb1` 9.358151052、exp139 `lgb1` 9.324907641 / `lgb_mean` 9.370584225。

## Scope

- Route: `ml_model`
- Parent: `exp092_u_projection_correction_disagreement_fullrun`
- Rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Variant: `u_projection_rank_slot_replacement_only`
- Active mode: `gpu_repro_guard_dp_threads8`
- Train plan: 1 variant x 3 LightGBM configs x 5 folds = 15 boosters
- Dropped generated columns: 22
- Replacement rank-slot columns: 25
- Control retraining: none; exp092/exp098/exp139 saved metrics are fixed references

## Findings

Kaggle train v1 は `kentookumura/exp147-rank-slot-replacement-train` で完了した。

| model | pooled OOF RMSE |
| --- | ---: |
| `lgb2` | 9.397013393 |
| `lgb1` | 9.423893838 |
| `lgb_mean` | 9.438575715 |
| `lgb0` | 9.665612516 |

Best `lgb2` は exp092 `lgb1` 9.322479896 より +0.074533497 悪く、exp139 `lgb1` 9.324907641 よりも +0.072105752 悪い。rank-slot U-shape features は重要度上位に入ったが、overlap columns を落とす replacement-only は exp092 の強い U-projection/disagreement surface を壊した。

Decision: rejected。inference port / submit はしない。

## Expected Outputs

- `exp147_exp092_exp098_rank_slot_replacement_only_metrics.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_by_well.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_bucket_metrics.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_projection_feature_summary.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_rank_slot_feature_summary.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_feature_importance.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_feature_importance_mean.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_feature_importance_mean_top.png`
- `exp147_exp092_exp098_rank_slot_replacement_only_predictions.csv.gz`
- `exp147_exp092_exp098_rank_slot_replacement_only_feature_schema.csv`
- `exp147_exp092_exp098_rank_slot_replacement_only_lgb_models/manifest.json`
- `exp147_exp092_exp098_rank_slot_replacement_only_summary.json`
