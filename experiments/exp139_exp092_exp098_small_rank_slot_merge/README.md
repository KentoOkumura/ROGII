# exp139_exp092_exp098_small_rank_slot_merge

## Status

Kaggle train v1 完了。OOF が exp092 best を更新できなかったため rejected。inference / submit は行わない。

## Hypothesis

exp098 の rank-slot features は standalone では exp092 に届かなかったが、exp077 より明確に改善し、rank-slot idea 自体は有用だった。一方、exp105 compact、exp107 top-n candidate-only、exp108 top-n related prune は exp098 full rank-slot より弱く、単純な pruning は支持されなかった。

そこで exp092 の強い U-projection correction / disagreement surface を親にし、exp098 full 64列を一括 union せず、代表的な rank-slot signal だけを small add-only で足す。非重複ゲインが残るかを小さく検証する。

## Validation Strategy

exp072 deterministic full replay train feature cache、exp092 U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM family を固定する。追加する rank-slot features は exp098 と同じ target-free score で作り、候補 TVT path の direct selector / soft average / postprocess replacement は行わない。

比較対象は exp092 `lgb1` 9.322479896 / Public LB 8.350、exp098 `lgb1` 9.358151052 / Public LB 8.441、exp107 best 9.437602823、exp108 best 9.479370656。

## Scope

- Route: `ml_model`
- Parent: `exp092_u_projection_correction_disagreement_fullrun`
- Rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Variant: `u_projection_rank_slot_small_merge`
- Models: `lgb0` / `lgb1` / `lgb2` and `lgb_mean`
- Inference: train-side OOF と guard review 後に判断

## Findings

Kaggle train v1 は `kentookumura/exp139-small-rank-slot-train` で完了。

| model | pooled OOF RMSE |
| --- | ---: |
| `lgb1` | 9.324907641 |
| `lgb2` | 9.337578311 |
| `lgb_mean` | 9.370584225 |
| `lgb0` | 9.613226293 |

Best `lgb1` は exp092 `lgb1` 9.322479896 より +0.002427745 悪く、anchor 更新にならない。`lgb2` は exp092 `lgb2` から -0.000614093 と微小改善したが、exp092 best には届かない。

rank-slot shape features は feature importance 上位に入ったが、global OOF best を更新しないため submit しない。

## Expected Outputs

- `exp139_exp092_exp098_small_rank_slot_merge_metrics.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_by_well.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_bucket_metrics.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_projection_feature_summary.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_rank_slot_feature_summary.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_feature_importance.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_feature_importance_mean.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_feature_importance_mean_top.png`
- `exp139_exp092_exp098_small_rank_slot_merge_predictions.csv.gz`
- `exp139_exp092_exp098_small_rank_slot_merge_feature_schema.csv`
- `exp139_exp092_exp098_small_rank_slot_merge_lgb_models/manifest.json`
- `exp139_exp092_exp098_small_rank_slot_merge_summary.json`
