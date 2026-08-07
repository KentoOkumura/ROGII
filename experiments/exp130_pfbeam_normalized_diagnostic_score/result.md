# exp130_pfbeam_normalized_diagnostic_score 結果

## 仮説

PF/Beam は直接置換では exp092 ML anchor に届かないが、normalized U/MD 空間での滑らかさ、prefix slope との整合、候補間 disagreement は、候補 path を信用できる場面の confidence feature として使える可能性がある。

## 実装

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- route: `ml_model`
- 追加特徴量:
  - well-local `md_since_norm` と target-free `u_scale`
  - candidate ごとの normalized U、slope、curvature、roughness、prefix residual
  - PF/Beam/likPF 候補間の normalized U disagreement
  - `normalized_diagnostic_score`、instability、shape score margin、confidence flags

## 結果

棄却。

Kaggle train v1 は OOM で失敗した。失敗した package は古い設定のまま `exp092_full_row_control` と `pfbeam_normalized_diagnostic_addonly` の両方を学習しており、control 学習後、addonly の `lgb2` fold0 付近で kernel が落ちた。

途中ログで比較できる範囲では、addonly は有効とは言えない。

| model | 比較 fold | control RMSE 単純平均 | addonly RMSE 単純平均 | add-control |
| --- | ---: | ---: | ---: | ---: |
| `lgb0` | 5/5 | 9.499024 | 9.563375 | +0.064351 |
| `lgb1` | 5/5 | 9.291006 | 9.336510 | +0.045503 |
| `lgb2` | 1/5 | 8.380249 | 8.373010 | -0.007238 |

`lgb2` fold0 のみは微改善だが、1 fold だけで判断不能。完走した `lgb0` / `lgb1` では悪化しているため、追加 GPU を使って full rerun する価値は低いと判断した。

## 判断基準

inference port、submit、後続の `u_state_pf_candidate` 入力化は行わない。normalized diagnostic score は、少なくともこの full-row exp092 add-only 形では棄却する。
