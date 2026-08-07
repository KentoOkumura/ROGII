# exp147_exp092_exp098_rank_slot_replacement_only 結果

## 状態

Kaggle train v1 完了。結論は rejected。inference / submit は行わない。

## 目的

exp139 の small rank-slot add-only が exp092 best を更新できなかったため、rank-slot と重複する exp092 generated features を限定的に落とし、rank-slot signals で代替した場合に非重複ゲインが出るか確認した。

## 実装

- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- variant: `u_projection_rank_slot_replacement_only`
- base features: exp072/exp073 full replay 196 features を維持
- dropped generated columns: 22
- replacement rank-slot columns: 25
- final model features: 243
- candidate path replacement / blend / stack: なし

## Kaggle Train v1

- Kernel: `kentookumura/exp147-rank-slot-replacement-train`
- URL: https://www.kaggle.com/code/kentookumura/exp147-rank-slot-replacement-train
- Runtime: 15,532.769 sec
- Rows / wells: 3,783,989 / 773
- Model count: 15
- Output dir: `experiments/exp147_exp092_exp098_rank_slot_replacement_only/kaggle/output/train_v1`

## OOF

| model | pooled RMSE | vs exp092 same model | note |
| --- | ---: | ---: | --- |
| `lgb0` | 9.665612516 | +0.132486078 vs exp092 lgb0 | 悪化 |
| `lgb1` | 9.423893838 | +0.101413942 vs exp092 lgb1 | 悪化 |
| `lgb2` | 9.397013393 | +0.058820988 vs exp092 lgb2 | best だが exp092 best に届かない |
| `lgb_mean` | 9.438575715 | +0.095511649 vs exp092 lgb_mean | 悪化 |

Best `lgb2` は exp092 `lgb1` 9.322479896 より +0.074533497 悪い。exp139 small add-only `lgb1` 9.324907641 よりも +0.072105752 悪く、rank-slot replacement-only は anchor 更新にならない。

## Bucket / Worst Well

Distance bucket:

| model | `000_050` RMSE | `1000_plus` RMSE |
| --- | ---: | ---: |
| `lgb1` | 1.301245 | 10.346889 |
| `lgb2` | 1.351854 | 10.317358 |
| `lgb_mean` | 1.143712 | 10.365809 |

Worst wells は既存 warning と同じ方向で、`lgb1` は `86454a6f` 57.324829、`1b1eba53` 41.660320、`fb03ae90` 40.288345。`lgb2` も `86454a6f` 57.717678、`1b1eba53` 42.359455、`fb03ae90` 39.913269。exp092 の worst-well 問題を解消する根拠はない。

## Feature Importance

rank-slot U-shape features は上位に残った。

- `rank1_u_curvature`: mean importance 6571.07
- `rank2_u_curvature`: 6381.80
- `rank3_u_curvature`: 6257.40
- `rank2_u_slope`: 5276.40
- `rank3_u_slope`: 5203.47
- `rank1_u_slope`: 5113.60

ただし、重要度が高くても global OOF は大きく悪化した。exp092 generated overlap columns を落とすと、rank-slot が置換しきれない情報を失うと判断する。

## 判断

exp147 は rejected。inference port / submit はしない。

rank-slot signal は特徴として学習されるが、exp092 の U-projection correction / disagreement と置換するには弱い。exp139 add-only は微小悪化、exp147 replacement-only は明確悪化なので、rank-slot 系は replacement / pruning で追わない。残すなら full 64列 add-only の低優先対照実験に限定する。
