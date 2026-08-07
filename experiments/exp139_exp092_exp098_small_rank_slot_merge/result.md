# exp139_exp092_exp098_small_rank_slot_merge 結果

## 状態

Kaggle train v1 完了。結論は rejected。inference / submit は行わない。

## 目的

exp092 の U-projection correction / disagreement surface に、exp098 の rank-slot signal を小さく add-only merge し、exp092 に対して非重複の改善が残るか確認した。

## 実装

- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- variant: `u_projection_rank_slot_small_merge`
- base features: exp072/exp073 full replay 196 features
- model features: 255
- U-projection feature groups: `projection_correction`, `u_disagreement`
- rank-slot additional columns: 15 columns from rank1/rank2 score, source, U-shape, and rank-slot U spread
- candidate path replacement: なし

## Kaggle Train v1

- Kernel: `kentookumura/exp139-small-rank-slot-train`
- URL: https://www.kaggle.com/code/kentookumura/exp139-small-rank-slot-train
- Runtime: 15,866.843 sec
- Rows / wells: 3,783,989 / 773
- Model count: 15

Full output download は model artifact 取得中に止めたが、完全な Kaggle log summary と、bucket / by-well / feature importance / schema / manifest は取得済み。

## OOF

| model | pooled RMSE | vs exp092 same model | note |
| --- | ---: | ---: | --- |
| `lgb0` | 9.613226293 | +0.080099855 vs exp092 lgb0 | 悪化 |
| `lgb1` | 9.324907641 | +0.002427745 vs exp092 lgb1 | best だが exp092 best に届かない |
| `lgb2` | 9.337578311 | -0.000614093 vs exp092 lgb2 | 微小改善だが exp092 best より悪い |
| `lgb_mean` | 9.370584225 | +0.027520159 vs exp092 lgb_mean | 悪化 |

exp098 `lgb1` 9.358151052 よりは改善したが、採用基準の exp092 `lgb1` 9.322479896 を更新できなかった。

## Bucket / Worst Well

`lgb1` の distance bucket は `000_050` RMSE 1.410827、`1000_plus` RMSE 10.234380。`lgb2` は `000_050` 1.339466、`1000_plus` 10.250080。

`lgb1` worst wells は `86454a6f` 57.459114、`1b1eba53` 42.140781、`fb03ae90` 41.229336。exp092 の既存 by-well warning を解消する根拠はない。

## Feature Importance

rank-slot shape features は上位に入った。

- `rank1_u_curvature`: mean importance 6719.27
- `rank2_u_curvature`: 6585.07
- `rank2_u_slope`: 5475.80
- `rank1_u_slope`: 5442.07
- `rank2_u_resid_mad`: 3031.80

ただし重要度が高くても global OOF best は更新しない。rank-slot shape signal は exp092 と重複、または追加ノイズとして働いた可能性が高い。

## 判断

exp139 は rejected。inference port / submit はしない。

exp098 rank-slot signal は exp073 には効いたが、exp092 上ではすでに U-projection correction / disagreement が近い情報を吸収しており、small add-only merge では非重複ゲインが残らなかったと判断する。今後 rank-slot を続ける場合は full union / pruning ではなく、normalized shape や candidate quality diagnostic として別仮説に分ける。
