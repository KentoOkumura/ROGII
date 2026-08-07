# exp131_gr_shape_descriptor_matching_ablation 結果

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

既存の raw GR / NCC / DTW add-only は弱かったが、PF/Beam 候補集合には oracle headroom がある。GR を直接 TVT path にするのではなく、local shape descriptor cost として候補の当たり外れを評価すれば、observation likelihood / verifier feature として使える可能性がある。

## 設定

- Kernel: `kentookumura/exp131-gr-shape-descriptor-train` v1
- output: `experiments/exp131_gr_shape_descriptor_matching_ablation/kaggle/output/train_v1`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- rows / wells: 3,783,989 / 773
- runtime: 5,987.899 sec
- 候補: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`
- score variants: `raw_point_real`、`ncc_window_real`、`banded_shift_real`、`shape_descriptor_real`、`combo_descriptor_real`、`combo_descriptor_shuffled`、`no_gr_constant`

## 結果

best single candidate は既存 `likpf_mean` のまま。

| candidate | RMSE | MAE | within10 |
| --- | ---: | ---: | ---: |
| `likpf_mean` | 11.594897 | 7.067633 | 0.772807 |
| `pf_ancc` | 14.493051 | 8.921559 | 0.691741 |
| `beam_mean` | 15.774327 | 10.898586 | 0.591649 |

candidate-long AUC / logloss では `combo_descriptor_real` が最良。

| score variant | AUC | logloss | note |
| --- | ---: | ---: | --- |
| `combo_descriptor_real` | 0.659206 | 0.653906 | best |
| `banded_shift_real` | 0.658538 | 0.672613 | close |
| `raw_point_real` | 0.630080 | 0.779421 | positive |
| `combo_descriptor_shuffled` | 0.570007 | 0.736530 | negative control |
| `shape_descriptor_real` | 0.559753 | 0.703394 | weak alone |
| `ncc_window_real` | 0.500051 | 0.833901 | no signal |
| `no_gr_constant` | 0.500000 | 0.877025 | control |

`combo_descriptor_real` は shuffled から AUC +0.089199、no-GR から +0.159206 で、real GR signal は negative control を明確に上回った。

一方で direct top1 scorer としては壊れる。

| top1 score | RMSE | MAE | within10 | selected top |
| --- | ---: | ---: | ---: | --- |
| `no_gr_constant` | 14.493051 | 8.921559 | 0.691741 | `pf_ancc` |
| `combo_descriptor_real` | 84.919128 | 33.664665 | 0.560979 | `pf_ancc` |
| `banded_shift_real` | 86.157570 | 34.466015 | 0.557261 | `pf_ancc` |
| `combo_descriptor_shuffled` | 99.416473 | 49.403240 | 0.466587 | `pf_ancc` |

baseline oracle は RMSE 7.434030 / within10 0.906525 で、候補集合 headroom は引き続き大きい。

## 解釈

`combo_descriptor_real` は候補が within10 かどうかを説明する likelihood feature としては支持される。特に shuffled-GR / no-GR controls を上回ったため、GR shape cost 自体には signal がある。

ただし score 最大候補を直接選ぶと RMSE 84.919128 まで崩壊する。`pf_ancc` 偏重になり、`likpf_mean` 単体にも大きく負けるため、direct scorer、hard switch、direct candidate path、inference port、submit はしない。

## 再現性

- deterministic anchor: false
- seed policy: no new RNG in exp131
- train feature cache rows: 3,783,989
- train feature cache features: 138
- train feature raw SHA256: `e24a0803d2ade801e3bc655ea104df5e3042ef08b488e7489470ba379fed3e58`
- train feature decompressed SHA256: `e8d5fba94a6a9f0be401c023fc6b968c2e1dd5f3eeb40bacf98a2b262399cd4e`
- train feature schema SHA256: `ce378bd872ac2139dde2e1daa74e4122047a02caeed73048063d74bbeef46838`

`kaggle kernels output` は large train feature gzip の途中で長時間止まったため、ローカルには full output を完全取得していない。Kaggle logs 上では train feature cache size 1,790,873,153 bytes と上記 SHA を確認済み。診断に必要な summary は logs から `/tmp/exp131_summary_from_logs.json` に抽出した。

## 次

`gr_shape_descriptor_matching_ablation` は完了として backlog から外す。残す場合は `combo_descriptor_real` / `banded_shift_real` の score、margin、top1-top2 gap、candidate 別 score を exp092 系 ML confidence feature または continuity-constrained verifier の低-中優先 feature 候補に限定する。直接 scorer / candidate replacement には戻さない。
