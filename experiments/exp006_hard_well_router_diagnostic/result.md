# exp006_hard_well_router_diagnostic 結果

## 仮説

exp002 all-GR / exp003 no-GR / exp005 guarded の OOF 差分から、hidden test でも使える well 条件だけで hard well と public-like well を分けられる可能性がある。

## 設定

- 親: `exp005_gr_gate_recalibration`
- 入力: `experiments/exp003_residual_ablation/artifacts/exp002_exp003_well_delta.csv`
- 入力: `experiments/exp005_gr_gate_recalibration/artifacts/well_metrics.csv`
- ルール入力: GR 欠損率、prefix fraction、eval length、trajectory、GR shift
- 目的: 次の router 実装前の診断 artifact 作成
- Kaggle train kernel: `kentookumura/exp006-hard-well-router-diagnostic-train` version 1

## 結果

| Metric | Value |
| --- | ---: |
| Source wells | 773 |
| Evaluation rows | 3,783,989 |
| exp002 all-GR CV | 14.124569 |
| exp003 no-GR CV | 13.882944 |
| exp004 any low-GR gate CV | 13.932968 |
| exp005 strict low-GR gate CV | 13.936732 |
| Mean fold RMSE, selected | 13.913383 |
| Oracle all-GR/no-GR CV | 13.299351 |

| Router bucket | Wells |
| --- | ---: |
| ambiguous | 332 |
| hard_no_gr_candidate | 248 |
| public_like_keep_all_gr | 193 |

| Inference-safe rule | Selected wells | CV | Delta vs all-GR |
| --- | ---: | ---: | ---: |
| `low_gr_any_to_no_gr` | 297 | 13.932968 | -0.191601 |
| `low_gr_all_to_no_gr` | 214 | 13.936732 | -0.187837 |
| `low_gr_or_long_eval_steep_to_no_gr` | 258 | 13.970032 | -0.154537 |
| `short_prefix_low_gr_to_no_gr` | 101 | 14.006086 | -0.118483 |
| `default_all_gr` | 0 | 14.124569 | 0.000000 |

## 解釈

`low_gr_any_to_no_gr` は exp004 selected gate と同じで、診断上の最良 inference-safe rule だった。ただし選択 wells の no-GR win rate は 0.538721 に留まり、meaningful hurt rate も 0.400673 あるため、そのまま router として伸ばす余地は小さい。

一方、oracle best-of all-GR/no-GR は 13.299351 まで改善する。OOF 上では router headroom が大きく、次は `public_like_keep_all_gr` を守りつつ `hard_no_gr_candidate` を拾う条件を作る価値がある。

## 次

`exp007` では `router_diagnostic_well_tags.csv` と `router_candidate_rules.csv` を元に、all-GR / no-GR / guarded prediction を選ぶ fold-safe rule-based router を実装する。
