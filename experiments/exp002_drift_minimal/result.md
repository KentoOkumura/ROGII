# exp002_drift_minimal 結果

## 仮説

`last_anchor` の残差を trajectory、GR、既知 prefix のみから学習すると、単純な slope 外挿より安定して drift を補正できる。

## 設定

- 親: `exp001_baseline`
- 検証: `well_id` GroupKFold、`TVT_input` NaN 行のみ評価
- メトリック: RMSE
- シード: 42
- モデル: `HistGradientBoostingRegressor`
- target: `TVT - last_anchor_tvt`

## 結果

| メトリック | 値 |
| --- | --- |
| Full CV OOF RMSE (`drift_hgb`) | 14.124569 |
| Full mean fold RMSE (`drift_hgb`) | 14.101909 |
| Full CV OOF RMSE (`last_anchor`) | 15.909853 |
| Full mean fold RMSE (`last_anchor`) | 15.894391 |
| Debug CV OOF RMSE (`drift_hgb`, 30 wells) | 17.770814 |
| Debug CV OOF RMSE (`last_anchor`, 30 wells) | 12.145321 |
| Public LB | 12.533 |
| Private LB | - |

## 解釈

Kaggle train full CV では `drift_hgb` が 14.124569、同一 split の `last_anchor` が 15.909853 だった。exp001 比で RMSE 1.785284、約 11.22% 改善しており、`last_anchor` residual 学習は有効。

Kaggle inference kernel version 1 から提出した ref `53211155` は public LB 12.533。exp001 public LB 15.883 から 3.350 改善した。Public LB は CV より 1.591569 良いが、公開 test は小さいため private 一般化判断では CV を重視する。

Debug 30 wells では悪化していたため、小標本 debug score はこの実験の成否判断には使えない。train artifact は `/tmp/kaggle-output/exp002_drift_minimal/train`、submission は `/tmp/kaggle-output/exp002_drift_minimal/inference/submission.csv` から取得済み。

## 次

sampling / shrink / feature ablation を次の小実験にする。
