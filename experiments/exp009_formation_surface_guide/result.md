# exp009_formation_surface_guide 結果

## 状態

Kaggle full CV 完了。提出なし。

## スコア

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `formation_knn_no_gr` | `no_gr_signal_plus_formation_guide` | 14.558630 | +0.434061 |
| `formation_knn_all` | `all_plus_formation_guide` | 14.739226 | +0.614657 |

## 解釈

fold-safe KNN formation surface guide は、no-GR / all-GR のどちらに足しても CV を悪化させた。selected `formation_knn_no_gr` は `control_exp003_no_gr` より 0.675686 悪く、`control_exp002_all` より 0.434061 悪い。

この設計では formation surface guide を採用しない。train-only formation columns を直接使っていないためリークは避けたが、KNN surface からの距離特徴は residual model の一般化に寄与しなかった。

## 次のアクション

1. exp009 は提出しない。
2. 次は backlog の `exp010_trajectory_drift_ablation` に進み、formation guide なしで trajectory drift 特徴を切り分ける。
