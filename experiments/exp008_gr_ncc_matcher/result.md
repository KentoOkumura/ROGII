# exp008_gr_ncc_matcher 結果

## 状態

Kaggle full CV は完了。NCC 追加の 2 variants がどちらも CV を悪化させたため、推論と提出は実行していない。

## 結果

| Variant | Feature Set | CV | Mean Fold RMSE | exp002 差分 |
| --- | --- | ---: | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | 13.859376 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 14.101909 | 0.000000 |
| `gr_ncc_no_gr_multi` | `no_gr_signal_plus_gr_ncc` | 14.641514 | 14.619825 | +0.516945 |
| `gr_ncc_all_multi` | `all_plus_gr_ncc` | 14.661017 | 14.642361 | +0.536448 |

## 解釈

typewell / horizontal GR の NCC 追加特徴は、この形では有効ではなかった。no-GR 特徴に NCC を加えると `control_exp003_no_gr` より +0.758570 悪化し、all-GR 特徴に NCC を加えると `control_exp002_all` より +0.536448 悪化した。

この実験は提出しない。現在の bounded-shift NCC 信号はノイズが大きいか、残差 target と噛み合っていないか、HGB モデルが過適合しやすい可能性が高い。

## 生成物

- `artifacts/ablation_metrics.csv`
- `artifacts/fold_metrics.csv`
- `artifacts/fold_model_training.csv`
- `artifacts/well_metrics.csv`
- `artifacts/exp008-gr-ncc-matcher-train.log`
