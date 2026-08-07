# exp010_trajectory_drift_ablation 結果

## 状態

Kaggle full CV 完了。提出なし。

## スコア

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `trajectory_direction_no_gr` | `no_gr_signal_plus_trajectory_direction` | 14.023223 | -0.101346 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `trajectory_slope_no_gr` | `no_gr_signal_plus_trajectory_slope` | 14.177009 | +0.052440 |
| `trajectory_full_no_gr` | `no_gr_signal_plus_trajectory_drift` | 14.236694 | +0.112125 |
| `trajectory_full_all` | `all_plus_trajectory_drift` | 14.332077 | +0.207508 |

## 解釈

trajectory 形状特徴は direction だけなら exp002 control より改善したが、exp003 no-GR control には届かなかった。slope / full variants は exp002 control よりも悪化し、trajectory feature の add-only 追加は現行 HGB residual model では採用しない。

best row は `control_exp003_no_gr` 13.882944。selected `trajectory_full_no_gr` は 14.236694 で、exp003 control より 0.353750 悪い。

## 追加診断

2026-06-03 に `trajectory_feature_error_audit` を実行した。出力は `artifacts/trajectory_feature_error_audit/`。

- `trajectory_full_no_gr` は 351 wells を meaningful hurt、308 wells を meaningful better。
- 悪化が強い group は hard-no-GR 候補 248 wells: 16.721940 -> 17.688955、steep trajectory 186 wells: 15.208859 -> 15.979471、high GR missing 293 wells: 13.250584 -> 13.959989、long eval 235 wells: 14.240710 -> 14.886789。
- `public_like_keep_all_gr` 193 wells では 14.684601 -> 14.289867 と改善しており、trajectory signal は add-only ではなく selector / router 用に限定すべき。

## 次のアクション

1. exp010 は提出しない。
2. `exp011_tracker_divergence_features` に進む場合は、hard-no-GR / steep trajectory / high GR missing / long eval を事前に分け、trajectory full をそのまま再投入しない。
