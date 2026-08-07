# exp011_tracker_divergence_features 結果

## 状態

Kaggle full CV 完了。提出なし。

## スコア

| Variant | Feature Set | CV | exp002 差分 |
| --- | --- | ---: | ---: |
| `control_exp003_no_gr` | `no_gr_signal` | 13.882944 | -0.241625 |
| `control_exp002_all` | `all` | 14.124569 | 0.000000 |
| `tracker_divergence_no_gr` | `no_gr_signal_plus_tracker` | 14.903823 | +0.779254 |
| `tracker_direction_no_gr` | `no_gr_signal_plus_tracker_direction` | 14.918102 | +0.793533 |
| `tracker_divergence_all` | `all_plus_tracker` | 14.955276 | +0.830707 |

## 解釈

selected `tracker_divergence_no_gr` は CV 14.903823 で、CV 基準 の `control_exp003_no_gr` 13.882944 より 1.020879 悪い。tracker features は no-GR、all-GR、限定 trajectory direction 併用のすべてで悪化したため、提出しない。

group summary でも exp010 audit で注意していた条件が悪化した。

| Group | control_exp003_no_gr | tracker_divergence_no_gr | 差分 |
| --- | ---: | ---: | ---: |
| hard_no_gr_candidate | 10.675618 | 11.334843 | +0.659225 |
| steep_trajectory | 12.078737 | 13.375446 | +1.296709 |
| high_gr_missing | 10.746502 | 11.387577 | +0.641075 |
| long_eval | 12.253799 | 13.171136 | +0.917337 |

Kaggle train runtime は log timestamp 上で約 1,914 秒。tracker variants は各 8-9 分程度で、見えない test well 用推論 に入れる前に feature pruning か routing が必要。

## 次のアクション

1. exp011 は提出しない。
2. 現行 deterministic tracker add-only features は凍結する。
3. 次は `exp012_model_diversity_or_postprocess` に進む。tracker を再検討する場合は、まず well-level failure audit と confidence/routing の診断に限定する。
