# exp087_prefix_backtest_tvt_confidence 結果

## 状態

完了。Kaggle train version 2 が完了し、生成物を取得済み。

## 要約

exp072 の既存 full replay train feature cache を使い、PF/Beam / likelihood-PF confidence が TVT error を説明できるかを fold-safe に診断した。

主結果:

- rows / wells: 3,783,989 / 773
- primary candidate: `pf_ancc`
- primary PF RMSE: 14.493050690
- primary PF MAE: 8.921559334
- expected error vs absolute error Pearson: 0.519681049
- high-error threshold: 15.234375
- unstable expected error threshold: 13.889335522
- unstable flag rate: 20.000005%
- unstable flag high-error rate: 53.837748%
- stable flag high-error rate: 11.541393%
- top-vs-bottom confidence bin observed MAE lift: 7.745938

confidence bin は observed MAE を明確に分離した。

| bin | rows | observed MAE | high-error rate |
| --- | ---: | ---: | ---: |
| bin_0_low | 756,798 | 2.460268 | 0.015496 |
| bin_1_mid | 756,798 | 5.386803 | 0.076981 |
| bin_2_mid | 756,797 | 7.663323 | 0.136118 |
| bin_3_mid | 756,798 | 10.040319 | 0.233061 |
| bin_4_high | 756,798 | 19.057079 | 0.538377 |

signal correlation は `pf_likpf_abs` が最も強い。

| signal | Pearson | Spearman |
| --- | ---: | ---: |
| pf_likpf_abs | 0.589850 | 0.523773 |
| md_since | 0.288224 | 0.322989 |
| pf_beam_abs | 0.359971 | 0.264001 |
| beam_likpf_abs | 0.272138 | 0.245015 |
| likpf_delta_abs | 0.247383 | 0.211492 |

## 解釈

PF/Beam 系 signal は直接 TVT 置換には弱いが、誤差信頼度としては十分に使える。特に `pf_likpf_abs`、`pf_beam_abs`、`beam_likpf_abs`、`likpf_delta_abs`、`md_since` は high-error row / long-tail bucket の識別に効いている。

holdout phase は calibration phase より明確に難しく、observed MAE は 4.996978 から 11.035052 に上がった。`md_2500_plus` holdout は observed MAE 11.897038、high-error rate 0.289617 で、long-tail instability を再確認した。

## 判断

`prefix_backtest_tvt_confidence` は成功条件を満たした。予測値置換や hard router には進めず、次は `pf_beam_disagreement_sample_weight` の confidence feature / sample weight 候補として吸収する。

優先する特徴は `pf_likpf_abs`、`pf_beam_abs`、`beam_likpf_abs`、`likpf_delta_abs`、`md_since`。exp086 の `beam_std_d` / `dense_dist` 系と合わせ、まず add-only feature と sample-weight only を小さく比較する。

## 生成物

- `artifacts/prefix_backtest_tvt_confidence_summary.json`
- `artifacts/prefix_backtest_tvt_confidence_candidate_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_confidence_bin_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_bucket_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_phase_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_fold_metrics.csv`
- `artifacts/prefix_backtest_tvt_confidence_signal_correlations.csv`
- `artifacts/exp087-prefix-backtest-tvt-confidence-train.log`

row-level predictions は 124MB のため repo 配下には同期せず、`/tmp/kaggle-output/exp087_prefix_backtest_tvt_confidence/train_v2/artifacts/prefix_backtest_tvt_confidence_predictions.csv.gz` に保持する。
