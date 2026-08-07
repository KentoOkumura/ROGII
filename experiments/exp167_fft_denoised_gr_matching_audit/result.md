# exp167_fft_denoised_gr_matching_audit 結果

## 仮説

GR sensor rotation 由来の周期ノイズを target-free FFT notch で落とすと、typewell GR matching の localization surface が raw GR より改善する。

## 設定

- 親: `fft_denoised_gr_matching_audit` backlog
- 検証: train-side typewell GR shift-scan diagnostic
- Kaggle kernel: `kentookumura/exp167-fft-denoised-gr-matching-audit-train` v2
- URL: https://www.kaggle.com/code/kentookumura/exp167-fft-denoised-gr-matching-audit-train
- rows: 395,776 sampled row-filter evaluations per all-region filter
- wells: 773 train wells
- メトリック: RMSE / MAE / within2 / within5 / within10 / top1-top2 gap / entropy / decoy gap / raw-vs-denoised gain
- シード: 42。ただし乱数は使わない。

## 結果

| filter | region | RMSE | MAE | within10 | gap | entropy | decoy gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | all | 108.659395 | 69.872190 | 0.163022 | 1.166088 | 0.712493 | 5.846472 |
| fft_notch_top2 | all | 108.581079 | 69.807558 | 0.163499 | 1.165975 | 0.711578 | 5.848499 |
| rolling_median_11 | all | 108.532613 | 69.544342 | 0.165760 | 1.298408 | 0.697719 | 6.352187 |
| savgol_31_p2 | all | 108.522038 | 69.569115 | 0.165121 | 1.343931 | 0.693299 | 6.520488 |
| raw | hidden_tail | 125.711348 | 76.615849 | 0.151576 | 1.156021 | 0.711906 | 5.815072 |
| fft_notch_top2 | hidden_tail | 125.580817 | 76.453655 | 0.151722 | 1.152656 | 0.711042 | 5.814574 |
| rolling_median_11 | hidden_tail | 125.690496 | 76.406712 | 0.153122 | 1.289864 | 0.696782 | 6.321843 |
| savgol_31_p2 | prefix_backtest | 87.711709 | 62.465740 | 0.177393 | 1.352752 | 0.694301 | 6.552275 |

raw 比 gain:

| filter | region | mean abs-error gain | improved rate | gap gain | entropy reduction | decoy gap gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fft_notch_top2 | hidden_tail | 0.162194 | 0.191917 | -0.003365 | 0.000864 | -0.000498 |
| fft_notch_top2 | prefix_backtest | -0.032930 | 0.202933 | 0.003138 | 0.000964 | 0.004552 |
| rolling_median_11 | hidden_tail | 0.209137 | 0.186055 | 0.133844 | 0.015125 | 0.506772 |
| rolling_median_11 | prefix_backtest | 0.446558 | 0.181790 | 0.130796 | 0.014422 | 0.504657 |
| savgol_31_p2 | hidden_tail | -0.056642 | 0.203934 | 0.179089 | 0.019608 | 0.673629 |
| savgol_31_p2 | prefix_backtest | 0.662791 | 0.203019 | 0.176596 | 0.018778 | 0.674403 |

## 再現性

- deterministic anchor: false。診断生成物であり submission anchor ではない。
- seed policy: no_rng_deterministic_linspace_sampling
- kernel version: `kentookumura/exp167-fft-denoised-gr-matching-audit-train` v2
- `filter_metrics.csv` SHA256: `1062b2ea50743e895b409426d8f51cdce920261f073b48aede520d34c8c8bc48`
- `filter_gain_vs_raw.csv` SHA256: `782c2463d8fea29a344d33b585904f141703acdd3e40d1beca8dbdbd2d00b125`
- `bucket_metrics.csv` SHA256: `6c645c89a15be49c6872744bb9ebf26239312759b36e8da1e3c29dfaef7fcec2`
- model SHA / manifest SHA: 対象外
- prediction SHA: 対象外
- submission SHA: 対象外
- output note: `row_context.csv.gz` は大きく、`kaggle kernels output` が `IncompleteRead` で中断したためローカルには保存しない。取得済みの集計 CSV だけを根拠にする。

## 解釈

FFT notch は raw に対して all RMSE -0.078316、hidden_tail RMSE -0.130530 とごく小さい改善に留まる。hidden_tail では top1-top2 gap と decoy gap が raw より微悪化しており、matching cost surface が明確に鋭くなったとは言えない。prefix_backtest でも RMSE はわずかに良いが、MAE / within2 / within5 は悪化した。

一方、rolling median と Savitzky-Golay fallback は gap、entropy、decoy gap の改善が大きい。これは「FFT rotation denoise」よりも、短周期 smoothing が typewell matching surface を少し安定化している可能性を示す。ただし direct top1 RMSE は 100ft 級で、直接 candidate や submit に使う水準ではない。

結論として、`fft_denoised_gr_matching_audit` は完了/不採用。FFT notch を `denoised_gr_pfbeam_generation_audit` にそのまま進めない。続ける場合は、rolling/savgol smoothing と heel calibration を組み合わせた別の shift-scan audit として切る。

## 次

`heel_calibrated_shift_scan_pfbeam_audit` を優先し、GR denoise 単体ではなく known-prefix gain/offset calibration が matching / PF observation likelihood を改善するかを見る。
