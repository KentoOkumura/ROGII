# exp170_heel_calibrated_shift_scan_pfbeam_audit 結果

## 仮説

known prefix の `TVT_input` で typewell GR を horizontal GR に affine calibration すると、raw / flat-calibrated よりも shift-scan localization と固定 PF/Beam 候補の observation likelihood が改善する。

## 設定

- 親/参照: `heel_calibrated_shift_scan_pfbeam_audit` backlog、`exp167_fft_denoised_gr_matching_audit`、`exp072_exp063_full_replay_feature_cache`
- route: `pf_beam`
- Kaggle kernel: `kentookumura/exp170-heel-calib-shift-scan-pfbeam-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp170-heel-calib-shift-scan-pfbeam-train
- 検証: train-side sampled hidden-tail / prefix-backtest diagnostic
- rows: hidden_tail 197,888 / prefix_backtest 197,888
- wells: 773 train wells
- PF/Beam 再生成: なし
- ML 学習: なし
- inference / submit: なし

## 結果

raw `raw__raw` を基準にした mean abs-error gain は、heel calibration で悪化した。

| surface | region | mean abs-error gain vs raw | improved rate | gap gain | entropy reduction | decoy gap gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw__heel_calibrated | hidden_tail | -2.058258 | 0.203186 | -0.008637 | -0.018298 | -0.185166 |
| rolling_median_11__heel_calibrated | hidden_tail | -2.076816 | 0.270507 | 0.121226 | -0.006411 | 0.254398 |
| savgol_31_p2__heel_calibrated | hidden_tail | -2.784873 | 0.276934 | 0.140620 | -0.008400 | 0.317823 |
| raw__flat_calibrated | hidden_tail | -27.570070 | 0.284065 | -0.191514 | -0.104525 | -1.691606 |
| rolling_median_11__raw | hidden_tail | 0.209137 | 0.186055 | 0.133844 | 0.015125 | 0.506772 |
| raw__heel_calibrated | prefix_backtest | -2.160277 | 0.195120 | -0.049655 | -0.020143 | -0.262186 |
| rolling_median_11__heel_calibrated | prefix_backtest | -1.956323 | 0.262184 | 0.081025 | -0.009204 | 0.188582 |
| savgol_31_p2__heel_calibrated | prefix_backtest | -2.691592 | 0.273372 | 0.092466 | -0.012295 | 0.171283 |
| rolling_median_11__raw | prefix_backtest | 0.446558 | 0.181790 | 0.130796 | 0.014422 | 0.504657 |
| savgol_31_p2__raw | prefix_backtest | 0.662791 | 0.203019 | 0.176596 | 0.018778 | 0.674403 |

固定 PF/Beam 候補の hidden_tail RMSE は calibration mode によらず同じで、best は `likpf_mean`。

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| likpf_mean | 11.471434 | 6.989252 | 0.775439 | -1.036218 |
| pf_ancc | 14.106718 | 8.641896 | 0.701377 | -1.060010 |
| beam_mean | 15.453396 | 10.617674 | 0.602988 | -1.401851 |

PF/Beam observation cost は `raw__heel_calibrated` が `likpf_mean` の mean gap だけは 10.852016 -> 10.771421 と小さく下げたが、mean rank は 18.163254 -> 19.346994、top1 rank rate は 0.052105 -> 0.045445、top5 rank rate は 0.233162 -> 0.215344 に悪化した。candidate selection / likelihood replacement の根拠にはならない。

## 再現性

- deterministic anchor: false。診断生成物であり submission anchor ではない。
- seed policy: no_rng_deterministic_linspace_sampling
- upstream stochastic component: exp072 PF/Beam/likelihood-PF train feature cache
- Kaggle kernel version: v1
- `gain_vs_raw.csv` SHA256: `d9ce3885758a47ae8dd1d7353b0fc192c90e92c910747a20548c3c243f19b11c`
- `pfbeam_candidate_metrics.csv` SHA256: `f9700ad384db6040ab2650ee712a0b892edec82e740b02abfb3dd8f1e707545e`
- `pfbeam_observation_metrics.csv` SHA256: `8553bf732b7441035ff55b365c84eb2f5a32d94462e18f3a5678d456b9a9216f`
- `bucket_metrics.csv` SHA256: `294dc65f7f93e1e8cd1935054d1f46a986b4f260c1a1ba360fcd50116afbc557`
- model SHA / manifest SHA: 対象外
- prediction SHA: 対象外
- submission SHA: 対象外
- output note: `kaggle kernels output` は大きい `row_context.csv.gz` 取得中に中断し、集計 CSV だけを根拠にした。

## 解釈

known-prefix heel calibration は、今回の affine calibration 実装では shift-scan top1 localization を改善しなかった。flat calibration は大きく壊れ、heel calibration も raw に対して hidden_tail / prefix_backtest とも約 2ft 悪化した。

rolling median / Savitzky-Golay の raw smoothing は exp167 と同じく surface の gap / entropy / decoy gap を改善するが、heel calibration との組み合わせでは top1 error が悪化する。PF/Beam observation likelihood も rank/top5 が悪化しており、PF/Beam likelihood、initial offset prior、exp148 ML feature へ heel calibration を進める根拠はない。

結論として、`heel_calibrated_shift_scan_pfbeam_audit` は完了/不採用。inference port / submit はしない。

## 次

`denoised_gr_pfbeam_generation_audit` は、heel calibration 依存では進めない。続ける場合は exp167 で支持された rolling/savgol smoothing の小さな observation likelihood audit に限定する。GR alignment 系は、direct candidate 改善ではなく `bimodal_posterior_pfbeam_candidate_audit` のような mode ambiguity / posterior diagnostics を優先する。
