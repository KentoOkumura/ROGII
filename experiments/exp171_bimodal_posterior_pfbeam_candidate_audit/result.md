# exp171_bimodal_posterior_pfbeam_candidate_audit 結果

## 仮説

GR shift-scan / Beam 系の二峰性 tail では、top1 mode に commit するより、top2 cost 差から作る posterior mean 候補の方が大外しを抑える可能性がある。

## 設定

- 親/参照: `bimodal_posterior_pfbeam_candidate_audit` backlog、`exp133_gr_bimodal_match_ambiguity_detector`、`exp167_fft_denoised_gr_matching_audit`、`exp170_heel_calibrated_shift_scan_pfbeam_audit`
- route: `pf_beam`
- Kaggle kernel: `kentookumura/exp171-bimodal-posterior-pfbeam-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp171-bimodal-posterior-pfbeam-train
- 検証: train-side sampled hidden-tail / prefix-backtest diagnostic
- rows: 1,187,328 row-context rows。hidden_tail 197,888 rows / prefix_backtest 197,888 rows x 3 filters。
- wells: 773 train wells
- PF/Beam 再生成: なし
- ML 学習: なし
- inference / submit: なし

## 結果

固定 exp072 候補を含めた全体 best は `likpf_mean` で、posterior / midpoint / hard commit を大きく上回った。

| candidate | filter | eval | RMSE | MAE | within10 |
| --- | --- | --- | ---: | ---: | ---: |
| `likpf_mean` | raw | all hidden-tail rows | 11.471434 | 6.989252 | 0.775439 |
| `pf_ancc` | raw | all hidden-tail rows | 14.106718 | 8.641896 | 0.701377 |
| `beam_mean` | raw | all hidden-tail rows | 15.453396 | 10.617674 | 0.602988 |
| `posterior_mean_t16` | rolling_median_11 | all rows | 76.698097 | 39.759649 | 0.224286 |
| `midpoint` | rolling_median_11 | all rows | 76.616702 | 39.660410 | 0.222164 |
| `commit_top1` | savgol_31_p2 | all rows | 77.455386 | 41.032059 | 0.221047 |

hidden_tail では posterior / midpoint は hard commit より小さく改善したが、`likpf_mean` には全く届かなかった。

| candidate | filter | hidden_tail RMSE | MAE | within10 |
| --- | --- | ---: | ---: | ---: |
| `likpf_mean` | raw | 11.471434 | 6.989252 | 0.775439 |
| `midpoint` | savgol_31_p2 | 102.245744 | 50.426430 | 0.206010 |
| `posterior_mean_t16` | savgol_31_p2 | 102.301054 | 50.511337 | 0.207900 |
| `commit_top1` | savgol_31_p2 | 102.774632 | 51.729033 | 0.199791 |

commit 比の mean abs-error gain は positive だが、直接候補としては弱い。

| candidate | surface | hidden_tail mean abs-error gain vs commit | improved rate |
| --- | --- | ---: | ---: |
| `midpoint` | rolling_median_11 | +1.328095 | 0.556320 |
| `posterior_mean_t16` | raw | +1.243621 | 0.558043 |
| `posterior_mean_t2` | rolling_median_11 | +0.845398 | 0.572182 |

prefix_backtest でも best は `midpoint` で、posterior はそれを超えなかった。

| candidate | filter | prefix_backtest RMSE | MAE | within10 |
| --- | --- | ---: | ---: | ---: |
| `midpoint` | rolling_median_11 | 35.809334 | 28.895598 | 0.237958 |
| `posterior_mean_t16` | rolling_median_11 | 35.999769 | 29.009512 | 0.240358 |
| `commit_top2` | rolling_median_11 | 36.120250 | 29.109647 | 0.226891 |

## 再現性

- deterministic anchor: false。診断生成物であり submission anchor ではない。
- seed policy: no_rng_deterministic_linspace_sampling
- upstream stochastic component: exp072 PF/Beam/likelihood-PF train feature cache
- Kaggle kernel version: v1
- runtime: 889.676942 sec
- `row_context.csv.gz` raw SHA256: `1f210ec405be93cec466e0094480f19d149c0582d056651abb34d17f5b80b14a`
- `row_context.csv.gz` decompressed SHA256: `8899ead616b05182788f77581097f606f1b15984ab036dc8bab8a75f1b70ebbd`
- `candidate_metrics.csv` SHA256: `5d67c63c0c61d8bc635c1a7de84752dd568c910e590b7fb9b4d3ebbd694a74eb`
- `bucket_metrics.csv` SHA256: `43ce84f003b655285ca468857c38105bab5bfbf5df3111a60867501ee0223cdc`
- `gain_vs_commit.csv` SHA256: `7c8c7bc9c5eae2597bea18e3a13975ccfe084b9514f47485b4fab9e7ff4c3c0b`
- model SHA / manifest SHA: 対象外
- prediction SHA: 対象外
- submission SHA: 対象外
- output note: full output は `/tmp/kaggle-output/exp171_bimodal_posterior_pfbeam_candidate_audit/train` に取得した。290MB の `row_context.csv.gz` は実験配下へ常設せず、小さい集計 CSV / summary のみ `artifacts/` に保存した。

## 解釈

top2 mode の posterior mean / midpoint は hard commit の外れを少し緩和するが、GR shift-scan surface 自体が train hidden-tail で弱く、固定 `likpf_mean` の RMSE 11.471434 に対して posterior 系は RMSE 102 前後まで悪化する。これは `exp133` の midpoint/proxy 失敗と整合する。

したがって、`bimodal_posterior_pfbeam_candidate_audit` は完了/不採用。posterior candidate の direct replacement、PF/Beam likelihood 変更、inference port、submit は行わない。`p`、entropy、mode separation、top2 gap も現状の実装では exp148 add-only feature へ進める根拠が弱い。

## 次

GR alignment 系で続けるなら、posterior mean ではなく、exp167 で相対的に支持された rolling / Savitzky-Golay smoothing の surface sharpness 診断、または exp157 / exp158 の候補 ranker confidence surface を exp148 へ渡す方向を優先する。
