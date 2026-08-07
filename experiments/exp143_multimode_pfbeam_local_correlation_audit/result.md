# exp143_multimode_pfbeam_local_correlation_audit 結果

## 結論

Kaggle v3 は完了。6 wells / 12,000 rows の scoped train-side diagnostic として、期待した 8 生成物をすべて保存した。

この scoped audit では、主比較対象を従来 Beam の `exp072_beam_mean` と見ると、best multimode は明確に改善した。`exp072_beam_mean` RMSE 70.297647 に対し、best multimode `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` は RMSE 60.763085 で、RMSE -9.534561 / MAE -10.771005。

一方、全体候補としては `exp072_pf_ancc` RMSE 50.721842、`exp072_likpf_mean` 52.758772、`exp072_pf_z` 57.641691 に届かない。したがって従来 Beam 改善の診断としては positive だが、inference port / submit はしない。

診断上は mode diversity が well 依存で、`1b1eba53` と `91b301ce` では z-accel variant が mode count / topK spread を増やした。一方、`fb03ae90` と `86454a6f` ではほぼ単一 mode へ潰れており、topK spread も小さい。PF/Beam の mode retention を広げても、今回の非 oracle scorer では真値近傍候補を安定して選べない。

## 実行

- Kernel: `kentookumura/exp143-multimode-pfbeam-corr-train`
- Version: 3
- Runtime: 294.693 sec
- Scope: representative/failure 6 wells, max 2000 rows per well
- Rows / wells: 12,000 / 6
- Output: `experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v3/artifacts/`

## 主要指標

| 候補 | RMSE | MAE | bias | within10 |
| --- | ---: | ---: | ---: | ---: |
| `exp072_pf_ancc` | 50.721842 | 43.416754 | -3.273040 | 0.164833 |
| `exp072_likpf_mean` | 52.758772 | 47.635939 | -0.122496 | 0.000833 |
| `exp072_pf_z` | 57.641691 | 45.050145 | -12.649786 | 0.216667 |
| best multimode | 60.763085 | 55.599689 | -1.204916 | 0.006500 |
| best strict multiseed | 64.039534 | 59.543231 | 1.248467 | 0.005417 |
| `exp072_beam_mean` | 70.297647 | 66.370694 | 7.521430 | 0.004667 |

Best multimode delta:

- vs `exp072_beam_mean`: -9.534561 RMSE
- vs `exp072_likpf_mean`: +8.004313 RMSE
- vs `exp072_pf_z`: +3.121394 RMSE

Best strict multiseed delta:

- vs `exp072_beam_mean`: -6.258113 RMSE
- vs `exp072_likpf_mean`: +11.280762 RMSE
- vs `exp072_pf_z`: +6.397843 RMSE

## Mode / Local Correlation

Variant aggregate:

| variant | mean mode count | mode_count<=1 rate | mean entropy | mean p90-p10 spread | mean local corr topK spread | mean topK corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zmean_s006_noise025` | 1.309000 | 0.717000 | 0.224718 | 3.425996 | 2.214723 | 0.060912 |
| `zacc_s010_a020_noise050` | 1.462917 | 0.645333 | 0.301848 | 4.078184 | 3.605253 | 0.064942 |

well-level では、`zacc_s010_a020_noise050` が `1b1eba53` で mean mode count 2.447 / topK spread 7.2475、`91b301ce` で mean mode count 1.8435 / topK spread 9.5574 と複数 mode を残した。一方、`fb03ae90` は両 variant とも mean mode count 1.0、`86454a6f` もほぼ単一 mode で、collapse を避けるだけでは救えない well がある。

## 注意点

v1 full-run は timeout。v2 は scoped run の parity guard 誤判定で失敗。v3 は scoped audit として完了したが、full 773-well 結論ではない。

strict PF-Z parity は scoped slice で `max_abs_diff=60.40625` / `rmse_diff=25.161156` と大きく外れた。これは scoped v3 では raise しないが、exp072 strict parity の再現候補としては扱わない。今回の読みは mode diversity / collapse 診断に限定する。

## 次のアクション

`multimode_pfbeam_local_correlation_audit` は完了として閉じる。従来 Beam からの改善は positive だが、`pf_ancc` / `likpf_mean` / `pf_z` を超えないため、直接 submit 候補へ進める根拠は弱い。続けるなら、候補生成を増やす方向ではなく、exp092 系 ML の confidence feature、segment-level verifier、または learned likelihood / normalized shape feature に吸収する。
