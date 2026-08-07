# exp189_denoised_gr_pfbeam_generation_audit 結果

## 仮説

raw GR の局所ノイズを rolling median / Savitzky-Golay で固定 smoothing すると、PF/Beam の observation likelihood が安定し、candidate RMSE、effective sample size、resampling 頻度、path jump、selector oracle headroom のいずれかが raw baseline より改善する可能性がある。

## 設定

- 親: `denoised_gr_pfbeam_generation_audit` backlog
- route: `pf_beam`
- 検証面: exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows`
- target wells: 64 wells / 478,958 rows
- GR filters: `raw`、`rolling_median_w11`、`savgol_w31_p2`
- PF config: 240 particles x 8 seeds / filter
- Beam config: beam size 14、move radius 2
- seed policy: filter 間で同じ query well / seed index の stable SHA256 seed を共有
- kernel: `kentookumura/exp189-denoised-gr-pfbeam-audit-train` v1

## 結果

| メトリック | 値 |
| --- | --- |
| CV | diagnostic only |
| Public LB | - |
| Private LB | - |

主要 candidate:

| candidate | rows | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| `oracle_best_filter_candidate` | 478,958 | 9.394252 | 5.083778 | 0.844360 | oracle diagnostic |
| `oracle_best_smoothed_candidate` | 478,958 | 10.643257 | 6.496276 | 0.792900 | oracle diagnostic |
| `exp072_pf_ancc` | 478,958 | 17.494197 | 10.454963 | 0.668491 | best non-oracle, saved reference |
| `beam_rolling_median_w11_top1` | 478,958 | 18.028587 | 12.620731 | 0.529414 | best generated non-oracle |
| `beam_savgol_w31_p2_top1` | 478,958 | 18.136752 | 12.513397 | 0.546071 | generated |
| `beam_raw_top1` | 478,958 | 18.339188 | 13.121684 | 0.509375 | raw Beam baseline |
| `pf_raw_lik_mean` | 478,958 | 20.225464 | 13.027728 | 0.564546 | primary baseline |
| `pf_rolling_median_w11_lik_mean` | 478,958 | 26.893376 | 19.974596 | 0.379854 | PF smoothing, 悪化 |
| `pf_savgol_w31_p2_lik_mean` | 478,958 | 27.943343 | 21.166070 | 0.359994 | PF smoothing, 悪化 |

raw family との比較:

- Beam rolling median: `beam_raw_top1` 18.339188 -> 18.028587、delta -0.310600。
- Beam Savitzky-Golay: `beam_raw_top1` 18.339188 -> 18.136752、delta -0.202435。
- PF rolling median: `pf_raw_lik_mean` 20.225464 -> 26.893376、delta +6.667912。
- PF Savitzky-Golay: `pf_raw_lik_mean` 20.225464 -> 27.943343、delta +7.717879。

by-well:

- `beam_rolling_median_w11_top1`: improved 32/64 wells、worsened 32/64 wells、max regression +17.732656。
- `beam_savgol_w31_p2_top1`: improved 35/64 wells、worsened 29/64 wells、max regression +15.722090。
- `pf_rolling_median_w11_lik_mean`: improved 17/64 wells、worsened 47/64 wells、max regression +32.262938。
- `pf_savgol_w31_p2_lik_mean`: improved 17/64 wells、worsened 47/64 wells、max regression +44.016493。
- `oracle_best_smoothed_candidate`: improved 54/64 wells、worsened 10/64 wells、max regression +7.191860。

PF diagnostics:

| filter | ESS mean | resampling rate | GR sigma |
| --- | ---: | ---: | ---: |
| raw | 175.913089 | 0.051202 | 13.897759 |
| rolling_median_w11 | 177.722133 | 0.039357 | 11.663793 |
| savgol_w31_p2 | 180.393690 | 0.039204 | 11.598912 |

## 再現性

- deterministic anchor: false
- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_gr_filters`
- kernel version: `kentookumura/exp189-denoised-gr-pfbeam-audit-train` v1
- kernel URL: https://www.kaggle.com/code/kentookumura/exp189-denoised-gr-pfbeam-audit-train
- kernel id_no: 125901169
- runtime: summary 1,416.547 sec / logs last time 1,539.590 sec
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- row candidates decompressed SHA: `614225c6265a04e30917bc8e417d70d1b6ecab1597816a700786cb99728adc55`
- row candidates raw gzip SHA: `5059ee061d3bef78eaccf3efe2e3ae5b467526fadeaefbfeb7d35365e33fd66f`
- model SHA / manifest SHA: model なし
- submission SHA: submission なし

## 解釈

PF likelihood に smoothed GR を直接入れる案は不採用。rolling median / Savitzky-Golay は ESS を少し上げ、resampling rate を下げるが、候補 TVT としては RMSE が大きく悪化した。GR smoothing によって likelihood が滑らかになりすぎ、PF が wrong depth へ安定して吸い込まれている可能性が高い。

Beam は rolling median / Savitzky-Golay で raw Beam より小幅改善した。特に longtail / mid range では改善するが、near row では raw Beam が良く、best generated でも `exp072_pf_ancc` 17.494197 には届かない。max well regression も +15ft 以上あるため、direct replacement、inference port、submit には進めない。

一方、`oracle_best_smoothed_candidate` は RMSE 10.643257 で headroom がある。これは smoothing 候補そのものではなく、「どの row/well で smoothed Beam を信用できるか」を判断する selector / confidence feature の材料としてのみ価値がある。

## 次

`denoised_gr_pfbeam_generation_audit` は完了/診断のみとして閉じる。PF/Beam generation likelihood の直接変更、direct candidate replacement、inference port、submit は行わない。残す場合は既存 backlog `denoised_calibrated_matching_features_on_exp148` の範囲で、smoothed Beam の raw-vs-smoothed delta、oracle headroom、path disagreement、PF diagnostics を selector / ML confidence feature として扱う。
