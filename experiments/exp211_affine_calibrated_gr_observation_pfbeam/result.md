# exp211_affine_calibrated_gr_observation_pfbeam 結果

## 状態

Kaggle train v1 完了。train-side diagnostic のみで、inference / submit は行わない。

| item | value |
| --- | --- |
| kernel | `kentookumura/exp211-affine-gr-pfbeam-train` v1 |
| URL | https://www.kaggle.com/code/kentookumura/exp211-affine-gr-pfbeam-train |
| id_no | `126197987` |
| runtime | summary 3,300.654 sec / log last 3,543.860 sec |
| score surface | `TVT_input_missing_equivalent_exp063_rows` |
| rows / wells | 478,958 rows / 64 wells |
| output | `experiments/exp211_affine_calibrated_gr_observation_pfbeam/kaggle/output/train_v1` |

## 仮説

known prefix で horizontal GR と typewell GR のscale/offsetを補正すると、raw GR observation likelihood よりPF/Beamが正しいtypewell深度へ乗りやすくなる可能性がある。効果分解のため、raw/affine observation と classic/prefix structural transition の2x2を同時比較した。

## 主要結果

| candidate | rows | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| `oracle_best_variant_candidate` | 478,958 | 12.731058 | 6.520979 | 0.803645 | oracle diagnostic |
| `oracle_best_nonraw_variant_candidate` | 478,958 | 13.179088 | 7.157056 | 0.771594 | oracle diagnostic |
| `exp072_pf_ancc` | 478,958 | 17.494197 | 10.454963 | 0.668491 | best non-oracle |
| `beam_affine_top1` | 478,958 | 18.065010 | 13.014080 | 0.512043 | raw Beam から -0.274177 |
| `beam_affine_structural_top1` | 478,958 | 18.176860 | 13.095639 | 0.510565 | raw Beam から -0.162328 |
| `beam_raw_top1` | 478,958 | 18.339188 | 13.121684 | 0.509375 | generated raw Beam |
| `pf_raw_lik_mean` | 478,958 | 18.640063 | 12.097552 | 0.598904 | primary baseline |
| `pf_affine_lik_mean` | 478,958 | 21.184758 | 14.008033 | 0.550432 | raw PF から +2.544695 |
| `pf_affine_structural_lik_mean` | 478,958 | 21.143708 | 14.148794 | 0.529666 | raw PF から +2.503645 |

## 診断

- affine fallback は 0/64 wells。fit自体は全wellでguardを通過した。
- affine slope は mean 0.852530 / median 0.852239 / range 0.506014-1.121831。
- affine prefix RMSE は mean 7.893849 / median 7.514946 / max 13.680113。
- PF diagnostics は raw `log_likelihood_mean` -1762.210 に対し、affine は -2318.942、affine+structural は -2684.768 と悪化した。
- ESS は raw 175.944、affine 176.168 とほぼ同等だが、resampling rate は raw 0.051148、affine 0.052524 とやや増えた。
- `beam_affine_top1` は 31/64 wells 改善、33/64 wells 悪化、max well regression +20.781499。
- `pf_affine_lik_mean` は 26/64 wells 改善、38/64 wells 悪化、max well regression +19.521376。

## 解釈

affine calibration は prefix上では安定してfitできたが、PF/likelihood-PFの観測likelihoodとしては悪化した。GR scaleをtypewell側へ正規化したことで、prefix fitの局所整合は上がっても、tail側では wrong depth への吸い込みを強めた可能性が高い。prefix structural prior も今回の弱prior設定では救済にならず、PF側は raw より悪化した。

Beam は `beam_affine_top1` が raw Beam より RMSE -0.274177 改善し、longtail でも 19.234020 -> 18.925897 と改善した。ただし best non-oracle は既存 `exp072_pf_ancc` RMSE 17.494197 のままで、affine Beam は within10 も低く、max well regression が +20ft 以上残る。direct replacement と inference port の根拠には足りない。

oracle headroom は `oracle_best_nonraw_variant_candidate` RMSE 13.179088 と大きい。これは affine candidate を直接使うのではなく、affine-vs-raw disagreement、prefix calibration quality、Beam confidence を selector / confidence feature として使う余地を示す。

## 判断

`affine_calibrated_gr_observation_pfbeam` は direct PF/Beam generation 変更としては不採用。direct replacement、raw-test inference port、submit は行わない。

残す場合は `topk_path_confidence_features` などの selector/confidence feature 材料に限定し、affine slope/intercept、prefix RMSE、raw-vs-affine Beam disagreement、oracle gap を小さい診断列として扱う。

## 生成物

- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_candidate_metrics.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_filter_delta_metrics.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_bucket_metrics.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_by_well.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_group_metrics.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_pf_diagnostics.csv`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_row_candidates.csv.gz`
- `artifacts/exp211_affine_calibrated_gr_observation_pfbeam_summary.json`

row candidates decompressed SHA256: `8dba28dfe8a82536293f56e7f204715679ce2f7354f8e14ac46c6b079ec71465`
