# exp213_prefix_structural_prior_pfbeam 結果

## 状態

Kaggle train v1 完了。train-side diagnostic のみで、inference / submit は行わない。

| item | value |
| --- | --- |
| kernel | `kentookumura/exp213-prefix-structural-prior-pfbeam-train` v1 |
| URL | https://www.kaggle.com/code/kentookumura/exp213-prefix-structural-prior-pfbeam-train |
| runtime | summary 3,415.409 sec / log last 3,632.211 sec |
| score surface | `TVT_input_missing_equivalent_exp063_rows` |
| rows / wells | 478,958 rows / 64 wells |
| output | `experiments/exp213_prefix_structural_prior_pfbeam/kaggle/output/train_v1` |

## 仮説

known prefix の `TVT_input + Z` を structural surface として robust fit し、raw GR PF/Beam の初期速度、PF transition pull、Beam step-delta cost、absolute TVT soft costに使う。hard window は使わず、二峰性や datum alias を殺さないよう top-K path と cost gap / spread を保存する。

## 主要結果

| candidate | rows | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| `oracle_best_variant_candidate` | 478,958 | 13.940251 | 7.933458 | 0.740468 | oracle diagnostic |
| `oracle_best_nonraw_variant_candidate` | 478,958 | 15.795705 | 9.811364 | 0.659095 | oracle diagnostic |
| `exp072_pf_ancc` | 478,958 | 17.494197 | 10.454963 | 0.668491 | best non-oracle |
| `beam_structural_base_top3_oracle` | 478,958 | 18.287587 | 13.066916 | 0.510761 | Beam top-K oracle diagnostic |
| `beam_structural_base_top1` | 478,958 | 18.312677 | 13.115793 | 0.507662 | raw Beam から -0.026510 |
| `beam_raw_top1` | 478,958 | 18.339188 | 13.121684 | 0.509375 | generated raw Beam |
| `pf_raw_lik_mean` | 478,958 | 21.081279 | 14.491598 | 0.513446 | primary baseline |
| `pf_structural_weak_lik_mean` | 478,958 | 28.230909 | 18.833653 | 0.453543 | raw PF から +7.149629 |
| `pf_structural_base_lik_mean` | 478,958 | 29.564037 | 20.534350 | 0.392231 | raw PF から +8.482757 |
| `pf_structural_slope_only_lik_mean` | 478,958 | 30.621856 | 21.033016 | 0.407257 | raw PF から +9.540576 |

## 診断

- `well_status`: 64/64 wells が `ok`。
- structural prior は 64/64 wells で active。
- exp072 reference candidates は `pf_ancc` / `pf_z` のみ存在した。
- `structural_base` Beam は raw Beam から changed rows 12,108、row diff RMSE 0.760959。
- `beam_structural_base_top1` は 35/64 wells 改善、29/64 wells 悪化、max regression +15.387904。
- `beam_structural_base_top1` は distance bucket 全体で小幅改善: `1000_plus` は 19.234020 -> 19.205349。
- PF structural variants は resampling rate が raw 0.051346 から 0.066-0.076 へ増え、log likelihood mean も raw -1731.846 から -3531 から -3927 へ悪化した。

## 解釈

prefix structural prior は Beam の deterministic path cost には小さな regularization として効いたが、改善幅は RMSE -0.026510 と小さく、既存 `exp072_pf_ancc` に届かない。Beam top-K oracle でも RMSE 18.287587 で、candidate family としての headroom は限定的。

一方、PF では structural prior が大きく悪化した。特に longtail `1000_plus` で `pf_raw_lik_mean` 22.185965 に対し、`pf_structural_weak_lik_mean` 30.132545、`structural_base` 31.545706、`structural_slope_only` 32.679869 まで悪化した。prefix surface fit の slope / delta prior が tail 側で wrong path へ継続的に引っ張った可能性が高い。

## 判断

`prefix_structural_prior_pfbeam` は direct PF/Beam generation 変更としては不採用。direct replacement、raw-test inference port、submit、P0-C の direct generation follow-up には進めない。

残す場合は、Beam top-K gap、path spread、raw-vs-structural disagreement、structural surface fit diagnostics を P2 `topk_path_confidence_features` などの selector/confidence feature 材料に限定する。

## 生成物

- `artifacts/exp213_prefix_structural_prior_pfbeam_candidate_metrics.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_filter_delta_metrics.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_bucket_metrics.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_by_well.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_group_metrics.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_pf_diagnostics.csv`
- `artifacts/exp213_prefix_structural_prior_pfbeam_row_candidates.csv.gz`
- `artifacts/exp213_prefix_structural_prior_pfbeam_summary.json`

row candidates decompressed SHA256: `138e6fb9116630325ebe6bccc136955f472fb787e92c88f6eebdf8f7608ee3b6`
