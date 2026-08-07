# exp214_public_raw_gr_residual_scale_control 結果

## 結論

Kaggle train v1 は完了。公開 notebook lineage に近い raw GR + known-prefix residual scale の control として、`pf_raw_scale_5` は RMSE 15.596465、best non-oracle は `pf_raw_scale_12` の RMSE 15.223857 だった。

この結果により、exp211/213 の軽量 raw control は public-like PF control としては弱すぎたことが確認できた。GRCAL-PFBEAM 系の比較では、今後は exp214 の `pf_raw_scale_*` を raw public-like control として参照する。direct inference / submit は行わない。

## 設定

- Route: `pf_beam`
- 親: `public_raw_gr_residual_scale_control` backlog、`exp072` feature cache、`exp211/213` pseudo-tail 評価面
- PF: 500 particles x 128 seeds
- scale: 3 / 5 / 8 / 12
- primary: `pf_raw_scale_5`
- LightGBM: なし
- inference / submit: なし

## 結果

| candidate | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | --- |
| `pf_raw_scale_12` | 15.223857 | 9.218137 | 0.673913 | best non-oracle、primary から -0.372608 |
| `pf_raw_scale_8` | 15.436026 | 9.356569 | 0.666407 | primary から -0.160439 |
| `pf_raw_scale_5` / `pf_raw_lik_mean` | 15.596465 | 9.503912 | 0.661881 | primary baseline |
| `pf_raw_scale_3` | 15.676055 | 9.585502 | 0.657717 | primary から +0.079590 |
| `pf_raw_best_seed` | 15.752051 | 9.652642 | 0.656847 | seed best diagnostic |
| `pf_raw_seed_mean` | 16.029065 | 10.497578 | 0.635546 | seed mean diagnostic |
| `exp072_pf_ancc` | 17.494197 | 10.454963 | 0.668491 | exp072 reference |
| `beam_raw_top1` | 18.339188 | 13.121684 | 0.509375 | raw Beam reference |
| `exp072_pf_z` | 24.165177 | 13.864957 | 0.614252 | exp072 reference |
| `pf_raw_top3_oracle` | 14.236926 | 7.992557 | 0.731966 | oracle diagnostic |
| `oracle_best_variant_candidate` | 11.104328 | 5.609022 | 0.839702 | oracle diagnostic |

Kaggle summary runtime は 3,369.568 秒。対象は 478,958 rows / 64 wells。validation source は exp072 train feature cache 3,783,989 rows / 773 wells。

PF diagnostics は `gr_sigma` mean 13.897759、ESS mean 366.112576、resampling rate 0.053373。`pf_raw_scale_12` は primary `pf_raw_scale_5` に対して long-tail 側を中心に改善し、`1000_plus` bucket では RMSE 16.403659 -> 16.022127 だった。

## 解釈

この実験は改善候補ではなく、GRCAL-PFBEAM 系の raw public-like control を固定するための診断実験である。exp211 の `pf_raw_lik_mean` RMSE 18.640063、exp213 の `pf_raw_lik_mean` RMSE 21.081279 と比べると、public-like `TVT + Z` surface-state likelihood-PF の raw control は明確に強い。

一方で oracle best は RMSE 11.104328 まで下がるため、scale / seed / path confidence には selector feature としての余地がある。予測値そのものを direct replacement するのではなく、`topk_path_confidence_features` などの confidence / uncertainty 材料として扱う。

## 生成物

- Kaggle output: `experiments/exp214_public_raw_gr_residual_scale_control/kaggle/output/train_v1`
- row candidates decompressed SHA: `bef105d23466b13be8d3caee907dd1e5cea1d4f7468907116a95f9bf49344da1`
- candidate metrics SHA: `e68595c55cc8bc3a935f87086792330df958627e51eb1f209c27f260e58e15f7`
- PF diagnostics SHA: `a7675f68a015a7177b65697ea6ed5b913d6b66a7e00eaaadf59299490cae4643`
