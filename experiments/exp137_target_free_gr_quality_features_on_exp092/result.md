# exp137_target_free_gr_quality_features_on_exp092 結果

## 仮説

GR 値や波形 matching score そのものは使わず、coverage、missing run、interpolation gap、prefix/eval mismatch、native typewell overlap context だけを confidence feature として exp092 系 ML に渡すと、GR が信頼できない regime や typewell context が効く regime を LightGBM が利用できる可能性がある。

## 実装

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- quality context 親: `exp065_typewell_supertype_cluster_cv_audit`
- route: `ml_model`
- 追加特徴量:
  - prefix/eval/full GR missing rate
  - prefix/eval missing run max
  - row-level GR missing flag、missing run length、nearest finite GR gap、finite GR bracket flag
  - prefix/eval GR median shift と robust scale ratio
  - exp065 native overlap / exact hash / shifted NCC / DTW cluster size と multiwell flag
  - native overlap pair count、max exact match rate、max containment

## 結果

Kaggle train v1 は CPU runtime timeout/cancel で未完了。`exp092_full_row_control` は 3 LGBM config すべて完走したが、`target_free_gr_quality_addonly` は `lgb0` / `lgb1` 完了後、`lgb2` の前に止まった。

v2 は `model.training.active_model_indices: [0]` とし、full-row の `exp092_full_row_control` と `target_free_gr_quality_addonly` を `lgb0` のみで比較して完走した。

| variant | model | rows | features | pooled RMSE |
| --- | --- | ---: | ---: | ---: |
| `exp092_full_row_control` | `lgb0` | 3,783,989 | 240 | 9.535793 |
| `target_free_gr_quality_addonly` | `lgb0` | 3,783,989 | 272 | 9.729657 |

add-only は control から +0.193863 悪化した。fold 別では fold1 だけ改善したが、fold0 / fold2 / fold3 / fold4 で悪化し、特に fold3 と fold4 の悪化が大きい。

bucket 別では near-prefix の `000_050` が -0.007979、`050_100` が -0.021365 とわずかに改善した一方、`1000_plus` longtail は +0.215175 悪化した。well 別 delta は中央値 +0.043437、最大悪化 +12.713932、最大改善 -8.368730 で、局所的な救済はあるが悪化 well の尾が重い。

feature importance では `grq_prefix_eval_gr_scale_ratio`、`grq_prefix_eval_gr_median_shift_norm`、`grq_prefix_eval_missing_rate_gap`、`grq_prefix_gr_missing_rate` が使われている。つまり特徴量が完全に無視されたわけではなく、使われた結果として global / longtail を悪化させた。

## 判断

`target_free_gr_quality_addonly` は不採用。inference port、submission、3-config full rerun は行わない。

この結果は、GR coverage / missingness / native-overlap quality を exp092 に素朴に add-only すると near-prefix の微小改善よりも longtail / well-level regression の悪化が勝つ、という負例として扱う。後続で GR 系 confidence feature を試す場合は、全 row への一括 add-only ではなく、near-prefix guard や high-disagreement segment verifier のように適用範囲を限定する。

## 判断基準

global OOF 改善だけでは採用しない。worst-well 悪化、near-prefix bucket、1000+ longtail、exp115 hidden-like stress、raw-test feature parity を確認してから inference port を判断する。
