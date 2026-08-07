# exp227_z_scale_replacement_on_exp218 結果

## 状態

Kaggle CPU split train と split OOF 集計は完了。CV が比較基準より悪いため不採用。inference / submit は行わない。

## 仮説

exp218 の raw `z` / `dz` / `dzdmd` / `slp_z` 4 列を、exp224 で作った target-free well-scaled z 系特徴に置き換える。exp224 add-only は exp218 比 +0.062893 RMSE 悪化済みのため、raw 4 列を残す再試行ではなく replacement-only の切り分けにした。

## 設定

- route: `ml_model`
- parent: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- active variant: `z_scale_replacement`
- dropped base columns: `z`, `dz`, `dzdmd`, `slp_z`
- runtime: CPU
- train split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- control retraining: なし
- boosters: 各 split 1 LightGBM config x 5 folds = 5 boosters、合計 15 boosters

## Kaggle 実行

- `train_lgb0`: `kentookumura/exp227-zscale-exp218-lgb0` v1 / `COMPLETE`
- `train_lgb1`: `kentookumura/exp227-zscale-exp218-lgb1` v1 / `COMPLETE`
- `train_lgb2`: `kentookumura/exp227-zscale-exp218-lgb2` v1 / `COMPLETE`

## CV

| model | RMSE TVT | RMSE target |
| --- | ---: | ---: |
| `lgb0` | 8.665460921 | 8.665460904 |
| `lgb1` | 8.578969303 | 8.578969203 |
| `lgb2` | 8.599021781 | 8.599021937 |
| `lgb_mean` | 8.561884247 | 8.561884364 |

3-config `lgb_mean` は exp218 parent 8.475793752 から +0.086090495、exp148 feature surface 8.501281182 から +0.060603065、exp224 add-only 8.538687042 から +0.023197205 悪化。

## Readout

- coverage: 3,783,989 rows / 773 wells、3 split とも pass
- prediction SHA256: `38f7632b3e0442d1e667bff6b866cc3613775826e01ce22d76c6f395dac7c460`
- top importance: `slp_b_d_50`, `spatial_knn_dist`, `wsz_dz_over_likpf_tvt_p05_p95_range`, `ll_learned_pred_abs_error_beam_mean`, `grwr_fft_rotation_ratio_x_log1p_md_since`
- worst wells top3: `86454a6f` 48.296673、`1b1eba53` 45.455573、`fb03ae90` 45.418222

## 判定

`z_scale_replacement` は exp218 / exp148 / exp224 add-only のすべてより悪く、不採用。raw z 系を well-scaled z 系に置き換える方向は、少なくともこの LightGBM surface では改善しない。saved-booster aggregate manifest、inference port、submit は行わない。
