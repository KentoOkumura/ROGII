# exp165_coordinate_frame_normalization_features_on_exp148

## 状態

Kaggle CPU split train 完了。train-side OOF が exp148 historical `lgb_mean` より悪化したため、推論化・提出はしない。

## 仮説

exp149 で anchor 基準の normalized U/MD shape features は exp092 上で小幅改善し、importance 上位にも入った。現行 ML route anchor の exp148 に対しても、raw MD/X/Y/Z を known-prefix anchor と prefix-tail azimuth で正規化した trajectory confidence features を add-only すれば、exp148 が外れやすい trajectory regime を LightGBM が拾える可能性がある。

## 実装

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 追加: `coordinate_frame_geometry` / `coordinate_frame_direction` / `coordinate_frame_derivative` / `coordinate_frame_interaction`
- 比較: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- 実行: CPU deterministic threads8、split train notebook
- control 再学習: なし

## 実行対象

- `exp165_coordinate_frame_normalization_features_on_exp148_train_lgb0.ipynb`
- `exp165_coordinate_frame_normalization_features_on_exp148_train_lgb1.ipynb`
- `exp165_coordinate_frame_normalization_features_on_exp148_train_lgb2.ipynb`
- `exp165_coordinate_frame_normalization_features_on_exp148_train_aggregate.ipynb`

## 検証方針

GroupKFold by `well`、5 fold、seed 42。各 split notebook は 1 LightGBM config x 5 folds = 5 boosters を学習する。合計は 1 variant、3 LightGBM configs、5 folds、15 boosters。

OOF RMSE、fold 別 RMSE、by-well、near-row bucket、`1000_plus`、feature importance、raw-test/current-test parity を確認する。座標正規化値は direct TVT candidate、hard correction、row-wise selector、blend、postprocess replacement には使わない。

## 所見

3-model `lgb_mean` RMSE は 8.549931602。exp148 historical `lgb_mean` 8.501281182 から +0.048650420 悪化した。

| model | RMSE |
| --- | ---: |
| lgb0 | 8.623039477 |
| lgb1 | 8.586673413 |
| lgb2 | 8.616753590 |
| lgb_mean | 8.549931602 |

coordinate-frame normalization features は exp148 add-only では改善しなかった。direct TVT candidate、hard correction、row-wise selector、postprocess replacement には展開しない。
