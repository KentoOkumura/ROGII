# exp163_typewell_neighbor_prior_as_ml_features_on_exp148

## 状態

Kaggle CPU split train 完了。OOF が exp148 historical baseline より悪化したため、推論化・提出は行わない。

## 仮説

exp109/120 で typewell neighbor prior は直接補正としては worst-well regression が残ったが、longtail / high-drift の bias と不確実性を示す特徴量としては exp148 LightGBM に有益な可能性がある。

## 実装

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 追加: fold-safe native typewell-overlap prior の value / quality / interaction / clipped correction proxy
- 比較: exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960
- 実行: CPU deterministic threads8、split train notebook

## 実行対象

- `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb0.ipynb`
- `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb1.ipynb`
- `exp163_typewell_neighbor_prior_as_ml_features_on_exp148_train_lgb2.ipynb`

## 検証方針

- GroupKFold by `well`、5 fold、seed 42。
- exp148 historical `lgb_mean` CV 8.501281182 / Public LB 7.960 を baseline とし、control は再学習しない。
- `lgb0` / `lgb1` / `lgb2` を別 notebook で CPU 実行し、各 split 5 boosters、合計 15 boosters として timeout risk を下げる。
- OOF RMSE、fold 別 RMSE、by-well、near-row bucket、feature importance で add-only feature の有効性を確認する。

## 所見

- Kaggle train v1 は `lgb0` / `lgb1` / `lgb2` すべて COMPLETE。
- 3-model `lgb_mean` RMSE は 8.519739843。exp148 historical `lgb_mean` 8.501281182 から +0.018458661 悪化。
- typewell prior は direct selector / soft average / blend / postprocess replacement には使わず、ML feature としてのみ使う。
- train-side rejected。推論 notebook は実装しない。
