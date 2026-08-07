# exp163_typewell_neighbor_prior_as_ml_features_on_exp148 結果

## 状態

Kaggle CPU split train 完了。train-side OOF は exp148 historical `lgb_mean` より悪化したため、推論化・提出はしない。

## 評価設計

exp148 の既存 feature surface と learned likelihood confidence features を固定し、fold-safe typewell neighbor prior features だけを add-only する。control 再学習は行わず、保存済み exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960 を比較基準にする。

## 実行単位

タイムアウト対策として `lgb0`、`lgb1`、`lgb2` を別 notebook で実行する。各 notebook は 5 folds、計 5 boosters を学習する。

## 結果

Kaggle train v1 は `lgb0` / `lgb1` / `lgb2` の 3 notebook すべて COMPLETE。

| model | pooled RMSE |
| --- | ---: |
| lgb0 | 8.575290758 |
| lgb1 | 8.572174727 |
| lgb2 | 8.571366316 |
| lgb_mean | 8.519739843 |

exp148 historical `lgb_mean` 8.501281182 に対して、exp163 `lgb_mean` は +0.018458661 悪化。typewell prior 系 feature は重要度上位にも入るが、add-only では既存 exp148 surface の汎化を改善しなかった。

## 判断

train-side rejected。inference port / submit は行わない。typewell neighbor prior は direct correction としても ML feature としても worst-well risk が残るため、同方向は追加で広げない。
