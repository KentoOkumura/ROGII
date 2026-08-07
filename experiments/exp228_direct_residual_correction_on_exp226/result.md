# exp228_direct_residual_correction_on_exp226 結果

## 状態

- Kaggle train split 3 本と `train_aggregate` v1 完了。
- Kaggle inference: 未実行。
- Public LB: 未提出。
- 判断: 不採用。inference / submit は行わない。

## 実装要約

exp218 と同じ特徴量面を使い、target だけ `TVT - exp226_oof_pred` に変更した direct residual correction 実験。推論では split train の LightGBM booster を平均し、exp226 inference v1 の `submission.csv` を base として `exp226_pred + residual_pred` を出す。

## 実行計画

- `train_lgb0`: CPU / 1 config / 5 folds / 5 boosters
- `train_lgb1`: CPU / 1 config / 5 folds / 5 boosters
- `train_lgb2`: CPU / 1 config / 5 folds / 5 boosters
- `train_aggregate`: 学習なし。3 split OOF を平均して `lgb_mean` CV を集計
- `inference`: 3 split の saved boosters と exp226 inference submission を使う

## 判断

`lgb0/lgb1/lgb2` の split OOF mean は RMSE 8.944085501。exp226 K16 fallback CV 9.427109597 からは -0.483024096 改善したが、同じ ML feature surface の親である exp218 CV 8.475793752 から +0.468291749 悪化した。

この実験の目的は exp226 fallback を direct residual correction で救うことだったが、現行の exp218 ML anchor に届かない。CV の時点で弱いため、inference / submit には進めない。

失敗原因としては、exp226 fallback 自体の誤差構造が exp218 feature surface の LightGBM で直接足し戻すには粗く、強い ML anchor を置き換えるほどの信号になっていない可能性が高い。exp226 を使う場合は、直接補正ではなく「fallback が危険な well / bucket を示す confidence readout」や selector 補助特徴に限定する。

## Kaggle 結果

| model | RMSE TVT |
| --- | ---: |
| lgb0 | 9.042170562 |
| lgb1 | 8.940004291 |
| lgb2 | 8.940366893 |
| lgb_mean_from_split_lgb0_lgb1_lgb2 | 8.944085501 |

- rows: 3,783,989
- wells: 773
- aggregate prediction SHA256: `239d53622af7cf3f3b421522de9a1f9cdda0a6ac3b99ba283dc8796032209da8`
- split train はすべて CPU 実行。親 exp226 / control の再学習なし。

## 次アクション

- exp228 の inference / submit はしない。
- exp226 系を再利用するなら、prediction を直接補正対象にせず、exp226 residual / disagreement / bucket risk の readout に限定する。
