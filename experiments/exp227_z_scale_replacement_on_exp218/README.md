# exp227_z_scale_replacement_on_exp218

## 状態

Kaggle CPU split train と split OOF 集計は完了。CV が exp218 / exp148 / exp224 add-only より悪いため不採用。inference / submit は行わない。

## 仮説

exp224 の add-only 版は exp218 から +0.062893 RMSE 悪化したが、raw `z` / `dz` / `dzdmd` / `slp_z` と scaled z 系特徴を同居させたことが悪化要因の可能性がある。raw 4 列を model feature list から外し、target-free well-scaled z 系特徴へ置き換えれば、well 差の強い絶対深度・傾き信号をより安定して使えるかを切り分ける。

## 検証方針

- `z_scale_replacement` の 1 variant だけを学習する。
- `drop_base_columns: [z, dz, dzdmd, slp_z]` で raw 4 列を model feature list から削除する。
- 親 exp218 control は再学習しない。保存済み exp218 CV / Public LB を baseline とする。
- CPU 実行とし、タイムアウト対策として `train_lgb0` / `train_lgb1` / `train_lgb2` に分割する。
- 各 split は 1 LightGBM config x 5 folds = 5 boosters。合計 15 boosters。
- feature importance、distance bucket、特に 100-1000 bucket と worst-well regression を確認する。

## 所見

`train_lgb0` / `train_lgb1` / `train_lgb2` は Kaggle version 1 で `COMPLETE`。3-config `lgb_mean` aggregate は RMSE TVT 8.561884247 で、exp218 `lgb_mean` 8.475793752 から +0.086090495 悪化した。exp224 add-only 8.538687042 よりも +0.023197205 悪いため、raw `z` / `dz` / `dzdmd` / `slp_z` を well-scaled z 系へ置き換える仮説は不採用。

## 参照ファイル

- `config.yaml`
- `z_scale_replacement_on_exp218.py`
- `exp227_z_scale_replacement_on_exp218_train.py`
- `exp227_z_scale_replacement_on_exp218_train_lgb0.py`
- `exp227_z_scale_replacement_on_exp218_train_lgb1.py`
- `exp227_z_scale_replacement_on_exp218_train_lgb2.py`
- `aggregate_split_oof.py`
- `exp227_z_scale_replacement_on_exp218_inference.py`
