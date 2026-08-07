# exp224_well_scaled_z_dz_features_on_exp218

## 状態

Kaggle CPU split train 完了。不採用。親実験は `exp218_gr_wavelet_rotation_confidence_features_on_exp148`。

## 仮説

exp218 は Public LB 7.843 の ML submitted anchor だが、100-1000 bucket と一部 worst wells に悪化が残る。well 内で `z` / `dz` / `dzdmd` / `slp_z` を robust scale した特徴は、絶対深度・傾きの well 差を抑え、exp218 の GRWR confidence とは別の形状・レンジ情報を LightGBM に渡せる可能性がある。

## 検証方針

- `well_scaled_z_dz_addonly` の 1 variant だけを学習する。
- 親 exp218 control は再学習しない。保存済み exp218 CV / Public LB を baseline とする。
- Kaggle GPU 枯渇対策として CPU 実行にする。
- タイムアウト対策として `train_lgb0` / `train_lgb1` / `train_lgb2` に分割する。
- 各 split は 1 LightGBM config x 5 folds = 5 boosters。合計 15 boosters。
- feature importance、distance bucket、特に 100-1000 bucket と worst-well regression を確認する。

## 所見

2026-07-09 に `train_lgb0` / `train_lgb1` / `train_lgb2` を Kaggle に push し、3 本とも `COMPLETE`。split OOF を取得して 3-config `lgb_mean` を集計した。

- lgb0: 8.683606336
- lgb1: 8.573438105
- lgb2: 8.534973570
- 3-config `lgb_mean`: 8.538687042
- exp218 parent `lgb_mean`: 8.475793752。差分 +0.062893290。

CV 悪化のため、inference / submit には進めない。

## 参照ファイル

- `config.yaml`
- `well_scaled_z_dz_features_on_exp218.py`
- `exp224_well_scaled_z_dz_features_on_exp218_train.py`
- `exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py`
- `exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py`
- `exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py`
- `exp224_well_scaled_z_dz_features_on_exp218_inference.py`
