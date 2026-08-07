# 要件

## 依頼

KAGGLE_DIRECTION backlog `exp226_direct_residual_correction` を実装する。特徴量は exp218 と同じ surface を使い、target を exp226 K16 fallback の group-safe OOF 残差 `TVT - exp226_oof_pred` に変更する。実行は CPU とし、タイムアウト対策のため LightGBM 学習コードは `lgb0`, `lgb1`, `lgb2` に分ける。

## 制約

- Route: `ensemble`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- ML feature surface: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- train residual は exp226 group-safe OOF prediction からのみ作る。
- full-train exp226 prediction で train residual を作らない。
- CPU-only Kaggle notebook とし、GPU は使わない。
- `lgb0/lgb1/lgb2` は別 notebook / 別 Kaggle kernel で学習する。
- control / parent retraining はしない。
- submit は split OOF aggregate と stress readout 後に限定する。

## 受け入れ基準

- `experiments/exp228_direct_residual_correction_on_exp226/` に config、settings、train/inference notebook source、記録ファイルがある。
- `train_lgb0`, `train_lgb1`, `train_lgb2` がそれぞれ 1 LightGBM config x 5 folds だけを実行する。
- `train_aggregate` が3 split の OOF predictions を平均して `lgb_mean` CV を作れる。
- inference が3 split の saved boosters を読み、exp226 inference submission に residual を加える。
- `config.yaml` に route、lineage、CPU runtime、split plan、leakage policy が明記されている。
- `SESSION_NOTES.md` に push 前の variant/config/fold/booster 数と parent/control retraining なしが記録されている。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
