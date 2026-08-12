# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `gr_wavelet_rotation_confidence_features_on_exp148`
を `exp218_gr_wavelet_rotation_confidence_features_on_exp148` として実装する。
exp148 の learned-likelihood ML surface に、GR wavelet / rotation denoise 由来の
target-free confidence / uncertainty features を add-only で追加し、LightGBM OOF で評価できる状態にする。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092` とし、control 再学習はしない。
- 参照診断は `exp167_fft_denoised_gr_matching_audit`、`exp189` 系 denoise audit、`exp216_affine_shift_landscape_ruler_readout`、必要に応じて `exp214_public_raw_gr_residual_scale_control` とする。
- DWT/FFT/rolling/Savitzky-Golay 由来の GR signal は予測値や PF/Beam top1 置換に使わず、confidence / uncertainty / candidate disagreement feature に限定する。
- hidden-tail true TVT、oracle best、true-error rank、OOF absolute error を feature source に使わない。
- Kaggle train push 前に active variant 数、LightGBM config 数、fold 数、合計 booster 数を `SESSION_NOTES.md` に記録する。

## 受け入れ基準

- `experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/` に config、train/inference notebook source、補助 `.py`、記録ファイルがある。
- train notebook から、exp072/exp145/exp148 surface を読み、wavelet/rotation confidence feature を生成し、1 enabled variant を学習できる。
- active run plan は 1 enabled variant x 1 mode x 3 LightGBM configs x 5 folds = 15 boostersで、exp148 control は再学習しない。
- Jupytext 変換、`py_compile`、`ruff --select F821,F401`、`validate_experiment.py` が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
