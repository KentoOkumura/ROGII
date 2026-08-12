# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `denoised_calibrated_matching_features_on_exp148` を `exp190_denoised_calibrated_matching_features_on_exp148` として実装する。GPU 実行で `lgb0`, `lgb1`, `lgb2` に notebook を分ける必要はない。

## 制約

- Route: `ml_model`
- 親は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 control / parent は再学習しない。
- 1 train notebook 内で 1 variant x 1 active mode x 3 LightGBM configs x 5 folds = 15 boosters を実行する。
- direct TVT replacement、hard selector、postprocess blend、submit candidate 化は初期実装に含めない。
- FFT notch は exp167 で弱く、heel calibration は exp170 で不採用のため feature に入れない。
- feature source に hidden-tail true TVT、oracle best、true-error rank、abs error を使わない。
- prefix backtest は観測済み `TVT_input` prefix 内だけで行う。
- 再現性は `docs/06_reproducibility.md` に従い、GPU 学習、Kaggle bootstrap、SHA 記録方針を `SESSION_NOTES.md` と config に残す。

## 受け入れ基準

- `.steering`、`config.yaml`、train/inference notebook script、helper module、README/result/metrics/SESSION_NOTES が exp190 として整合している。
- 追加 feature は exp148 surface に add-only で入る。
- train notebook は lgb config ごとに分割しない。
- Kaggle train 前の active variant/config/fold/booster 数が `SESSION_NOTES.md` に記録されている。
- Jupytext conversion、py_compile、ruff F821/F401、`validate_experiment.py` が通る。
