# 要件

## 背景

`exp092_u_projection_correction_disagreement_fullrun` は U-projection correction と disagreement feature による LightGBM surface で、保存済み baseline として `lgb1` CV RMSE 9.322479896 / Public LB 8.350 がある。`exp114_spatial_neighbor_prior_signal_audit` では、well 間の空間近傍 prior が単独補正としては弱い一方、prior value / uncertainty / neighbor quality / candidate disagreement は ML feature として使える可能性が残った。

## 目的

`exp092` の feature surface を維持したまま、`exp114` の spatial neighbor prior を target-free な confidence feature として add-only し、train-side OOF で改善するかを検証する。

## スコープ

- 対象実験: `exp164_spatial_prior_confidence_features_on_exp092_kaggle`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 参照 cache: `exp072_exp063_full_replay_feature_cache`
- spatial prior 参照: `exp114_spatial_neighbor_prior_signal_audit`
- 実行前提: Kaggle Notebook GPU / internet disabled

## 必須要件

- `exp159` の Colab runner、manual upload、checkpoint 再開機構は使わない。
- Kaggle Notebook の train package を生成できること。
- `exp092_full_row_control` は再学習せず、保存済み exp092 metrics を baseline として参照すること。
- 学習対象は `spatial_prior_confidence_addonly` の 1 variant のみとすること。
- LightGBM config は `lgb0`, `lgb1`, `lgb2`、fold は 5 GroupKFold by well とすること。
- GPU train push 前に active variant 数、config 数、fold 数、合計 booster 数、control 再学習の有無を `SESSION_NOTES.md` に記録すること。
- notebook は Kaggle 上で入力確認、設定確認、学習実行、metrics / generated artifacts 確認が追える構成にすること。
- inference / submission は train-side OOF と raw-test feature parity を確認するまで disabled とすること。

## 成功条件

- `py_compile`、notebook JSON validation、`ruff`、`make validate-exp` が通る。
- `make prepare-kaggle-notebooks ... --notebook train ... --strict` が通り、`experiments/exp164.../kaggle/train/kernel-metadata.json` が生成される。
- Kaggle train 実行後に pooled RMSE、fold metrics、by-well、bucket metrics、feature importance、model manifest、prediction / feature SHA を確認できる。

## 非目標

- spatial prior を hard selector、direct correction、oracle candidate choice、postprocess gate として使わない。
- `exp092` control を同じ Kaggle train で再学習しない。
- この段階では `submission.csv` を生成しない。
