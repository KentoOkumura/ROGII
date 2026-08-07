# exp164_spatial_prior_confidence_features_on_exp092_kaggle

`spatial_prior_confidence_features_on_exp092` を Kaggle Notebook 前提で作り直した実験。

## 状態

- 状態: Kaggle CPU train v2 complete / negative
- 実行環境: Kaggle Notebook CPU
- route: `ml_model`
- active variant: `spatial_prior_confidence_addonly`
- control 再学習: なし
- 学習規模: 3 notebooks x 1 LightGBM config x 5 folds = 15 boosters
- Kaggle train packages: `train_lgb0` / `train_lgb1` / `train_lgb2` pushed with `run_on_push=true`

## 仮説

`exp092_u_projection_correction_disagreement_fullrun` の U-projection LightGBM surface に、`exp114_spatial_neighbor_prior_signal_audit` の fold-safe spatial neighbor prior を add-only confidence feature として追加する。

spatial prior は直接補正、hard switch、selector candidate としては使わない。prior value、neighbor quality、prior variants disagreement、PF/Beam/likPF との差、near / longtail interaction を LightGBM に渡し、exp092 が外れやすい regime を補助的に表現できるかを見る。

## 検証方針

exp092 historical baseline (`lgb1` CV 9.322479896 / Public LB 8.350) と比較し、global OOF だけでなく by-well、distance bucket、near prefix、longtail、feature importance を確認する。train-side positive でも、raw-test/full-train parity と hidden-like stress を確認するまでは inference / submit へ進めない。

## 入力

- exp072 full replay feature cache
- exp114 spatial neighbor prior OOF predictions
- raw train wells for prefix anchor recovery

## 所見

Kaggle-first notebook として実装済み。`exp159` の Colab runner / manual upload / checkpoint 再開機構は採用しない。

CPU 分割 train kernel:

- `kentookumura/exp164-spatial-prior-conf-exp092-lgb0-train`
- `kentookumura/exp164-spatial-prior-conf-exp092-lgb1-train`
- `kentookumura/exp164-spatial-prior-conf-exp092-lgb2-train`

2026-07-02 時点で 3 本とも version 2 が COMPLETE。version 1 は kernelspec 不足で Papermill 起動前に失敗したため、`--set-kernel python3` で notebook を再生成して再 push した。

CV は `lgb0` 9.660879008、`lgb1` 9.429441976、`lgb2` 9.415444308。最良の `lgb2` でも exp092 baseline 9.322479896 から +0.092964412 悪化したため、採用しない。
