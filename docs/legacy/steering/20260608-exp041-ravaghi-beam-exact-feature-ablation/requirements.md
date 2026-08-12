# 要件

## 依頼

`ravaghi_beam_exact_feature_ablation` を実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp040_ravaghi_pf_ancc_pfz_feature_ablation` とする。
- 入力 surface は `exp029_public_sel15_pf_oof_feature_generation` の train well の途中以降を隠した疑似 test row artifact を使う。
- Ravaghi exact beam は pseudo cutoff 後の true `TVT` を使わず、visible prefix `TVT_input`、horizontal `GR/MD`、typewell `TVT/GR` だけで再生成する。
- train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) は特徴に使わない。
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は特徴から除外する。
- direct PF/Beam replacement は report control に留め、single LightGBM の add-only feature ablation として評価する。

## 受け入れ基準

- `experiments/exp041_ravaghi_beam_exact_feature_ablation/` に config、settings、notebook、audit script、記録ファイルが揃う。
- exact beam path / diagnostics / disagreement feature family が config で切り分けられている。
- local smoke と静的検証が通る。
- Kaggle full audit 前の状態、実行コマンド、未提出であることが `SESSION_NOTES.md` と `result.md` に記録されている。
