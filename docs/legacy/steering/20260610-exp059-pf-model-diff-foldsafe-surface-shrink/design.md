# 設計

## アプローチ

`exp058` の LightGBM capacity paired-control audit を親にし、同一 exp029 pseudo-test surface 上で次を追加する。

1. `exp052` capacity pseudo-tail と `exp054` seed-bag pseudo-tail の設定を読み、audit split ごとに train-fold wells だけで source model を再学習する。
2. validation-fold wells の exp029 pseudo-test rows に source model を当て、`exp052_foldout` / `exp054_foldout` 予測を作る。
3. `PF-vs-exp052/054`、`Beam-vs-exp052/054`、`exp054-vs-exp052`、最小 PF-model diff、seed spread を特徴量化する。
4. 候補 model は geometry control、exp058 相当の confidence-only control、model-diff 追加 candidate に限定する。
5. postprocess は raw、fold-out bucket shrink、confidence-conditioned fold-out bucket shrink に限定する。alpha は held-out split 以外の rows だけで fit する。

## 実験範囲

- 対象実験: `exp059_pf_model_diff_foldsafe_surface_shrink`
- Route: `ml_model`
- 親実験: `exp058_lgbm_pf_confidence_only_features`
- 変更する変数:
  - fold-out `exp052/054` source prediction features
  - model-diff feature family
  - surface-specific fold-out shrink
- 固定する変数:
  - exp029 pseudo-test row surface
  - exp051 LightGBM capacity params
  - exp058 geometry / confidence feature controls
  - well-level original-fold and well-hash holdout audits

## リスク

- リークリスク: source prediction と alpha fit が validation fold target を見ないよう、split ごとに生成する。
- CV/LB 不一致リスク: exp029 pseudo-test surface は hidden code branch の代理評価なので、exp052/054 Public LB とは同一基準に混ぜない。
- ランタイム/メモリリスク: exp052 と exp054 source model を各 audit split で再学習するため exp058 より重い。Kaggle train 実行を正とし、local full run はしない。
