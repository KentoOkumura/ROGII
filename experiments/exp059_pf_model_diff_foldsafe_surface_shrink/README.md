# exp059_pf_model_diff_foldsafe_surface_shrink

## 状態

- ルート: `ml_model`
- 状態: submitted complete
- CV: 15.037567 (`lgbm_capacity_pf_model_diff_foldsafe_raw`, original-fold pseudo-test surface)
- Public LB: 11.878
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp058_lgbm_pf_confidence_only_features`
- model anchor parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- seed-bag anchor parent: `exp054_pseudo_tail_seed_bagging_inference_submit`

## 仮説

`exp058` では PF/Beam confidence-only features が same-surface LightGBM
geometry control を大きく改善したが、`PF-vs-exp052/054` 差分がなく
`exp014` 固定 bucket-shrink も surface に合っていなかった。`exp052/054`
相当の fold-out ML anchor と PF/Beam の差分を特徴量化し、さらに alpha を
held-out fold 外で fit する surface-specific shrink に置き換えると、
confidence-only control を上回るかを検証する。

## 検証方針

同一 exp029 pseudo-test surface で、`lgbm_capacity_pf_confidence_only_raw` を
主 reference にし、`lgbm_capacity_pf_model_diff_foldsafe` の raw /
fold-out bucket shrink / confidence-conditioned fold-out bucket shrink を
original-fold と well-hash holdout の両方で比較する。

## 注意

source model diff は submission から読まず、audit split ごとに train-fold wells
だけで exp052/054 相当モデルを再学習して作る。alpha も held-out split の target
を使わない。Kaggle train 実行前のため、現時点では結果数値を持たない。

## 所見

Kaggle train version 3 が完了。選択候補
`lgbm_capacity_pf_model_diff_foldsafe_raw` は original-fold 15.037567、
well-hash 14.735200 で、`lgbm_capacity_pf_confidence_only_raw` から
-0.908111 / -0.834016 改善し、同一 surface の `pf090_hold010`
15.089532 と `public_pf_selector` 15.172636 も全体 RMSE では上回った。

一方、fold-out bucket shrink と confidence-conditioned shrink は raw より悪く、
2500+ rows では `pf090_hold010` / `public_pf_selector` の方がまだ強い。推論 port
する場合は raw 候補だけを対象にし、遠距離 bucket の悪化と prediction range を
必ず確認する。

## 推論

`lgbm_capacity_pf_model_diff_foldsafe_raw` だけを同じ exp059 内で推論化し、
Kaggle inference version 1 が完了した。

- kernel: `kentookumura/exp059-pf-model-diff-infer`
- URL: https://www.kaggle.com/code/kentookumura/exp059-pf-model-diff-infer
- status: completed
- submit-check: PASS
- submit ref: `53549815`
- Public LB: 11.878
- submission SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- public sample changed rows: 0
- public sample branch: `physical_visible` only

Public sample output は exp027 / exp058 と同一で、hidden branch は public sample
では発火しなかった。submit する場合は hidden branch 仮説の code submission
検証として扱い、public output 改善候補とは見なさない。

提出結果は Public LB 11.878。exp058 の 12.778 からは -0.900 改善したが、
exp054 の 11.856 には +0.022 届かなかったため、現時点の ML route LB anchor
にはしない。
