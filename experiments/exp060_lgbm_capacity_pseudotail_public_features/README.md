# exp060_lgbm_capacity_pseudotail_public_features

## 状態

- ルート: `ml_model`
- 状態: submitted_complete
- CV: 15.562057 (`lgbm_capacity_public_core_spatial_multicutoff_raw`, original-fold exp056 pseudo-test surface)
- Public LB: 12.046
- Private LB: -
- Submit ID: 53581051
- 作成日: 2026-06-11
- 親実験: `exp056_public_sel15_pf_oof_multicutoff_artifact`
- model anchor parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- inference anchor parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- implementation source: `exp059_pf_model_diff_foldsafe_surface_shrink`

## 仮説

`exp056` の multi-cutoff public sel15 PF/Beam 生成物を使い、`exp051/052`
LightGBM capacity pseudo-tail residual model に public notebook 由来特徴を
追加すると、0.65-only geometry control より改善する可能性がある。

## 検証方針

exp056 の 0.45 / 0.65 / 0.82 cutoff rows を入力にし、original-fold と
well-hash holdout で cross-fit する。比較は 0.65 geometry control、
0.65 public PF core、multi-cutoff equal-budget、0.65 を優先保持する
multi-cutoff augmentation、NCC/GR match minimal subset、spatial context
追加に分ける。

## 所見

Kaggle train version 1 が完了。5,499,624 rows / 773 wells の exp056
multi-cutoff surface で、best direct control は `pf090_hold010` の
15.023697 / 15.023697。public-feature model の最良は
`lgbm_capacity_public_core_spatial_multicutoff_raw` で 15.562057 /
15.731138。

この候補は 0.65 geometry control から -3.339331 / -3.203251 改善し、
NCC/GR + PF context paired control からも -0.221582 / -0.064659 改善した。
ML route の train-side candidate としては positive。

`pf090_hold010` と `public_pf_selector` は direct PF 診断 control であり、
この ML route 実験の採用判定条件ではない。

Inference version 1 と code submission も完了。`ref=53581051` の Public LB は
12.046 で、exp052 12.076 からは -0.030 改善したが、exp054 11.856、exp061
11.826、exp039 11.740、exp027 8.781 には届かない。したがって ML route /
pseudo-tail Public LB anchor は更新しない。
