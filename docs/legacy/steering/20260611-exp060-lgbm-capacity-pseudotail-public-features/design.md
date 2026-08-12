# 設計

## アプローチ

`exp056` の multi-cutoff feature artifact を入力にし、`exp051/052` の
LightGBM capacity pseudo-tail 設定を固定して、public notebook 由来 feature
family を追加する。audit script は exp059 の cross-fit skeleton を再利用するが、
exp052/054 foldout source prediction はこの実験では使わない。

Feature family:

- `geometry_anchor_context`: cutoff、row index、distance、trajectory、GR availability、last anchor。
- `public_pf_prediction`: PF prediction と scale/seed prediction。
- `public_pf_uncertainty`: PF likelihood、entropy、effective particles、seed/scale spread。
- `public_beam_disagreement`: Beam prediction、spread、PF/Beam disagreement。
- `ncc_gr_match_minimal`: exp043/048 系の NCC/GR match subset。
- `spatial_context`: cutoff からの geometry context。

Variants:

- 0.65-only geometry control。
- 0.65-only public PF core。
- multi-cutoff equal budget public PF core。
- 0.65 preserve + augmentation public PF core。
- multi-cutoff NCC/GR + PF context。
- multi-cutoff NCC/GR + PF context + spatial context。

## 実験範囲

- 対象実験: `exp060_lgbm_capacity_pseudotail_public_features`
- Route: `ml_model`
- 親実験: `exp056_public_sel15_pf_oof_multicutoff_artifact`
- 変更する変数: public feature family、multi-cutoff training policy
- 固定する変数: LightGBM capacity params、residual shrink、well-level original-fold / well-hash audits、exp014 bucket shrink candidate

## リスク

- リークリスク: exp056 の pseudo-test masking 後 rows だけを使い、target は label/scoring のみに使う。
- CV/LB 不一致リスク: exp056 pseudo-test surface RMSE は exp052/054 Public LB と直接混ぜない。
- ランタイム/メモリリスク: exp056 artifact は 5,499,624 rows / 783MB gzip なので Kaggle train 実行を正にする。NCC/GR match regeneration は重いため full run は Kaggle 上で確認する。
