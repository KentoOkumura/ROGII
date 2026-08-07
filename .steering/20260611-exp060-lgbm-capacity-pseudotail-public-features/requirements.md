# 要件

## 依頼

`lgbm_capacity_pseudotail_public_features` を実装する。実験親は
`exp056_public_sel15_pf_oof_multicutoff_artifact` とし、exp056 の
multi-cutoff public sel15 PF/Beam feature artifact を、exp051/052 の
LightGBM capacity pseudo-tail residual model に接続する。

## 制約

- Route: `ml_model`
- 親実験は `exp056_public_sel15_pf_oof_multicutoff_artifact`。
- exp059 は実装再利用元としてのみ扱い、lineage parent にはしない。
- Public PF/Beam は direct replacement ではなく add-only model features として使う。
- 0.65-only control と multi-cutoff augmentation を分けて比較する。
- train-only formation columns は使わない。
- Kaggle Notebook 実行を正とし、ローカル full run はしない。

## 受け入れ基準

- `experiments/exp060_lgbm_capacity_pseudotail_public_features/` に config、settings、train/inference notebook、audit script、記録ファイルがある。
- `config.yaml` の `lineage.parent` が exp056 である。
- train notebook が exp056 feature artifact を読み、public feature variants を cross-fit audit できる。
- static checks と `scripts/validate_experiment.py --experiment exp060_lgbm_capacity_pseudotail_public_features` が通る。
