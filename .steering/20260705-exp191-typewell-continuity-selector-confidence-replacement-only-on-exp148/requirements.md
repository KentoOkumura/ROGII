# 要件

## 依頼

`exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` backlog を実装する。CPU 実行の timeout 対策として、学習コードは `lgb0`、`lgb1`、`lgb2` の split notebook に分ける。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- upstream continuity selector: `exp191_typewell_late_range_continuity_selector_on_exp176`
- upstream candidate ranker: `exp176_typewell_late_range_pfbeam_candidate_prior`
- `projection_correction` と `u_disagreement` は維持する。
- `learned_likelihood_confidence` (`ll_*`) は active feature set から外す。
- 初回 variant では raw `tlr191_selected_tvt`、selected-minus-exp148、direct replacement、blend、postprocess、hard gate、submit をしない。
- Parent/control 再学習はしない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream artifact、feature schema、model manifest、prediction SHA の記録方針を config / notes に残す。

## 受け入れ基準

- `experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/` に config、helper、正規 train/inference notebook、`train_lgb0/1/2` notebook 起点がある。
- active variant は 1 個で、CPU mode x 3 LGB configs x 5 folds = 15 boosters として記録されている。
- `model.feature_ablation.active_variants` で `learned_likelihood_confidence` が disabled control のみに残り、active variant には入っていない。
- exp191 OOF selected path から `true_tvt`、`abs_error`、`oracle_candidate`、`oracle_label` を downstream feature に使わない。
- `jupytext` 変換、`py_compile`、`ruff --select F821`、`validate-exp` が通る。
