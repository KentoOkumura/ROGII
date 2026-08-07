# exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148 結果

## 現状

2026-07-05: Kaggle CPU split train v1 は `train_lgb0` / `train_lgb1` / `train_lgb2` すべて `KernelWorkerStatus.COMPLETE`。出力を取得し、3 split OOF prediction を `id` / `well` / `target_tvt` で align して `lgb_mean_split3` を集計した。

## 実装内容

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- upstream selector: `exp191_typewell_late_range_continuity_selector_on_exp176`
- upstream ranker: `exp176_typewell_late_range_pfbeam_candidate_prior`
- variant: `exp191_continuity_selector_confidence_replacement_only`
- active feature groups: `projection_correction`, `u_disagreement`, `exp191_continuity_selector_confidence`
- removed feature group: `learned_likelihood_confidence`
- runtime: CPU split `train_lgb0` / `train_lgb1` / `train_lgb2`

## 判定

completed_negative_no_submit。

`lgb_mean_split3` は RMSE TVT 9.321908826、RMSE target 9.321909300。exp148 historical `lgb_mean` 8.501281182 から +0.820627644、exp193 `lgb_mean` 8.456665439 から +0.865243388 悪化した。exp194 replacement-only `lgb_mean` 9.329893102 よりは -0.007984276 良いが、exp148 の `learned_likelihood_confidence` (`ll_*`) block を置き換えるには大きく不足する。

split 別 RMSE TVT:

- `lgb0`: 9.464292702
- `lgb1`: 9.331742862
- `lgb2`: 9.313152706
- `lgb_mean_split3`: 9.321908826

near bucket `000_050` は RMSE 1.133298942、`1000_plus` は 10.230017421。worst well top は `86454a6f` RMSE 56.312288515。replacement-only として不採用にし、current-test feature generation、inference port、submit は行わない。
