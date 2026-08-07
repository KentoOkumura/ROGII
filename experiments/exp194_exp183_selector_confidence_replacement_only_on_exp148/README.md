# exp194_exp183_selector_confidence_replacement_only_on_exp148

## 目的

exp188 の add-only では悪化した exp183 selector confidence surface を、exp148 の既存 `ll_*` learned-likelihood block と置換した場合に改善するかを確認する。

## 状態

- route: `ml_model`
- status: `train_completed_not_selected`
- CV: `lgb_mean` 9.329893102（Kaggle train v1）
- Public LB: 未提出
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`

## 仮説

exp188 の add-only は exp148 より悪化したが、exp183 selector confidence surface が exp145 learned-likelihood block と重複または競合しているだけなら、`ll_*` block を外した replacement-only で改善する可能性がある。

## 変更点

active variant は `exp183_selector_confidence_replacement_only` の 1 つだけ。`projection_correction` と `u_disagreement` は維持し、`learned_likelihood_confidence` を外して `exp183_selector_confidence` に置換する。

exp183 selected TVT は直接置換、blend、postprocess、hard gate、submission 候補としては使わない。

## 検証方針

GroupKFold 5 folds を well group で実行し、GPU LightGBM family 3 configs を単一 train notebook 内で学習する。予定は 1 active variant、3 configs、5 folds、合計 15 boosters。control / parent 再学習はしない。

比較基準:

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp188 add-only `lgb_mean` CV 8.539573790

## 実行入口

- 学習 notebook: `exp194_exp183_selector_confidence_replacement_only_on_exp148_train.ipynb`
- 推論 notebook: `exp194_exp183_selector_confidence_replacement_only_on_exp148_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148`

notebook 実行は Kaggle kernel run を正とする。初期実装では inference / submit は対象外。

## 所見

Kaggle train v1 は完了。rows / wells は 3,783,989 / 773、features は 272、feature join coverage は pass。pooled OOF は `lgb0` 9.489350463、`lgb1` 9.338468847、`lgb2` 9.306169088、`lgb_mean` 9.329893102。

exp148 `lgb_mean` 8.501281182 から +0.828611921、exp188 add-only 8.539573790 から +0.790319312 悪化したため、不採用。current-test feature port、inference、submit には進めない。
