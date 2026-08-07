# exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148

## 状態

completed_negative_no_submit。

既存の `exp191_typewell_late_range_continuity_selector_on_exp176` を親 artifact として使う、exp148 系の replacement-only 実験。ユーザー指定と backlog slug に合わせて同じ `exp191` 番号を含む別ディレクトリとして実装する。

## 仮説

exp191 は exp176 の typewell late-range candidate-error surface を Viterbi で安定化し、path switch を大きく減らした。selected TVT 自体は direct submission に使わないが、selector confidence / segment stability / late-range risk surface としてなら、exp148 の `learned_likelihood_confidence` (`ll_*`) block を置き換えられる可能性がある。

## 検証方針

active variant は `exp191_continuity_selector_confidence_replacement_only` のみ。`projection_correction` と `u_disagreement` は維持し、`learned_likelihood_confidence` は外して `exp191_continuity_selector_confidence` を入れる。

CPU 実行で timeout を避けるため、学習 notebook は `train_lgb0`、`train_lgb1`、`train_lgb2` に分割する。各 split は 1 variant x 1 CPU mode x 1 LGB config x 5 folds = 5 boosters、合計 15 boosters。親/control 再学習はしない。

## 所見

Kaggle CPU split train v1 は `train_lgb0` / `train_lgb1` / `train_lgb2` すべて完了。3 split OOF を align して平均した `lgb_mean_split3` は RMSE TVT 9.321908826 で、exp148 historical `lgb_mean` 8.501281182 から +0.820627644 悪化した。

exp191 continuity selector confidence block は exp145 `learned_likelihood_confidence` (`ll_*`) block の代替にならないと判断し、current-test feature generation、inference port、submit は行わない。raw exp191 selected TVT、selected-minus-exp148、direct replacement、blend、postprocess、hard gate も引き続き対象外。
