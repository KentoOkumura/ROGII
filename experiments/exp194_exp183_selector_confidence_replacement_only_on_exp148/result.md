# exp194_exp183_selector_confidence_replacement_only_on_exp148 結果

## 状態

Kaggle train v1 完了。train-side OOF が exp148 baseline と exp188 add-only の両方より大きく悪化したため、採用しない。inference port / submit には進めない。

## 仮説

exp188 の add-only は exp148 より悪化したが、exp183 selector confidence surface が exp145 learned-likelihood block と重複または競合しているだけなら、`ll_*` block を外した replacement-only で改善する可能性がある。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- route: `ml_model`
- variant: `exp183_selector_confidence_replacement_only`
- active feature groups: `projection_correction`, `u_disagreement`, `exp183_selector_confidence`
- excluded active feature group: `learned_likelihood_confidence`
- baseline: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 学習予定: GPU、3 LightGBM configs x 5 folds = 15 boosters
- control 再学習: なし

## 結果

- Kaggle kernel: `kentookumura/exp194-exp183-selconf-repl-exp148-train` version 1
- status: `COMPLETE`
- output: `experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/kaggle/output/train_v1/`
- rows / wells: 3,783,989 / 773
- features: 272
- feature join coverage: pass、dropped rows 0、dropped wells 0
- elapsed: 15,226.192 sec

| model | pooled RMSE |
| --- | ---: |
| lgb0 | 9.489350463 |
| lgb1 | 9.338468847 |
| lgb2 | 9.306169088 |
| lgb_mean | 9.329893102 |

exp148 `lgb_mean` 8.501281182 から、exp194 `lgb_mean` は +0.828611921 悪化した。exp188 add-only `lgb_mean` 8.539573790 からも +0.790319312 悪化した。

## 解釈

`learned_likelihood_confidence` block を exp183 selector confidence block へ置換すると、exp148 anchor の性能が大きく崩れた。exp188 add-only の小幅悪化よりも大きいため、悪化原因は単なる `ll_*` block との競合ではなく、exp148 の current anchor に対して exp183 selected-path confidence surface が代替情報として弱い、または `ll_*` block の情報量を置換できないことを示す。

exp183 selector confidence の exp148 ML anchor への add-only / replacement-only は閉じる。current-test exp183 selector feature generation、saved-booster inference、submit は実施しない。

## 次

この実験は完了/不採用。後続は exp183 系の追加ではなく、別系統の selector confidence または既存 backlog の DCM / typewell late-range replacement を個別に評価する。
