# exp262 selector LightGBM extra-trees ablation 結果

## 状態

Kaggle train v1完了。selector guard不通過のため不採用。downstream再学習、raw-test inference、submissionは未実行。

## 仮説

exp238 nested selectorのLightGBMへ`extra_trees=True`だけを追加することで、候補誤差scoreの汎化またはhistorical selectorとの補完性が改善するかを検証した。

## 設定

- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`。
- 変更: selector LightGBMの`extra_trees=True`のみ。
- 固定: 11候補、184 context、candidate-long 3列、outer 5 × inner 4、bounded sampling、objective、seed、fixed Viterbi。
- control: 保存済みexp238 nested OOF score / model manifest。再学習0。
- 実行量: 1 variant × 1 config × outer 5 × inner 4 = 20 CPU boosters。
- downstream exp218 LightGBM: 0 boosters。
- kernel: `kentookumura/exp262-sel-extra-trees-exp238-train` version 1 / `id_no=127468598`。
- runtime: 18,760.043秒（約5時間13分）、CPU、internet off、status `COMPLETE`。

## 結果

| 指標 | historical exp238 | extra-trees | 差 | guard |
| --- | ---: | ---: | ---: | --- |
| candidate error MAE | 4.532912 | 4.748342 | +0.215430 | FAIL |
| oracle-candidate logloss | 2.957410 | 3.025668 | +0.068258 | FAIL |
| pairwise rank accuracy | 0.792762 | 0.783041 | -0.009721 | FAIL |
| fixed top1 global RMSE | 8.512262 | 8.840332 | +0.328071 | FAIL |
| fixed Viterbi global RMSE | 8.492559 | 8.826521 | +0.333962 | FAIL |
| fixed Viterbi 000-050 RMSE | 0.667616 | 0.627142 | -0.040474 | PASS |
| fixed Viterbi 1000+ RMSE | 9.314887 | 9.694635 | +0.379748 | FAIL |
| fixed Viterbi exp115 spatial RMSE | 8.821805 | 9.620149 | +0.798344 | FAIL |
| fixed Viterbi exp115 typewell-purged RMSE | 8.778809 | 9.556358 | +0.777549 | FAIL |

- fixed Viterbi outer-fold差: `+0.677187 / +0.258589 / -0.092081 / +0.219028 / +0.615652`。nonworseは1/5 folds。
- current-vs-historical最大well回帰: `a959858c`、fixed Viterbi +12.835997 ft。
- known worst vs likpf: historical top1 +37.680899、current top1 +37.374155。既知worstの拡大だけは回避した。
- score surface差: flat Pearson 0.998484、rowwise Spearman 0.897034、top1 agreement 0.429939、mean absolute score difference 1.493698。
- best iteration: mean 1,172.4、median 1,199、min 914、max 1,200。20/20 modelsで187 featuresと`extra_trees=True`を確認した。
- Public / Private LB: 未提出。

## 採用判断

`score_surface_changed`は通り、nearだけは改善した。しかしcandidate error MAE、logloss、ranking、global top1/Viterbi、1000+、hidden-like 2面、4/5 folds、worst-well stabilityが悪化したためguardは不通過。`downstream_retraining_allowed=false`、`rawtest_inference_allowed=false`、`submission_allowed=false`を維持する。

実装、fold、input、model、SHA契約はすべて通っている。したがって実装失敗ではなく、random thresholdを加えたextra-treesがこの固定candidate-error surfaceのscore calibrationとranking汎化を損ねたnegative resultと判断する。iteration上限付近のmodelが多いが、tree数延長やparameter gridでの救済は単一parameter ablationを崩すため行わない。

## 再現性

- rows / wells: 3,783,989 / 773。
- selector feature content SHA: `096b6788fd915b98de4d4a3035f274415ddc8661f6d0021f8714902bfff6299d`。
- selector model manifest SHA: `5e0dbc5f3f1e6b76b214e24a3e523f1aaff03019f82b8fd20e2c7c51ec851807`。
- fixed selection prediction decompressed SHA: `ac3cc0bfa8d132641453f7f404f829ef3d9e2abc63e7151270b8a89c2363f41a`。
- 主要11 files raw SHA、5 nested score raw/decompressed SHA、20 model SHAはKaggle output実体と全件一致した。
- 同一kernel rerunのSHA一致は未確認なのでdeterministic anchorとは扱わない。

## 次

selector extra-trees branchを閉じる。tree / leaf / sampling / temperature / threshold grid、downstream 15 boosters、raw-test inference、submissionへ進まない。

次のselector routeは既存高優先の`exp264_exp263_candidate_confidence_dual_selector`を維持する。exp264 Stage Bはstandard LightGBM固定で、raw-test-ready candidate confidenceとdual objectiveにより今回のrandom-threshold失敗を回避する。新規backlogは追加しない。
