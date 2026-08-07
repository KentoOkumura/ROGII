# exp188_exp183_selector_confidence_addonly_on_exp148

## 目的

exp183 の cluster-outlier prior 入り selector selected path が、exp148 の ML anchor に add-only confidence feature として有効かを評価する。

## 状態

実装中。Kaggle train 未実行。

## 仮説

exp183 は exp158 continuity selector を train-side で改善した。exp148 の現行 selector/confidence feature は exp145 learned likelihood 系なので、exp183 の選択候補、path stability、candidate disagreement が別系統の信号として LightGBM に効く可能性がある。

## 変更点

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- route: `ml_model`
- variant: `exp183_selector_confidence_addonly`
- control: 再学習しない。exp148 CV / Public LB を historical baseline として参照する。

exp183 selected TVT は直接置換、blend、postprocess、hard gate、submit candidate として使わない。

## 検証方針

GroupKFold 5 folds を well group で実行し、GPU LightGBM family 3 configs を単一 train notebook 内で学習する。予定は 1 variant、3 configs、5 folds、合計 15 boosters。

比較基準:

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp160 `lgb_mean` CV 8.463718773783008 / Public LB 8.061
- exp183 best Viterbi RMSE 10.601481774

## 所見

未実行。train CV が positive でも、raw-test/current-test exp183 selector feature parity が未実装のため、そのまま submit しない。
