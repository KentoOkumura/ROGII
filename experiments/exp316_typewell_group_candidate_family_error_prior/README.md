# exp316_typewell_group_candidate_family_error_prior

## 状態

- ルート: `ml_model`
- 状態: 設計確定・exp313/315待ち・未実装
- 親: `exp315_typewell_group_candidate_likelihood_rank_features`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

同じType Well群では、物理candidate familyごとの得手不得手に再現性がある。outer-train truthでgroup×family error priorをcross-fitし、hard routerではなくnested selectorのsoft補助だけにする。

## 検証方針

Stage Aは0-model family-rank readout。PASS時だけStage Bで40 selector models。順位相関、4/5 folds、親比gain、hidden-like、worst guardを全PASSするまで利用不可。

## 所見

family priorはsoft featureに限定し、群ごとのhard ruleにはしない。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`で学習/push/runは禁止。
