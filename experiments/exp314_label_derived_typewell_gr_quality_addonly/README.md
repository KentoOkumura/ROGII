# exp314_label_derived_typewell_gr_quality_addonly

## 状態

- ルート: `ml_model`
- 状態: 設計確定・exp311/313待ち・未実装
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- CV / LB: 未実行 / 未提出

## 仮説と変更点

outer-train suffix labelsから得た群別support/noise/reliabilityは、calibrated GRを直接使わなくてもMLがGR信用度を判断する補助になる。exp148へ6列だけadd-onlyし、saved controlは再学習しない。

## 検証方針

1 variant × 3 configs × 5 folds = 15 boosters。CV gain、4/5 folds、全距離帯、hidden-like、worst guardを全PASSするまでinference不可。15 boosters実行は別承認が必要。

## 所見

label-derived priorはdirect correctionではなく、品質特徴としてだけ使う。

## 実行入口

notebookは未実装placeholder。`implementation.enabled=false`で学習/push/runは禁止。
