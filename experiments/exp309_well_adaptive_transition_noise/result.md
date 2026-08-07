# exp309_well_adaptive_transition_noise 結果

## 状態

実装・静的/synthetic検証完了。upstream exp307 chain FAILにより未実行のまま閉鎖。

## 設定

- parent: exp308
- change: well別`sig_r`だけ
- formula: robust prefix rate innovation、support shrink、clip 0.001--0.004
- fixed observation: exp307 finite-MAD `σ_GR` + exp308 missing-distance confidence
- execution plan: 1 variant、773 HMM runs、0 booster、parent/control再実行0

## 実装結果

self-contained train/inference source、正規Notebook、9件のexp309 contract testを実装した。親promotion status、prediction SHA、parent/direct/blend metricsが未確定のため、Kaggle実行入口はfail-closedのままである。

## 精度結果

CV、LB、full prediction、submissionはない。Kaggle package、push、Notebook実行も行っていない。

## 次

exp308がparent promotionに到達しないため、Kaggle実行、inference、submissionを行わない。
