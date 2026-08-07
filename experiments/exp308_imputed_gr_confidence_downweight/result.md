# exp308_imputed_gr_confidence_downweight 結果

## 状態

実装・静的/synthetic検証完了。exp307 promotion gate FAILにより未実行のまま閉鎖。

## 設定

- parent: exp307 finite-MAD exact HMM
- change: missing-distance emission weightだけ
- formula: observed 1、missing `max(0.25,2^(-d/8))`
- execution: 1 variant、773 HMM runs、0 booster

## 結果

CV、LB、full prediction、submissionはない。self-contained train/inference source、正規Notebook、12件のexp308 contract testを実装し、親dependency pending時のfail-closeを確認した。

## 解釈

最近傍finite距離、soft floor、parent interpolation/scale固定、weighted emission、truth late join、gap/distance readoutを実行可能な形にした。精度証拠はまだなく、exp307がFAILした場合は本実験を実行しない。

## 次

exp307 PASSを必須とした事前条件が成立しないため、Kaggle実行、inference、submissionを行わない。
