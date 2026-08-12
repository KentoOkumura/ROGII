# 要件

## 依頼

exp345のcausal time-varying GR affine scheduleとexp350のbidirectional RTS scheduleを、
現行temperature-5 likelihood-PFのparticle GR emissionへ適用する二variant
design-only実験を作成する。

## 根拠

- exp345 causal HMMはmasked last-640でparentを`0.169505 ft`改善し4/5 foldsだったが、
  worst `+9.354827 ft`でFAILした。
- exp350 RTS HMMはparentを`0.133499 ft`改善し5/5 foldsだったが、causalより
  `0.036006 ft`悪く、p95 `+1.346427 ft`、worst `+20.887374 ft`でFAILした。
- exp211 static affine PFはraw PFより`2.544695 ft`悪化したが、exp345/350の
  dynamic scheduleをPF filtering尤度へ使う実験はない。

## 制約

- Routeは`pf_beam`。親exp417、実装親・保存control exp404。
- base pathはSHA固定したexp209 posterior mean/std。schedule式はexp345/350を固定。
- Variant A causal EKF、Variant B fixed-interval extended RTS。二候補は独立報告する。
- PF emission centerだけを`a_t*GR_typewell(TVT_particle)+b_t`へ変更し、
  exp404 sigma、dynamics、particles、seeds、T=5は固定する。
- full score-row GRはtestでも観測可能なのでRTSはdeployableだが、suffix TVTは使わない。
- same-OOF winner selection、static/dynamic blend、parameter/grid救済は禁止。
- 2026-07-30の追加依頼で実装を承認済み。その後の「実行してください」で
  canonical train採用、Kaggle package/push、Stage 0実行を承認済み。
  Stage 1、inference、submissionは未承認。

## 受け入れ基準

- base path、causal/RTS schedule、emission timing、fallback、fixed値が一意である。
- raw GRのschedule updateとPF emissionへの二重利用リスクを明記する。
- 2×fixed32と2×773の実行量、variant別gate、truth-late、SHA契約を固定する。
- schedule/a/b/process noise/slope bound/scale/temperature/gate救済を禁止する。
