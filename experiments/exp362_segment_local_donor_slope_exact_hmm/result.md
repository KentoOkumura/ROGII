# exp362_segment_local_donor_slope_exact_hmm 結果

## 結論

Kaggle private CPU version 1 は完了したが、branch は fail closed とする。

notebookが報告したpooled RMSEはexp209の`11.938287235 ft`から`11.161677223 ft`へ
`0.776610012 ft`改善した。一方、改善foldは固定条件4/5に対して3/5、worst well
`86454a6f`は`+52.741425793 ft`悪化し、科学gateはFAILした。

さらにpost-run成果物監査で、12,368 target segmentsのうち局所donor gradientを採用したsegmentは
0件だった。11,596 segmentsはeffective donor数不足、772 segmentsは最近傍距離超過で、
全segmentの`mu_rate`がtarget既知prefix rateと完全一致した。このためversion 1のscoreは
意図したlocal donor-slope介入の評価ではなく、prefix-rate-only residual exact HMMへ退化した候補の
参考値としてのみ扱う。

## 実行

- Kernel: `kentookumura/exp362-segment-local-donor-slope-exact-hmm-train`
- Version / id_no: `1 / 128368310`
- Status: `COMPLETE`
- Runtime: `19,777.653141 sec`（約5時間29分38秒）
- Runtime条件: Kaggle CPU、GPU off、internet off
- 実行契約: 1 scientific variant / 5 reporting folds / 773 HMM well-runs
- Model config / trained fold / booster / parent control再実行: `0 / 0 / 0 / 0`
- Inference / submission: 未実行

## スコア

| Scope | Candidate RMSE | Parent RMSE | 改善量 |
| --- | ---: | ---: | ---: |
| pooled | 11.161677 | 11.938287 | +0.776610 |
| fold 0 | 10.218923 | 11.075149 | +0.856227 |
| fold 1 | 12.635564 | 12.444775 | -0.190789 |
| fold 2 | 9.805174 | 12.795686 | +2.990512 |
| fold 3 | 10.892624 | 11.598033 | +0.705409 |
| fold 4 | 11.965839 | 11.648854 | -0.316985 |
| distance 1000+ | 12.276555 | 13.135431 | +0.858877 |
| hidden-like spatial | 12.291305 | 12.564491 | +0.273187 |
| hidden-like typewell-purged | 12.015278 | 12.367244 | +0.351966 |

by-wellでは462/773 wellsが改善、311/773 wellsが悪化した。candidate p95 RMSEとparent p95
RMSEの差は`-2.896174 ft`でguardを通ったが、最大悪化wellは固定上限`+0.25 ft`に対して
`+52.741426 ft`だった。

## Gate

notebook内のtechnical gateは次を満たしてPASSした。

- 3,783,989 rows / 773 wells、finite coverage 1.0、duplicate 0
- outer-valid donor除外PASS
- truth-before-freeze 0、exp226 artifact resolve 0
- 773 HMM well-runs
- posterior正規化最大誤差 `3.77e-15`
- 保存済みexp209 control SHA一致
- runtime上限30,600秒以内

ただしtechnical gateには「局所gradientが非退化で採用されたsegment数」の条件がなかった。
post-run support auditは次の理由でFAILとする。

- accepted local-gradient segments: `0 / 12,368`
- `mu_rate == prefix_rate`: `12,368 / 12,368`
- finite gradient rows: `0`
- effective donors `>=10`: `0`
- fallback reason: effective donors `11,596`、nearest distance `772`

保存された`fallback`列が全行Falseだったのは、target rowを組み立てる辞書mergeでlocal-gradient側の
`fallback`がprefix-rate側の同名`fallback`に上書きされた記録バグである。`fallback_reason`と
`mu_rate`は上書きされておらず、完全退化の判定には影響しない。

## 再現性

- Raw identity SHA:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- Scientific contract SHA:
  `2e80fd0573acc601e3b5fe28e6673725c3e7038a1393d0ae6110ed46dd4b2128`
- Donor ledger logical / decompressed SHA:
  `92a6f7e9...f33` / `cafa6009...14c`
- Target prior logical / decompressed SHA:
  `b84487c4...a91` / `84318915...0c9`
- Rowwise schedule logical / decompressed SHA:
  `d6cbf1a7...cb7` / `5cabe7f3...8e8`
- Prediction logical / decompressed SHA:
  `bdf616e0...5cb` / `e1d672ff...7ef`
- SHA manifest raw SHA:
  `5ac384b2...fc3`
- 取得したmanifest対象10成果物のraw SHA: 10/10一致
- Rerun parity: 未確認
- Deterministic anchor: false

## 解釈

観測されたpooled改善は、局所donor slopeではなくtarget既知prefix rateを全suffixで遷移平均に
保つ介入によるものと考えるべきである。平均・1000+・hidden-likeでは正方向だったが、
fold 1/4とworst-well tailが大きく不安定で、prefix-rate-only候補としてもpromotion条件を満たさない。

また、固定bandwidth 500 ftとminimum effective donors 10の組合せは今回の空間密度に対して
全segmentで不成立だった。これは同じK16 donor supportへ依存する未実装exp356の先行リスクでもある。

## 判断

- exp362を完了済み・fail closedとする。
- K、近傍数、bandwidth、ridge、support/fallback、HMM parameterの同一OOF救済を行わない。
- reporting bug修正だけを理由にversion 2を再実行しない。
- inference、blend、selector、submissionへ進めない。
- exp356は同じsupport前提が非退化になる独立証拠が得られるまでblocked/demotedとする。
