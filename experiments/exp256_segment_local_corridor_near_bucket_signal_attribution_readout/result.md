# exp256_segment_local_corridor_near_bucket_signal_attribution_readout 結果

## 状態

Kaggle CPU train v1で保存済みexp250 Stage 1生成物のfull attribution readoutを完了した。
判定は`diagnostic_only_no_exp250_route_or_use_change`であり、exp250の不採用判断を変更しない。

## 仮説

exp250の0--100 ft pooled real AUC約0.82は、広いGR topology固有signalではなく、
nearでcandidate familyごとのbad rateが大きく異なること、well構成、distance-conditioned
base error、riskの飽和によって大部分が説明される可能性がある。

## 固定設定

- 親: `exp250_segment_local_negative_space_gr_corridor_audit`
- route: `pf_beam`
- 入力: exp250 Stage 1 candidate-segment / group / by-well / summary
- candidate rows / paired keys / wells: 291,710 / 145,855 / 773
- model config / fold / booster: 0 / 0 / 0
- PF/Beam / corridor再生成: 0 / 0
- parameter grid / inference / submission: なし / disabled / disabled

## Near 0--100 ft

| 診断 | real | shuffled | 差 |
| --- | ---: | ---: | ---: |
| pooled AUCの2 bucket評価weight加重平均 | 0.819846 | 0.773559 | +0.046287 |
| distance x family conditional AUC | 0.598678 | 0.574742 | +0.023936 |

- near evaluation weight: 38,299 / 3,652,581 = 1.048546%
- near bad rate: 0.281678（far 100+は0.512820）
- pooled real AUCはfamily条件付けで-0.221168低下した。
- near 10 family-bucket strataのうちAUC算出可能は6 strata / 4 families。
- 0--50 ftでは`beam_mean` / `likpf_mean` / `pf_ancc`のbad weightが0で、AUCを算出できない。
- AUC算出可能なnear 6 strataはreal-shuffled差が全て正だったため、弱いGR固有差は残る。ただし全familyで識別可能な広いsignalではない。

pooled AUC約0.82の大部分は、bad rateが約0のfamilyと約0.75のfamilyを同じpoolで比較し、
corridor riskのfamily差がlabel差と結び付いたcross-family base-rate attributionで説明される。
0--100 ftの値は2 bucketのdescriptive weighted meanであり、新しいpooled AUCの再計算ではない。

## Family x well attribution

family x well 3,865 strata中、bad/good双方がありAUC算出可能なのは2,330 strataだった。

| 診断 | 値 |
| --- | ---: |
| conditional real AUC | 0.522220 |
| conditional shuffled AUC | 0.511096 |
| real - shuffled | +0.011124 |
| 正方向strata | 1,138 / 2,330 |
| 正方向pair-mass share | 0.522241 |

candidate family別conditional AUC差は`sc_ens` +0.006061、`hyb` +0.008765、
`beam_mean` +0.009769、`likpf_mean` +0.012079、`pf_ancc` +0.016923だった。
全familyで平均差は正だが、well-stratumの正負はほぼ半々で、broadなwell一般化signalではない。

## Risk=1.0飽和

- pooled q90 thresholdはreal / shuffledとも1.0。
- risk=1.0の評価weight比はreal 0.188251、shuffled 0.270472。
- realでは`beam_mean` / `likpf_mean` / `pf_ancc`が約0.262--0.265飽和する一方、
  `hyb` / `sc_ens`は0.063007 / 0.053513で、飽和率自体にも大きなfamily差がある。
- risk=1側のbad rateはreal 0.396563で全体0.510396より低く、high-risk tailはbad candidate濃縮になっていない。

## 再現性と実行

- Kaggle kernel: `kentookumura/exp256-seglocal-near-signal-attribution-train` v1
- kernel id_no: `127322012`
- runtime: 6.486863秒
- cell source SHA256: `9f56ec1449b4ce6c8f94e948e9f4932034c60b058e87f02774613685b1b3567c`
- config SHA256: `b119d5cd02cfeaba2fc03dd2c13119000395b419b987edc5ee4b66657a285126`
- summary SHA256: `c2fc5c61980a621089e153e087f04d70842cb3548b4d680beabf292e2f1502a6`
- family x well gzip decompressed SHA256: `a7205f0e6bdec2af2c4a549fb618b538eedf844701419553ca3152e4edaeb6cc`
- paired weight / segment / risk-delta identity、output SHA 11件、寄与総和一致: PASS
- 新規乱数なし、single-process、gzip mtime 0。

## 結論

nearにはshuffledを上回る弱いGR固有差が残るが、pooled AUC約0.82は主にcandidate-family
base rateに帰属し、near weightは1.05%、family x wellの差は+0.0111で正負ほぼ半々だった。
したがって原因切り分けは完了し、exp250 segment-local corridorをhard use、feature、rule、
candidate変更へ再開しない。

## 次

対応バックログを完了扱いで削除する。新規backlogは追加せず、`topk_path_confidence_features`
にもexp250 segment-local signalを混ぜない。threshold/slack/segment grid、near rule、
raw-test inference、submissionは禁止のままとする。

