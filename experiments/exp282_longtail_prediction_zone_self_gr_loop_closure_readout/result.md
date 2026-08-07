# exp282_longtail_prediction_zone_self_gr_loop_closure_readout 結果

## 状態

Kaggle private CPU version 1でfull readoutを完了した。technical guardは全PASSしたが、
fixed scientific guardはFAILした。事前契約どおりparameter rescueを行わずbranchを閉じる。
補正、inference、submissionは実行せず、CV / LB anchorも更新しない。

## 仮説

1000 ft以降のlong-tail receiverと、予測開始後0～500 ftのearlier donorにGR loop closureがあれば、
known TVT区間を使わずに同一TVTらしさを検出でき、matched donor側のexp263 fixed予測が
receiverのpersistent offsetを軽減する方向を持つ可能性がある。

## 設定

- 親: `exp090_lateral_self_gr_match_pseudotail_probe`
- 固定予測参照: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 検証: 保存済みgroup-safe 5 folds、long-tail prediction-zone self-GR readout
- matching: rolling mean 5、half-window 8/15/25、stride 3、forward/reverse NCC
- high-confidence: well内target-free confidence上位10%
- 実行量: 1 readout variant / LightGBM config 0 / trained fold 0 / booster 0 / HMM 0 / PF 0

## 実行

- kernel: `kentookumura/exp282-longtail-self-gr-loop-closure-readout-train`
- Kaggle `id_no`: `127838798`、version 1、private CPU、GPU/TPU/internet off
- status: `COMPLETE`
- runtime: `248.206秒`
- 対象: 3,783,989 rows / 773 wells
- raw long-tail receiver: 3,012,442 rows
- eligible center / frozen edge: 997,733 / 997,733、coverage 1.0
- high-confidence: 100,103 edges / receiver coverage 3.323%

## Primary readout

| selection | real within10 | shuffled within10 | lift | real median delta | shuffled median delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all edges | 0.550644 | 0.549871 | +0.000774 | 8.760 ft | 8.770 ft |
| high confidence | 0.554309 | 0.551052 | +0.003257 | 8.680 ft | 8.750 ft |

high-confidence `within10`のfold別liftは
`[-0.000399, +0.006493, +0.005594, +0.001749, +0.002847]`で、正方向は4/5だった。
all-edgeは3/5、high-confidence median delta改善も4/5で、いずれも事前要求5/5に未達だった。
pooled precision 0.554309も要求0.60を下回った。

distance別high-confidence liftは1000～1500 / 1500～2500 / 2500+で
`+0.005911 / +0.001596 / +0.003375`だった。hidden-like spatial / typewell-purgedも
`+0.005491 / +0.004300`で正方向だったが、fold安定性と絶対precisionの失敗を救済しない。

## Donor-transfer readout

| selection | receiver baseline RMSE | matched donor-transfer RMSE | gain |
| --- | ---: | ---: | ---: |
| high confidence | 8.954770 ft | 15.849509 ft | -6.894739 ft |

fold別gainは`[-6.469817, -6.432448, -6.422177, -7.317443, -7.775890] ft`で、改善は0/5 folds。
well別でも改善は115/771 wellsだけで、中央値gainは`-4.841993 ft`だった。最悪well
`2fd68f7b`はbaseline 8.865714 ftから59.729281 ftへ悪化し、gainは`-50.863567 ft`だった。

## Guard

- technical: expected folds、edge coverage、finite score、forbidden score columns、truth-before-freeze zeroを全PASS
- scientific PASS: high-confidence receiver coverage、hidden-like 2面のpositive lift
- scientific FAIL: pooled precision、all/high-confidence 5/5 fold lift、5/5 median改善、5/5 donor-transfer改善、pooled donor-transfer gain
- final: `FAIL` / `close_branch_without_parameter_rescue`

## 再現性

- deterministic anchor: いいえ。成功runは1回のreadout-only。
- truth attachment before freeze: 0件
- frozen edge logical content SHA:
  `2b9ecbb956e2b84ee61ddefeb54ed0fcca98b984e76ba1be0e7c9321f5f74c28`
- target-free edge gzip raw SHA:
  `fa70053b4f290e2bb487bca2b48e99389b9db0f57e43b3df448c9b05b9e9d297`
- target-free edge decompressed SHA:
  `e2a425bde45ef8abea59838cf734856d6c5c27671503ce70305320ced9a408a1`
- summary SHA: `32896b76ad069fb7bc569ce8ab1c6b6e389d0f06800f5d7ef2c0cc27d38894e9`
- Kaggle log記載SHAと取得した11 artifactのfile SHAは全て一致した。
- model / prediction / submission SHA: 対象外。

## 解釈

target-free self-GR confidenceにはshuffledよりわずかに良いpooled signalがあり、hidden-like 2面でも
正方向だった。しかしliftは小さくfold 0で反転し、same-TVT precisionも55.43%に留まる。さらに
matched donor predictionの直接transferは全foldで大幅に悪化した。GR motif一致はabsolute TVTの
対応やreceiverのpersistent offset修正を保証せず、安全なpseudo-anchorとしては使えない。

## 次

window、stride、confidence weight、threshold、donor範囲の救済grid、soft correction、HMM/PF接続、
raw-test inference、submissionは行わない。新規救済backlogも追加せず、このloop-closure branchを閉じる。
