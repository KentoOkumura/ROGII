# exp440_ambiguity_gated_predictive_prior_hmm 結果

## 状態

Stage 0はFAIL closedだったが、2026-07-30の明示依頼により変更なしcandidateを
full 773-well OOFで確認した。4 CPU shardsとstrict mergeはtechnical gateを
全PASSした一方、scientific gateはFAILした。最終状態は
`stage1_full_oof_failed_closed`であり、inferenceとsubmissionへ進まない。

## 仮説

通常GR emissionを適用したcausal provisional filtered TVT marginalが二峰化した
raw-GR-observed行では、emissionをneutralizeしてtransition後のpredictive priorを
維持すると、wrong TVT basinへの更新を減らせる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 変更: exp236固定二峰判定がtrueの行だけemission lambda `1.0 -> 0.0`
- candidate: 1本、773 HMM well-runs、saved parent control rerun 0
- 4 shards: `193 / 193 / 193 / 194 wells`
- LightGBM / fitted model / booster / PF / Beam / GPU: すべて0
- true TVT / fold / hidden-like roleは全target-free生成物のstrict mergeと
  SHA freeze後だけ結合

## Full OOF結果

| メトリック | 値 | 判定 |
| --- | ---: | --- |
| candidate RMSE | 12.992063 ft | FAIL |
| parent exp209 RMSE | 11.938287 ft | - |
| gain vs parent | -1.053776 ft | FAIL（基準 +0.02 ft） |
| positive folds | 1 / 5 | FAIL（基準 4 / 5） |
| ambiguous rows | 653,589 | - |
| ambiguous-row SSE削減 | -21.3117% | FAIL（基準 +5%） |
| by-well RMSE delta p95 | +11.631749 ft | FAIL（上限 0 ft） |
| worst-well regression | +45.003490 ft | FAIL（上限 +0.25 ft） |
| technical gate | 全項目PASS | PASS |
| scientific gate | 全体FAIL | FAIL |

fold別はfold 3だけ`+0.228675 ft`改善し、fold 0/1/2/4はそれぞれ
`-1.005467 / -2.072584 / -0.300978 / -1.911774 ft`悪化した。

全てnonworse必須だったscopeも悪化した。

| scope | candidate - parent RMSE |
| --- | ---: |
| raw GR observed | +1.182350 ft |
| raw GR missing | +0.773104 ft |
| high missing fraction | +0.857190 ft |
| MD 1000+ | +1.192698 ft |
| hidden-like spatial | +2.186790 ft |
| hidden-like typewell-purged | +2.004114 ft |

## 実行と再現性

- shards: Kaggle private CPU version 1、全てCOMPLETE
- shard runtime:
  `5,603.734 / 8,695.083 / 8,162.896 / 5,549.296 sec`
- shard peak RSS:
  `1.571 / 1.499 / 1.509 / 1.527 GB`
- merge:
  `kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-merge`
  version 1、COMPLETE
- merge/readout runtime / peak RSS: `292.181 sec / 4.773 GB`
- raw identity SHA:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- merged prediction logical SHA:
  `d7745518e64f732c24d9a7323e55eae8be90384fd4112dd4bd827fc2e1513d79`
- merged schedule logical SHA:
  `decf076e6ce0b4912c33a42dcbe1fb20361b683b980ded7fa62f0f0a1a82d546`
- merged diagnostic logical SHA:
  `d804ce7eb3eb7ac71eb1a6abd53e317a18b3e611b581aa8d0e1f7244be96ecab`
- freeze前のtruth / fold / role read: 0
- scope metrics SHA:
  `baf4bf60ebcd8317d709f082883efc39aed5966754902d25bcdc64b0e6444a3d`
- by-well metrics SHA:
  `ef87808f6f4d89b8c66638ee5e28f791af96b661ba6dc2405879f531ea662f71`
- gate SHA:
  `409747334948e70995bc08267c0566846a9d0cd4e302c9ca2b9bb013ac7ebf93`
- summary SHA:
  `88dd2a28a9b86d3fe3b97de30b0d561d9cd05d80f564606c211ac14b1e42c13c`
- model / submission SHA: 非該当
- deterministic anchor: false（独立rerunなし）

## 解釈

fixed32ではpersistent寄りscopeのためpooled改善に見えたが、full OOFでは
1.0538 ft悪化し、曖昧row自体のSSEも21.3%悪化した。predictive priorが既に
wrong basinにある行でGR emissionを無効化し、その誤りをforward/backwardの
両方向へ保持した影響が支配的と考える。exp408の「current emissionより
transition/prior hysteresisが主因」というnegative evidenceと整合する。

## 次

decisionは
`close_without_blend_selector_continuous_gate_or_same_oof_rescue`。
threshold、lambda、transition、blend、selector、well/row gateで同じOOFを
救済せず、rerun、inference、submissionを行わない。
