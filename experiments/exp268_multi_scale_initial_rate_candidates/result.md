# exp268_multi_scale_initial_rate_candidates 結果

## 状態

Kaggle CPU shard 0/1とaggregate version 1が完了した。773 wells / 3,783,989 rowsをstrictに統合し、
候補bankのtarget-free生成契約とSHAを確認した。inferenceとsubmissionは行っていない。

## 仮説

known prefixの複数scaleから作る初期rateを独立HMM candidateとして保持すると、exp209
`tail_n=30` controlの単一rate依存を補完できる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 変更: initial-rate window `32/64/128/256`のみ
- 固定: exp209 exact-HMM grid / transition grammar / Gaussian GR emission / sigma / calibration
- control: 保存済みexp209 `tail_n=30` HMM、再生成なし
- split: `sha256("exp268::well_shard::<well>") % 2`
- 実行量: 4 HMM variants / 2 shards / 0 LightGBM configs / 0 folds / 0 boosters
- GPU / inference / submission: なし / なし / なし

## 結果

| メトリック | 値 |
| --- | ---: |
| tail30 direct RMSE | 11.938287 |
| best rate candidate | `hmm_ir_w128` |
| w128 direct RMSE / gain | 11.895581 / 0.042706 ft |
| initial-rate-5 row oracle RMSE / gain | 11.835929 / 0.102358 ft |
| initial-rate-5 H256 block oracle RMSE / gain | 11.836137 / 0.102151 ft |
| initial-rate-5 whole-well oracle RMSE / gain | 11.840973 / 0.097314 ft |
| rate spread median / p90 | 0.000 / 0.020 |
| zero-rate-spread wells | 423 / 773 |
| Public / Private LB | 対象外 |

window候補間のpath duplicate率はpairにより58.99%から88.36%だった。oracle prediction、candidate
mean、selectorは保存していない。

## 再現性

- aggregate kernel: `kentookumura/exp268-multi-scale-initial-rate-aggregate`, id `127887734`
- runtime: 295.676秒
- shard 0 decompressed SHA: `a38ac16d12c9cd650170d16a9eb0b75159dd6e119443d33b7d7290a9e5347066`
- shard 1 decompressed SHA: `30d6d7e930ffdec02f0da46108803c5640a03b26ea9b6cf8232ad7fdb06f0d36`
- aggregate prediction content SHA: `fc18952f564dcefed8222ee30510828a4fb47f51c06a0eec5b1ddf37887ecdd1`
- aggregate summary raw SHA: `8bd2064892f7eb05392785d602e810b9aea8b686225994cd515247609370e0c6`
- aggregate manifest raw SHA: `427aa3f15c8577b38448836d3adea58ef69dcf43d6a79237f0c603b9bf04494b`

## 解釈

固定window bankには約0.10 ftの小さいoracle headroomがあるが、423 wellsでは4 windowのrateが同一で、
候補重複も多い。direct bestのw128改善は0.0427 ftに留まるため、これだけでdeployable候補や
selectorを正当化しない。target-freeな識別可能性は子実験exp292で監査し、FAIL-closeとなった。

## 次

exp268候補を推論やsubmissionへ採用しない。exp292の停止条件に従い、同じ候補bank上の
frequency-warp救済gridも行わない。
