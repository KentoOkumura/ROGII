# exp364_signed_curvature_exact_hmm 結果

## 結論

Kaggle private CPU version 1でStage 0を完走したが、科学gateはFAILした。
`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`としてbranchを閉じ、Stage 1 exact HMM、
inference、submissionへ進まない。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- route: `pf_beam`
- 検証: Stage 0 signed-path separability + 16-well resource projection
- block / stride: `512 / 256`
- 候補符号: `[-1, 0, +1]`
- Stage 0 exact HMM / LightGBM config / trained fold / booster / control rerun:
  `0 / 0 / 0 / 0 / 0`
- kernel:
  `kentookumura/exp364-signed-curvature-exact-hmm-train` version 1、
  id_no `128529795`
- private CPU、GPU / TPU / internet off

## Stage 0結果

- 実行時間: `224.737080 sec`
- 入力: 773 wells、3,783,582 candidate rows
- 評価可能: 772 wells、13,631 complete blocks
- technical gates: `12 / 12 PASS`
- scientific gates: `6 / 9 PASS`
- top1: `0.550143 >= 0.40` PASS
- MRR gain vs zero-first: `0.252574 >= 0.01` PASS
- real-minus-circular top1: `0.003081 < 0.03` FAIL
- passing folds: `3 / 5 < 4 / 5` FAIL
- selected path RMSE gain vs zero path: `+1.844153 ft`
- 1000+: `+2.027732 ft`
- hidden-like spatial: `+1.466085 ft`
- hidden-like typewell-purged: `+1.459314 ft`
- projected runtime: `33857.604 > 30600 sec` FAIL
- projected peak RSS: `4.880433 < 25 GB` PASS

fold 1と2でreal-minus-circular top1が負になり、方向gateを通過したfoldは3/5だった。
また、overallでもrealと1-block circular controlの差は`0.003081`しかなく、
GR scoreがsigned pathを選べること自体は確認できても、時系列対応に固有な識別力は
事前条件を満たさなかった。

## 再現性

- deterministic anchor: true
- contract SHA:
  `742f172c6d602ba7bfe6c89df30cf1aa97999a2c40b8cc35d18b232da45edaf9`
- freeze前truth / hidden-like role read: `0 / 0`
- candidate、GR score、resource projectionの保存後SHA再読込: 全PASS
- candidate path decompressed content SHA:
  `64a8c744ec99730905032f950875f76c61c23df7504a0731592abc6101c63cbb`
- block GR score decompressed content SHA:
  `40bdad70b46670b6acd01b26baf8b319bbcfdb302899a6e7787145ce20955048`
- resource projection SHA:
  `cef21a2f44137df5a4b98520d80f6273045dd6aff2bea534194171ee393af4a9`
- freeze manifest SHA:
  `9c0d27a304ceff3778a4e5b69ae9d56224c1932ecbd64050c9327a7cc9ad2feb`
- gate report SHA:
  `52e1c3d4e4e46b67ba6e1ddbf1a39d22e421c76b72121b6e0ca2ca1ded2fb741`
- summary SHA:
  `75f32f0fef65f1af9ae8f8e58ccd1ae05a471f518acecf5653e0dd312df73541`

Kaggle outputから監査対象10成果物だけを取得し、freeze manifestのraw SHAと
gzip展開後content SHAを再計算して一致を確認した。

## 解釈

zero-firstに対するtop1 / MRRと全3 stress scopeのRMSE方向は良い一方、
circular controlとの差がほぼ消えるため、固定GR path scoreは局所的な曲率符号ではなく
GRの自己相関やblock-level適合を主に拾っている可能性が高い。加えて、状態数3倍の固定
runtime projectionがhard上限を超える。片方だけでなく独立した2条件がFAILしたため、
curvature magnitude / persistence、emission、sigma、adaptive noise、parallelism、
blendによる救済は行わない。

## 次

exp364と同じ科学branchは閉じる。将来のpath-ranking preflightに限り、truth-freeで
複数の十分離れたcircular nullを固定し、negative-control自体の検出力を監査する
独立0-HMM readoutを候補とする。これはexp364を再昇格させるものではなく、
exact HMM、予測、inference、submissionを含めない。
