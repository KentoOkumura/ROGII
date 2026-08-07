# exp363_sticky_gr_reliability_exact_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了した。technical gateはPASS、
scientific gateはFAILし、decisionは`stage_0_failed_close_without_rescue`である。
Stage 1、inference、submissionは行わない。

## 仮説

sticky な GR reliability 状態により、rate transition を変えず観測過信を抑えられる。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- 検証: saved exp209 path上のStage 0 reliability readout
- block: 512 rows / stride 256、15,174 blocks / 773 wells
- 実行量: diagnostic 1 / reporting folds 5
- HMM / model / LightGBM / trained fold / booster / control再実行:
  `0 / 0 / 0 / 0 / 0 / 0`
- kernel: `kentookumura/exp363-sticky-gr-reliability-exact-hmm-train`
  version 1、id_no `128370770`
- runtime: `497.082523 sec`

## 結果

| メトリック | 値 |
| --- | --- |
| pooled real bad10 AUC | 0.607552 |
| circular bad10 AUC | 0.583996 |
| real - circular AUC | +0.023556 |
| Q1 mean block RMSE | 5.240148 ft |
| Q4 mean block RMSE | 10.056454 ft |
| Q4 - Q1 mean block RMSE | +4.816306 ft |
| AUC > 0.50 folds | 5/5 |
| hidden-like spatial AUC | 0.546058 |
| hidden-like typewell-purged AUC | 0.552195 |
| row-weighted weak mass | 0.589441 |
| Public LB | - |
| Private LB | - |

technical gate、pooled AUC`>=0.60`、circular差`>=0.02`、
Q4-Q1`>=0.50 ft`、5/5 folds、hidden-like typewell-purgedはPASSした。
一方、hidden-like spatialは`0.546058 < 0.55`、weak massは
`0.589441 > 0.50`で、固定AND gateをFAILした。

fold別real bad10 AUCは`0.608712 / 0.619258 / 0.541517 / 0.627191 /
0.636879`で全foldが0.50を上回った。ただしfold 2ではcircularとの差が
`-0.008056`であり、局所的な安定性は十分でない。

## Technical gate

PASSした。

- 3,783,989 rows / 773 wells / 15,174 blocks、全score finite。
- expected / observed foldsはともに`0..4`。
- truth columns read before freezeは0、late attachment identity mismatchは0。
- multi-block circular offsetは全て非ゼロ、Q1/Q4境界はstrict。
- HMM well-run / model config / trained fold / booster / parent control rerunは全て0。

## Scientific gate

FAILした。

- PASS: pooled AUC、circular差、Q4-Q1、5/5 fold AUC、
  hidden-like typewell-purged。
- FAIL: hidden-like spatial AUCとrow-weighted weak mass。
- Stage 1 eligibilityはfalse。

## 再現性

- deterministic anchor: no（固定入力のdeterministic diagnosticでありsubmission anchorではない）
- seed policy: RNGなし、stable order
- kernel version: 1
- scientific contract content SHA:
  `bd0ed5b8d0c916fcda75cd791786da82b597d89a2fe2186760dc24131aecdbe2`
- block ledger content SHA:
  `967c495f91a6c4ff1aa5f897207c0bf0437a48cb201d82f8f3680981b4434843`
- weak posterior content SHA:
  `38e1fdaca08513143e281b53f43c6c4d64a621392fca4e47f2f5a3e682a83337`
- late-truth block readout content SHA:
  `672ab37aa3b94984e67c842a6c282bbd680cc1018407c6bf86e1fec2df5ed89c`
- gate raw SHA:
  `0a00934e1b3b3b1175fbe4ff5b7bf1c8a94bce67f4093eb6c32612fa018bb3a5`
- downloaded summary raw SHA:
  `248ca4197c29459989eea263ed5123445d1c5d3c0ef980427ce10da43b8834ce`
- model SHA / manifest SHA: estimatorなし
- prediction SHA: 未生成
- submission SHA: 提出無効
- rerun result: 未実行

## 解釈

weak posteriorはpooledではbad blockと相関し、単純なwithin-well circular controlも
僅かに上回った。しかし全posterior massの約59%をweakへ割り当てるため、固定状態は
「一時的な観測不良」より広い区間をweakと解釈している。hidden-like spatialへの転送も
事前下限を僅かに下回った。識別signalはあるが、固定q契約をexact HMMへ追加する根拠としては
不足している。

これは技術失敗ではなく科学仮説のFAILである。transition、multiplier、sigma、block長、
threshold、blendを同じOOF上で調整して救済しない。

## 次

branchを閉じる。Stage 1 exact HMMは実装・実行しない。同じq契約に依存する
exp368 marginalized reliability PFは本結果をnegative dependencyとして扱い、
独立したidentifiability根拠なしには実装へ進めない。
