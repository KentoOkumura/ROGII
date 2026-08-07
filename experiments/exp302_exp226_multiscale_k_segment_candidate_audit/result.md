# exp302_exp226_multiscale_k_segment_candidate_audit 結果

## 状態

Kaggle private CPU version 2を完了した。technical guardはPASS、direct guardはFAIL、
candidate novelty guardはK12/K24ともPASSした。direct候補は昇格させないが、exp302側の
exp303開始条件は満たした。

## 固定設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- control: 保存済みK16 OOF、実測RMSE `9.4271095966`
- variants: K12/K24
- 比較bank: exp293 fixed deployable12
- 検証: exp226保存済み5 fold、unknown suffix、truth late join
- 実行量: 2 variants × 5 folds = 10 CPU runs、0 booster、control再生成0
- Kernel: `kentookumura/exp302-ksegment-candidate-audit-train` version 2、id_no `128010921`
- 実行時間: audit完了 `1281.068 sec`、最終log `1291.656 sec`

## Direct結果

| variant | pooled RMSE | K16差 | 改善fold | Direct guard |
| --- | ---: | ---: | ---: | --- |
| K12 | 9.551938 | +0.124828 ft | 0/5 | FAIL |
| K24 | 9.413244 | -0.013865 ft | 3/5 | FAIL |

K24はpooled、1000+、hidden-like 2面、worst wellを改善したが、事前条件のpooled
`<=9.377110`と4/5 foldsを満たさなかった。K12はpooledと全foldを悪化させた。

## Candidate novelty結果

exp293 fixed deployable12へ各variantを別々にadd-oneした結果:

| variant | H512 oracle改善 | whole-well oracle改善 | H512 strict unique-best | 改善fold | Novelty guard |
| --- | ---: | ---: | ---: | ---: | --- |
| K12 | +0.066095 ft | +0.068466 ft | 10.6973% | 5/5 | PASS |
| K24 | +0.083901 ft | +0.066231 ft | 10.8899% | 5/5 | PASS |

両variantとも全条件を余裕を持って通過した。K24はH512改善が大きく、K12はwhole-well改善が
わずかに大きい。K12/K24を同時追加するoracleやdeployable selectorは評価していない。

## Technical・再現性

- 3,783,989 rows / 773 wells、finite coverage 1.0、各variant 5 fold runs。
- evaluation truth access before freeze: 0。
- K16 OOF decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- candidate bank content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`。
- block assignment decompressed SHA: `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`。
- K12 prediction content / decompressed SHA:
  `c3d7dfe20ad3b8c7d6d5220023bbb4526fb90d10cc73f01e612db847af70da63` /
  `63b381299ee46fa172680af57959d675c68b6b24af05664c8689dd291961f22d`。
- K24 prediction content / decompressed SHA:
  `dca92e8f21d3b8b33d1543fe3df0bf586be3a2604b76ee1bf19fa84a327f06ef` /
  `ca36d168b45acb15cc814ac3c1c3437894cd1050f6c51ba03f5b302efd0a31aa`。
- output manifest 16/16件でfile SHAを照合し、gzip 3件はdecompressed SHAも一致した。
- stochastic処理、model、booster、inference、submission、oracle prediction保存はない。

## 解釈

segment解像度をK24へ上げるだけでは事前登録したdirect昇格水準に届かなかったため、K24を単独の
direct予測やsubmission候補として採用しない。一方、K12/K24はfixed12 bankと高相関でも
H512/whole-wellで一貫したadd-one headroomを持つため、scale instabilityをtarget-freeな
selectability signalとして読むexp303の根拠は成立した。

## 次

exp303の「exp302 technical + novelty PASS」は充足した。ただし別の必須条件であるexp276の完了と
promotion guard FAILは未充足である。exp276がPASSした場合、または未完了の間はexp303を実装・実行しない。
inferenceとsubmissionは引き続き対象外とする。
