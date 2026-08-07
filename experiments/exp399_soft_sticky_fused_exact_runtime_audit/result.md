# exp399_soft_sticky_fused_exact_runtime_audit 結果

## 結論

Kaggle private CPU version 6（id_no `128546220`）で、773 wells /
3,783,989 rowsのfull OOFを完了した。全technical gateはPASSしたが、候補RMSEは
`11.395645678 ft`で、昇格基準のexp263 `8.238331667 ft`より
`3.157314012 ft`悪化した。decisionは
`promotion_rejected_no_parameter_rescue_blend_selector_inference_or_submission`。
同じOOFを見たparameter調整、blend、selector、inference、submissionは行わず閉じる。

## OOF比較

| 対象 | RMSE (ft) | exp399との差 |
|---|---:|---:|
| exp209 exact HMM | 11.938287556 | exp399が0.542641877改善 |
| exp355 saved reference | 11.291976616 | exp399が0.103669062悪化 |
| exp226 geometry | 9.427109836 | exp399が1.968535842悪化 |
| exp263 promotion baseline | 8.238331667 | exp399が3.157314012悪化 |
| exp399 soft-sticky | 11.395645678 | - |

fold別ではexp263に対して`+1.724406 / +3.503768 / +2.260748 /
+4.475109 / +3.482840 ft`で、改善foldは`0 / 5`だった。773 wellsのうち
非悪化は`40.4916%`、paired by-well delta p95は`+12.034886 ft`、
worst well `2364716c`は`+38.148059 ft`だった。

## Stress readout

- near 0--250: `+0.517525 ft`
- 1000+: `+3.488101 ft`
- hidden-like spatial: `+4.385663 ft`
- hidden-like typewell-purged: `+4.198109 ft`
- persistent offset episodes: `551 → 689`（`+138`）
- recovery within 512 rows: `0.090744 → 0.146589`（`+0.055845`）

recovery率は改善したが、persistent episode自体が増え、全主要scopeで悪化した。
branch occupancyはE/H=`0.052990 / 0.947010`、expected switchesは
`0.348381 / 1000 MD-ft`で非退化gateをPASSした。したがって失敗理由はbranchが
全く切り替わらなかったことではなく、主にH branchへ滞在するsoft-sticky posteriorが
exp263の固定物理blendほど正確でなかったことにある。

## Runtime

- total: `25,118.126809 sec`（`6.977 h`）
- prediction freeze: `24,669.330589 sec`（`6.8526 h`）
- late truth / promotion readout: 約`448.796 sec`
- fixed16からの投影: `18,277.265455 sec`（実測は約`37.4%`長い）
- fixed limit: `30,600 sec`（使用率`82.09%`）
- exp394 full projection `112,736.889439 sec`比: 約`4.49x`高速

fixed16投影より長かったのは、full 773 wellsの長さ分布とCPU割当の差に加え、
全OOF保存・truth-late join・stress集計を含むためである。それでも全TVT gridと
41 rate statesを維持したまま固定runtime gate内で完走した。

## Technical / leakage

- row / well / unique identity: `3,783,989 / 773 / PASS`
- finite prediction coverage: `1.0`
- posterior normalization max error: `5.528911e-14`
- transition row-sum max error: `8.881784e-16`
- exp263 saved baseline parity abs diff: `1.207029e-7 ft`
- truth / hidden role reads before prediction freeze: `0 / 0`
- model / booster / parent-control rerun: `0 / 0 / 0`
- GPU / inference / submission: なし

version 5のERROR原因だったexp226 reporting foldとexp263 generation foldは、
独立したgroup-safe OOF ledgerとしてそれぞれ検証した。631 / 773 wellsでfold labelが
異なることをprovenanceとして保存し、OOF予測のrow join条件には使っていない。

## 再現性

- kernel: `kentookumura/exp399-soft-sticky-fused-exact-runtime-audit-train`
- version / id_no: `6 / 128546220`
- preflight summary SHA:
  `ac800ac0ada91fd1df4486ea5ae580c32e7afd3c0ae7de48172476291550c660`
- prediction content SHA:
  `d44b382a310c7d53bf5dc90a238c44a247b2ba09d2a1f0648174f4d7c85fb18e`
- branch posterior content SHA:
  `d953cebc3d46ef8cd4a03d9faaceab0cf74fbe998a2fe9a9d2141563996cd8a1`
- schedule content SHA:
  `98e6f2781c615d5bcecc594904597a2a72707347c41ac150a601f40155b02a31`
- promotion gate raw SHA:
  `ace88e93431b264da907935812927492891a59844158ad6a4d84d591b216a747`
- OOF gzip raw SHA:
  `4edf39adc1d91d71a50a01c5623cf1a9ea75f8e374718a50e156e31047bc5cbd`
- output:
  `/tmp/kaggle-output/exp399_soft_sticky_fused_exact_runtime_audit/train_v6`

取得したOOF、branch posterior、rate scheduleのraw SHAはsummaryと一致し、
3つのgzip integrity checkもPASSした。

## 判断

runtime kernelは全状態・実用的数値精度を保った高速化基盤として成功した。一方、
exp394のsoft-sticky科学仮説はexp263を置換できず、stressとwell-tailも大きく悪化した。
したがって高速kernelは将来の独立した候補に再利用可能だが、この候補そのものは
昇格・推論・提出しない。

## 次

exp399内では追加実行しない。将来、別の事前固定されたexact-HMM候補で計算時間が
障害になった場合に限り、このruntime kernelを実装基盤として再利用する。
exp399のbranch weight、switch length、docking幅を同じOOFで調整する救済は行わない。
