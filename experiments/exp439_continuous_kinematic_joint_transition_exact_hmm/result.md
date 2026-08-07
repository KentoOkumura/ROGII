# exp439_continuous_kinematic_joint_transition_exact_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を実行した。最初の実データedgeで固定moment
contractが不可能と確定したため、事前登録どおりno-rescueで閉鎖した。
CV / LB / inference / submissionは対象外。

## 実行契約

| 項目 | 結果 |
| --- | --- |
| scientific variant | 1 |
| Stage 0 candidate HMM well-runs | 予定32、完了0 |
| Stage 0 attempted wells | 1 |
| parent HMM rerun | 0 |
| ML / booster / PF / Beam / GPU | すべて0 |
| canonical notebook / strict package | 採用済み / PASS |
| contract test | 12件PASS |
| py_compile / Ruff / Jupytext / strict validation | すべてPASS |
| CV / Public LB / Private LB | - / - / - |

## Kaggle結果

- kernel: `kentookumura/exp439-continuous-kinematic-joint-hmm-train`
- version / id_no: `1 / 129058811`
- runtime: 約`33.181 sec`でfailureを検出
- terminal status: `KernelWorkerStatus.ERROR`
- private / CPU / internet off

Kaggleの例外は次だった。

```text
moment projection infeasible; candidate fails closed at
row=0, source_rate=0, destination_rate=0,
mean_shift=-0.11000000000021828
```

fixed32 manifestをwell順に並べた最初の対象は`060ab2b8`である。HMM message計算や
prediction freezeより前のjoint-edge table構築で停止したため、完了HMM well-run、
prediction、moment audit artifact、truth/role/fold/episode readはいずれも0だった。

## 数値的な失敗理由

固定値はlattice step `0.35 ft`、effective position sigma `0.1225 ft`である。
失敗edgeの平均`-0.11000000000021828 ft`を挟むlattice点は`-0.35`と`0.0 ft`。
この平均を持つ非負lattice分布の最小分散は

```text
(-0.11000000000021828 - (-0.35))
* (0.0 - (-0.11000000000021828))
= 0.026400000000028373 ft^2
```

一方、保存すべきtarget varianceは
`0.1225^2 = 0.015006249999999999 ft^2`で、最小値より
`0.011393750000028374 ft^2`小さい。supportを5、7、9セルへ広げても、
平均近傍のlattice間隔が変わらないため最小分散は下がらない。

したがってsolverの収束問題ではなく、固定0.35 ft lattice上に指定mean/varianceを
同時に持つ非負確率分布が存在しない。実装前から明記したfail-close条件をKaggleの
実データで再現したもので、packageや実装の不具合ではない。

## 判断

technical gateは`nonnegative_lattice_moment_feasibility`でFAIL。scientific /
mechanism gateは未評価。support、moment、noise、grid、rate、emission、prior、
gateを同じexpで変更せず、Stage 1、inference、submissionへ進まない。

この結果はexp209のpersistent joint state自体を否定しない。否定されたのは、
exp209の固定0.35 ft latticeと固定process varianceを保ったまま、各continuous-
kinematic edgeのmean/varianceを非負lattice projectionで完全保存するexp439の
表現契約である。exp438など別仮説の結果も再分類しない。
