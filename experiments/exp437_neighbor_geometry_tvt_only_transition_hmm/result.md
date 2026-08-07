# exp437_neighbor_geometry_tvt_only_transition_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1完了。technical gateは全PASSしたが、
mechanism AND gateをFAILしたため`stage0_fail_closed`とする。fixed32は機構確認用で、
CVまたはpromotion evidenceではない。

## 仮説

exp435のTVT-only縮約を維持し、遷移中心だけを`-ΔZ`からfold-safeな
exp226 `tvt_geop`の隣接差へ置き換えると、persistent rate stateなしで
周辺井戸由来の符号付きgeometry driftを利用できるかを検証した。

## 実行

- Kernel:
  `kentookumura/exp437-neighbor-geometry-tvt-transition-hmm-train`
- Version / id_no: `1` / `129056603`
- Runtime: Kaggle private CPU、internet off、GPU 0
- Candidate: `neighbor_geometry_direct_transition` 1本
- 対象: fixed32 / 156,088 suffix rows
- Candidate HMM well-runs: 32
- parent/control HMM rerun、ML model、LightGBM config、trained fold、
  booster、PF、Beam: すべて0
- Stage 0 elapsed: `39.153270 sec`
- Candidate HMM total: `9.336708 sec`
- 773-well runtime projection: `225.539862 sec`
- Peak RSS: `0.415966 GB`

初回54文字slugはKaggle SaveKernel 400となり、notebook未作成を確認した。
同じexp437のまま`only`だけを省いた49文字のcanonical slugへ修正し、
version 1を正常完了した。科学コードと候補は変更していない。

## Technical gate

全項目PASS。

- 32 wells / 156,088 rows、5 folds、重複・欠損0
- exp226 source fold / fixed32 manifest fold一致率`1.0`
- forbidden geometry column、truth、roleのfreeze前read 0
- geometry first-difference parity最大差`0.0 ft`
- transition row-sum最大誤差`4.440892e-16`
- posterior normalization最大誤差`2.220446e-15`
- prediction readback logical SHA一致
- runtime / RSS guard PASS

## Mechanism結果

| Scope | Candidate RMSE | exp226 geometry RMSE | 差 |
| --- | ---: | ---: | ---: |
| fixed32 all | 13.019009088 | 9.267204778 | +3.751804309 |
| matched control 16 | 7.771561732 | 8.719886308 | -0.948324576 |
| persistent 16 | 16.592455298 | 9.768805034 | +6.823650264 |

matched controlではexp226 geometryと保存exp435 dz-only
`17.133652291`の両方を改善した。一方、仮説対象のpersistent 16では
`6.823650264 ft`悪化した。

| Fold | Candidate | exp226 geometry | 差 | 改善 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 16.211801295 | 7.387184874 | +8.824616421 | No |
| 1 | 19.646252787 | 11.891483061 | +7.754769726 | No |
| 2 | 8.672544426 | 8.522654192 | +0.149890234 | No |
| 3 | 6.897768249 | 9.236981905 | -2.339213655 | Yes |
| 4 | 8.003046189 | 9.621559153 | -1.618512963 | Yes |

- 改善fold: `2/5`（要求`>=4/5`）
- paired by-well delta p95: `+21.699228790 ft`
  （要求`<=+0.25 ft`）
- worst-well delta: `+24.452435654 ft`
  （要求`<=+2.0 ft`）
- mechanism gateはmatched-control 2項目だけPASSし、残り5項目FAIL。

## 再現性

- RNGなし、stable well / row / reduction order。
- prediction decompressed/logical SHA:
  `9eead0755e11fc5093ffedff59c9f4f3aeee3c7b5755d493af498fd4589bc2d8`
- schedule manifest SHA:
  `6d42fe8928c0f142902a3dd47679d64cdbac801eef5439c7dddb216d8e92b2d2`
- prediction manifest SHA:
  `d190c8fb9a92eb9ed3606205fcb9be1213f227a38b9529b0f07f95be5b3fb263`
- diagnostic manifest SHA:
  `a80b193bea1e4786cde21b1718dc1a3f172f681591d064aee04f23c8328e5646`
- 初回runだけなのでdeterministic anchorとは呼ばない。
- 科学値とSHAはKaggle logs / notebook cell outputを根拠とし、
  output archiveは取得していない。

## 解釈

TVT-only HMMへneighbor-geometry driftを直接入れるだけでは、geometry pathを安全に
GR補正できなかった。matched controlでは大きく改善したが、persistent sampleの
fold 0/1でmode slipが増え、全体とwell tailを強く悪化させた。したがって
exp435の失敗を`-ΔZ`中心だけに帰属する仮説は支持されない。

固定契約どおりtransition scale/noise、geometry clip、emission、grid、subset、
gate、blend、selectorのsame-OOF救済は行わない。exp355/394の既存FAILも含め、
geometry scheduleをTVT-only chainへ直接注入する枝を閉じる。

## 次

Stage 1、raw-test geometry再生成、inference、submissionへ進まない。
既存のexp438 U-state fixed lattice、exp439 continuous-kinematic joint transitionは
exp437救済ではなく独立仮説として扱う。
