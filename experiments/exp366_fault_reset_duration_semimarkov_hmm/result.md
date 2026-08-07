# exp366_fault_reset_duration_semimarkov_hmm 結果

## 結論

Kaggle private CPUのStage 0 version 2はtechnical gateを通過したが、固定scientific
gateを通過しなかった。判定は`stage0_failed_close_without_semimarkov_hmm`であり、
Stage 1 semi-Markov HMM、inference、submissionへ進めない。

- Kernel:
  `kentookumura/exp366-fault-reset-duration-semimarkov-hmm-train`
- Version / id_no: `2 / 128543224`
- 状態: `COMPLETE`
- Stage 0本体: `666.798832 sec`
- Kaggle job: 約`695.920789 sec`
- 評価量: `3,783,989 rows / 773 wells / 3,389,090 eligible rows`
- 実行量: diagnostic `1`、fixed branches `13`、reporting folds `5`
- semi-Markov HMM well-run / LightGBM config / trained fold / booster /
  parent control rerun / GPU: 全て`0`

## 固定scientific gate

| 判定 | 実測 | 閾値 | 結果 |
| --- | ---: | ---: | --- |
| trigger bad-event AUC | `0.500004` | `>=0.60` | FAIL |
| AUC gain vs 512-row circular | `0.000003` | `>=0.05` | FAIL |
| trigger row fraction | `0.0000118` | `[0.001, 0.10]` | FAIL |
| alternative branch within-10 coverage | `0.900000` | `>=0.60` | PASS |
| MRR gain vs base-first | `-0.123356` | `>=0.01` | FAIL |
| passing folds | `0 / 5` | `>=4 / 5` | FAIL |
| hidden-like spatial selected gain | `-0.344537 ft` | `>0` | FAIL |
| hidden-like typewell-purged selected gain | `-0.666357 ft` | `>0` | FAIL |

発火は40 events / 30 wellsだけで、eligible行の`0.0011803%`だった。trigger AUCは
circular controlとともにほぼ0.5で、raw GR changeとexp209 emission surpriseのq99.5
ANDはbad eventを識別しない。

固定jump family自体にはoracle headroomがあり、alternative within-10 coverageは
`0.90`、base / oracle RMSEは`17.892031 / 12.500137 ft`だった。一方、GR evidenceで
選んだbranchのRMSEは`18.897338 ft`で、baseより`1.005307 ft`悪化した。evidence MRRも
base-firstを`0.123356`下回り、40 eventsのselected-oracle率は`0.20`だった。したがって、
branch集合ではなくtarget-free triggerとGR-only branch selectionの識別性が不足している。

## Technical / 再現性

- freeze前truth / hidden-like role read: `0 / 0`
- freeze後truth / hidden-like role read: `5,090,197 / 773`
- raw well identity:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- saved exp209 decompressed SHA:
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- scientific contract SHA:
  `f59ad2d8ac0b084c23461805c4f393c6938cd17e953c3a922fd4fe531905c604`
- trigger ledger decompressed SHA:
  `5f2bce9e53245b0dd2e19a30364ed0504cc52d1f9f759f7bd77788d9b9ff9f51`
- branch ledger decompressed SHA:
  `e4f9fa04318fcf1856d370334e272b62f83e22d01da2c2b82b0be4aa3f913800`
- summary SHA:
  `cc048adc957ab8e71148c3dd68c3767157b9fdaba754c40c874f71643d99e7e6`

選択取得したinput manifest、branch ledger、event readout、scope/fold metrics、
gate reportはすべてKaggle summaryのraw / decompressed SHAと一致した。126.8 MBの
trigger readoutと58.7 MBのtrigger ledgerはsummary / freeze manifestのSHAだけを記録し、
ローカルへ全量取得していない。

version 1は約61秒で、科学処理前のraw well identity guardに停止した。親実験群の
column-aware logical SHAに対してexp366だけがCSV serialization SHAを使った実装誤りであり、
raw input変更ではなかった。version 2ではraw identity専用adapterだけを修正し、scientific
contract、trigger、branch、gate、実行量は変更していない。

## 判断

q99.5 threshold、AND条件、refractory、jump、duration、margin、negative controlを
事後変更して救済しない。Stage 1の773 semi-Markov HMM runsは未実装・不適格のまま閉じ、
inferenceとsubmissionも生成しない。この結果はexp289/290/231のnegative evidenceを補強し、
同じGR-only fault/reset familyを再開する新規backlogは追加しない。
