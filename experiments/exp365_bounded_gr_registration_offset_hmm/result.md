# exp365_bounded_gr_registration_offset_hmm 結果

## 結論

Kaggle private CPUのStage 0 version 2はtechnical gateを全て通過したが、
固定scientific gateを通過しなかった。判定は
`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`であり、Stage 1 exact HMM、inference、
submissionへ進めない。

- Kernel:
  `kentookumura/exp365-bounded-gr-registration-offset-hmm-train`
- Version / id_no: `2 / 128537562`
- 状態: `COMPLETE`
- Stage 0本体: `408.720069 sec`
- 評価量: `773 wells / 18,465 windows / 915,301 observed held-out rows`
- 実行量: diagnostic `1`、offset states `5`、reporting folds `5`、
  resource wells `16`
- exact-HMM well-run / LightGBM config / trained fold / booster /
  parent control rerun / GPU: 全て`0`

## 固定scientific gate

| 判定 | 実測 | 閾値 | 結果 |
| --- | ---: | ---: | --- |
| real predictive NLL gain | `5.430399%` | `>=1%` | PASS |
| passing folds | `0 / 5` | `>=4 / 5` | FAIL |
| real-minus-circular NLL gain | `-9.881025%` | `>=0.5%` | FAIL |
| nonzero posterior mean | `0.489435` | `[0.05, 0.50]` | PASS |
| boundary posterior mean | `0.182671` | `<=0.25` | PASS |
| adjacent-window sign agreement | `0.580771` | `>=0.60` | FAIL |
| projected runtime | `56,429.34 sec` | `<=30,600 sec` | FAIL |
| projected peak RSS | `7.358320 GB` | `<=25 GB` | PASS |

real側はdelta=0より改善したが、missing maskと観測値multisetを保ったcircular controlの
改善率`15.311425%`を大きく下回った。fold別real gainは全foldで
`5.068742%--5.666089%`だった一方、real-minus-circularは全foldで負だった。
したがって、5-state registration posteriorの改善は実GR配列固有の登録信号ではなく、
offset latentが持つ一般的な局所適応力で説明できる。符号安定性とruntime gateも独立に
不合格であり、Stage 1へ進む根拠はない。

## Technical / 再現性

- `suffix_truth_columns_read = 0`
- `physical_prediction_rows = 0`
- `exact_hmm_well_runs = 0`
- technical gate: 全項目PASS
- contract SHA:
  `83a15a82a966a44837be5f7c22dece5160c2324e112949bab66895f39b7225d9`
- rolling ledger content SHA:
  `58e0db25485cef614f002b200e8123922a0d50156c968168d4fbd550d52d3896`
- delta posterior content SHA:
  `0b6715000c1a8ff74df41340594c0e0a49b4e2dac7b5467ceb5c5497c6e7f711`
- resource projection SHA:
  `aca965e6752cbf57ef29e9e0efd812676f231557407b806b04f97052868ce462`
- version 2 summary SHA:
  `99d76d3a66b7822e627e7a9295a27d116746b9083391643cbd6ff72bdcb0e2b8`

取得したfreeze対象5ファイルはraw/content SHAがmanifestと一致した。
version 1とversion 2のscientific contract、input manifest、rolling ledger、
delta posterior、resource projection、fold metricsもbyte単位で一致した。

version 1ではin-memory posteriorが`1.0`を機械精度だけ超える場合をstrict
`between(0,1)`が拾い、`posterior_in_unit_interval`だけtechnical falseになった。
version 2は科学条件を変えず、区間判定だけ`atol=1e-12`とした。これによりtechnical
passへ復旧し、scientific resultが変わっていないことをSHAで確認した。

## 判断

offset grid、transition、sigma、runtime係数、gate、circular controlを事後変更して
救済しない。Stage 1は未実装のまま閉じ、physical prediction、inference、
submissionは生成しない。
