# exp428_similar_well_gr_registration_map_transfer_readout 結果

## 状態

Kaggle CPU Stage 0を完了し、固定technical gateをFAILしたためno-rescueで閉鎖した。
CV / LB / submissionは対象外。

## 仮説

同じType Well GR波形を持つwellの中では、Horizontal GR波形が似たdonorで得た
Type Well–Horizontal GR registration offset（何ftずらすと一致するか）が
held-out queryへ転送できる。

## 固定設定

- 親: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- Route: `pf_beam`
- 検証: 5-fold outer-valid pseudo-tail、0-model、equal-well shift MAE
- primary: `selected_top1_global_shift`
- donor選択: same-Type-Well内のHorizontal GR constrained-DTW rank 1
- registration: 512-row block、固定13 shifts、raw-finite ZNCC
- query truth: candidate/artifact freeze後だけ使用
- LightGBM / trained fold / booster / PF / HMM / Beam / GPU: すべて0

## 結果

canonical private CPU kernel version 2（id_no `128932184`）を約`225.6 sec`で完了した。
version 1はsupport maskをDTW入力にも適用して全pairを失う実装バグだったため、親exp423と
同じ決定的補間へ戻し、15件目の回帰testを追加してversion 2を実行した。

version 2の固定gate結果:

- supported query wells: `306 / 773 = 0.395860`。下限`0.70`をFAIL。
- post-freeze評価可能 wells: `290`。
- supported prediction finite fraction: `1.0`、PASS。
- identifiable query block fraction: `0.582638`、PASS。
- fold complete、donor/query交差0、freeze前query truth read 0、
  Type Well axis graph conflict 0: すべてPASS。
- deterministic SHA: support FAILが既に確定したため独立rerunを行わずpending。
- technical / scientific / local-shape gate: `FAIL / FAIL / FAIL`。
- terminal decision: `invalid_or_insufficient_registration_support`。

technical gate外のsupported-only参考値:

| candidate | wells | global shift MAE (ft) | zero比gain (ft) |
| --- | ---: | ---: | ---: |
| selected top-1 donor | 290 | 2.529310 | -1.424138 |
| zero shift | 290 | 1.105172 | 0 |
| stable random donor | 290 | 1.808621 | - |
| same-group median | 290 | 1.398276 | - |
| post-freeze top-5 oracle | 290 | 1.118966 | -0.013793 |

primaryはzeroを`1.424138 ft`悪化し、改善foldは`0 / 5`。top-5 oracleもzeroより
`0.013793 ft`悪く、cross-well offset転写のheadroomを示さなかった。DTW costと転写
誤差のpooled Spearmanは`0.075211`、mean ZNCC gainは`-0.057438`。local mapも
global shiftよりblock MAEを`5.050144 ft`悪化した。

## 再現性

- deterministic anchor: いいえ（technical support FAIL後は独立rerun不要と判断）
- seed policy: deterministic matching + stable SHA256 random control
- kernel: `kentookumura/exp428-gr-registration-map-transfer-readout-train` version 2、
  id_no `128932184`
- scientific contract SHA:
  `f3f084e9769da5b24c7c18497c2de101e025e8b340bbec797a554d1bb8f2cdf9`
- target-free logical content SHA:
  `54127363c066b75180af274e8bb4e076536d97c94554ccec2846f4986ff849d7`
- query truth rows: freeze前`0`、freeze後`3,783,989`
- model / prediction / submission SHA: 対象外

## 解釈

technical coverageが低いため、全wellへの一般化可能な科学結論としては不成立である。
ただし評価可能な290 wellsでも、rank-1、random、group median、top-5 oracleのすべてが
zero shiftを上回らず、registration offsetのcross-well転写を支持する兆候もなかった。
same-Type-Well / Horizontal-GR-DTWに基づくこの転写branchは閉じる。

## 次

support threshold、group、DTW、shift grid、block、primaryを同じOOFで救済しない。
inference、submission、HMM/PF/Beam observation offset統合へ進まない。原因追跡が別途必要に
なった場合だけ、保存artifactを使う0-prediction support/zero-dominance attributionを
別実験・別承認で行う。
