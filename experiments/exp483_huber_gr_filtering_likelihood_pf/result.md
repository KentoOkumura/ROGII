# exp483_huber_gr_filtering_likelihood_pf 結果

## 結論

Kaggle private CPU kernel version 2で全773 wellsのStage 1を完了した。
技術ゲートはすべてPASSしたが、Huber PFは保存exp404 controlより
pooled RMSEを`0.180882522 ft`悪化させ、科学ゲートをFAILした。
事前登録どおりHuber/PF/blend/selectorの救済探索を行わず、branchを閉じる。
inferenceとsubmissionは実行していない。

## 仮説と一因子変更

fixed Huber particle likelihoodがGR outlierによるPF mode lossを減らす、という
仮説を検証した。exp404 x1.0 PFのper-particle Gaussian GR filtering scoreだけを
Huber `delta=1.345`へ置換し、GR scale、500 particles、128 stable seeds、
PF dynamics、resampling、roughening、missing-GR処理、T=5集約は固定した。
保存control、HMM、Beam、model、booster、GPUは再実行していない。

## Stage 1主要結果

| scope | candidate RMSE | control RMSE | 改善量（正が改善） | 判定 |
|---|---:|---:|---:|---|
| overall | 11.095404595 | 10.914522073 | -0.180882522 | FAIL |
| fold 0 | 9.604710258 | 9.360014232 | -0.244696026 | FAIL |
| fold 1 | 10.889070772 | 10.979418534 | +0.090347762 | PASS |
| fold 2 | 10.606929729 | 10.694277027 | +0.087347298 | PASS |
| fold 3 | 10.637706694 | 10.747502029 | +0.109795335 | PASS |
| fold 4 | 13.293923425 | 12.482449117 | -0.811474308 | FAIL |

改善foldは`3 / 5`で、必要条件`4 / 5`を満たさなかった。特にfold 4の
`0.811474308 ft`悪化が大きい。

## 固定scopeとtail

| scope | 改善量（正が改善） | 判定 |
|---|---:|---|
| raw GR observed | -0.253381263 ft | FAIL |
| raw GR missing | -0.023391682 ft | FAIL |
| high missing fraction | +0.012222398 ft | PASS |
| MD since 1000+ | -0.208228055 ft | FAIL |
| hidden-like spatial | +0.145778317 ft | PASS |
| hidden-like typewell-purged | -0.109562968 ft | FAIL |

- improved / worsened wells: `369 / 404`（全773 wells）
- by-well delta RMSE p95: `+0.520909635 ft`（上限`0.0 ft`）
- worst well: `70e1788b`
- worst-well candidate / control: `37.587237674 / 4.128715142 ft`
- worst-well regression: `+33.458522531 ft`（上限`+0.25 ft`）
- fixed exp209 HMM/PF 50:50: candidate `10.162155358`、
  control `10.084909849`、`+0.077245509 ft`悪化

GR observed、長いsuffix、typewell-purged、well-tail、固定blendで悪化した。
Huber化は一部scopeと3 foldsを改善したが、wrong basinに入るwellの大きな悪化を
抑えられず、全件でGaussian filtering likelihoodを置き換える根拠にはならない。

## 技術ゲート

全技術チェックをPASSした。

- rows / wells / folds: `3,783,989 / 773 / 5`
- candidate PF well-runs: `773`
- seed-well trajectories: `98,944`
- particle starts: `49,472,000`
- control PF / HMM / Beam / LightGBM / booster / GPU rerun: すべて`0`
- prediction finite coverage、raw input identity、保存control parity、
  fixed HMM/PF control parity、SHA readback: PASS
- freeze前のtruth / control / fold / hidden-like role read: すべて`0`
- runtime to prediction freeze / total:
  `12,361.117 / 12,454.354 sec`
- peak RSS: `3.566319 GB`

Stage 0のfull projection `6,855.083 sec`より実測は長かったが、事前上限
`30,600 sec`とRSS上限`25 GB`は満たした。

## Stage 0参考結果

Stage 0はfixed32 technical preflightで、CVやpromotion判定ではない。
10/10 technical gateをPASSし、candidate / 保存controlは
`9.811671590 / 9.616740808 ft`、差`+0.194930782 ft`だった。

## 再現性

- deterministic anchor: no
- scientific contract SHA:
  `089765cb14c395c1ff678d93c4a4940481aa7a8b846811287a7168ddda25d3c6`
- Stage 1 prediction logical SHA:
  `5a3c58aaaa1f9810cacc78836a87dc2ac06a8f2be614253d948b5a76056c3ad2`
- prediction raw gzip SHA:
  `f805d83f4d6cc7e60a24d033333eebf1ddcfcb0e6209870d99c000a0831cad62`
- prediction decompressed SHA:
  `c0e2ea557d73eed0b463dfc4c4b17c3621cd3199a9d255e302e5ac35b91a274e`
- promotion gate SHA:
  `a6caa2259f90c13970313707bd9b10b568273e6b06745cce6d831585250dfe0c`
- summary SHA:
  `095f88f44af087145d602f1f0af41fd69ffce6cd5eaccf2047a26a59d9768b71`
- local notebook実行: なし

## 最終判断

`terminal_close_without_huber_or_pf_rescue`。
delta、scale、temperature、clip、mixture、particle/seed、transition、
resampling、well/row gate、blend/selector、same-OOF rescueは行わない。
この結果だけに依存する後続候補は追加せず、既存バックログを優先する。
