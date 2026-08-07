# exp484_student_t_gr_filtering_likelihood_pf 結果

## 結論

Kaggle private CPU kernel version 3で全773 wellsのStage 1を完了した。
18件の技術ゲートはすべてPASSしたが、Student-t PFの保存exp404 control比改善は
`0.017424990 ft`に留まり、必要な`0.05 ft`を満たさなかった。
改善foldも`2 / 5`で必要な`4 / 5`を満たさず、科学ゲートをFAILした。
事前登録どおりStudent-t/PF/blend/selectorの救済探索を行わずbranchを閉じる。
inferenceとsubmissionは実行していない。

## 仮説と一因子変更

fixed `df=4` Student-t particle likelihoodがlarge GR residualによるPF mode lossを
減らす、という仮説を検証した。exp404 x1.0 PFのper-particle Gaussian GR
filtering scoreだけをStudent-tへ置換し、GR scale、500 particles、
128 stable seeds、PF dynamics、resampling、roughening、missing-GR処理、
T=5集約は固定した。保存control、HMM、Beam、model、booster、GPUは
再実行していない。

## Stage 1主要結果

| scope | candidate RMSE | control RMSE | 改善量（正が改善） | 判定 |
|---|---:|---:|---:|---|
| overall | 10.897096923 | 10.914521913 | +0.017424990 | FAIL |
| fold 0 | 9.464670922 | 9.360014494 | -0.104656427 | FAIL |
| fold 1 | 11.126836844 | 10.979418436 | -0.147418408 | FAIL |
| fold 2 | 10.771522717 | 10.694277115 | -0.077245602 | FAIL |
| fold 3 | 10.662625585 | 10.747501382 | +0.084875797 | PASS |
| fold 4 | 12.206968745 | 12.482448796 | +0.275480050 | PASS |

改善foldは`2 / 5`で、fold 0–2が悪化した。MAEは
`6.349282517 / 6.429207277 ft`で改善したが、評価指標と事前固定gateはRMSEであり、
promotion根拠には使わない。

## 固定scopeとtail

| scope | 改善量（正が改善） | 判定 |
|---|---:|---|
| raw GR observed | -0.068900357 ft | FAIL |
| raw GR missing | +0.205368304 ft | PASS |
| high missing fraction | +0.165192042 ft | PASS |
| MD since 1000+ | +0.028529855 ft | PASS |
| hidden-like spatial | +0.129976587 ft | PASS |
| hidden-like typewell-purged | -0.130146256 ft | FAIL |

- improved / worsened wells: `442 / 331`（全773 wells）
- by-well delta RMSE p95: `+1.455066656 ft`（上限`0.0 ft`）
- worst well: `d924e971`
- worst-well candidate / control:
  `29.462353025 / 12.797463292 ft`
- worst-well regression: `+16.664889733 ft`（上限`+0.25 ft`）
- fixed exp209 HMM/PF 50:50: candidate `10.067803689`、
  control `10.084909779`、`+0.017106090 ft`改善でguard PASS

Student-t化はmissing-GR、高missing、長いsuffix、spatial holdoutでは改善したが、
実観測GRとtypewell-purgedで悪化した。過半数のwellは改善した一方、
少数wellの大きなwrong-basin悪化がpooled RMSEとtailを壊した。
fixed HMM/PF blend guardのPASSはprimary gateのFAILを救済しない。

## 技術ゲート

18/18 technical checksをPASSした。

- rows / wells / folds: `3,783,989 / 773 / 5`
- candidate PF well-runs: `773`
- seed-well trajectories: `98,944`
- particle starts: `49,472,000`
- control PF / HMM / Beam / LightGBM / booster / GPU rerun: すべて`0`
- prediction finite coverage、raw input identity、formula、stable seed、
  ESS/resampling、保存control parity、fixed blend parity、SHA readback: PASS
- freeze前のtruth / control / fold / hidden-like role read: すべて`0`
- runtime to prediction freeze / total:
  `10,871.426 / 10,959.720 sec`
- peak RSS: `3.309631 GB`

Stage 0のfull projection `15,199.523 sec`より実測は短く、事前上限
`30,600 sec`とRSS上限`25 GB`を満たした。

## Stage 0参考結果

Stage 0はfixed32 technical preflightで、CVやpromotion判定ではない。
16/16 technical gateをPASSし、candidate / 保存controlは
`16.536326063 / 17.358983593 ft`、改善`+0.822657530 ft`だった。
この方向性は全773-well Stage 1では再現しなかった。

## 再現性

- deterministic anchor: no
- scientific contract SHA:
  `af07896332346cccf722bcedc1cee5c371d93089e9fab2a49e19cada2cb5cc36`
- Stage 1 prediction logical SHA:
  `4dbe939363b1522dbc521680cd25d3ce7993ff8b94ddbdbf30b95073db2b28f4`
- prediction raw gzip SHA:
  `e1344c2dc63dda8905b997bb82b8c3e25ea9df3cfade2688e2e48dab9f8cc655`
- prediction decompressed SHA:
  `1b02c68b0e52cd031b9d06e931f8fa92e2fb841dbc9e4ee6c7778807af5f1962`
- primary / by-well / fixed-blend metrics SHA:
  - `d8c818d7bb52ee532119db158095e838d5a2a0f0ad8bf0e8294ebc0a548541c1`
  - `d82f7a7c6496fa84d80c72d1b35a8b5593fa7cc2f9223cafe7fd2773266c5b0f`
  - `b8c6687d4871796e3a63b2ff38a50a3d4a631ba8441ec3075d9f4db3e164a712`
- promotion gate SHA:
  `32a5c8473a4c94eb6ebbad8461332be5ffa40150f1cbeb9af2726b0db56afb50`
- summary SHA:
  `51d9658ea32fa45b0724b543169661ba36fea5f61b9e331d327dfd6e0db1a380`
- local notebook実行: なし

## 最終判断

`terminal_close_without_student_t_or_pf_rescue`。
df、scale、temperature、clip、Gaussian mixture、particle/seed、transition、
resampling、well/row gate、blend/selector、same-OOF rescueは行わない。
この結果だけに依存する後続候補は追加せず、既存バックログを優先する。
