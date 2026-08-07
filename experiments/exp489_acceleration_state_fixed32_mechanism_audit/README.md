# exp489 acceleration-state fixed32 mechanism audit

exp444の3状態acceleration exact-HMMを、exp458 v2の高速scaled CPU engineで
fixed32 wellまで進めるStage 0B機序監査です。Routeは`pf_beam`です。

exp458はexact parityの凍結閾値をわずかに超えたためFAILのまま保持します。
本実験は、その最大差（mean 0.000105 ft、std 0.000064 ft、acceleration
posterior 0.000009）をユーザーがStage 0B用途に限って許容した、という
明示waiverに基づきます。exact-equivalentやbitwise parity PASSとは扱いません。

## 状態

Kaggle private CPU version 1でStage 0Bを完了しました。technical gateは
全PASS、mechanism gateはFAILで、Stage 1へ進まずterminal closeです。

## 仮説

exp458で観測した約`1e-4 ft`以下の予測差を許容してfixed32へ広げても、
凍結したacceleration stateがpersistent rate lagの向きと誤差を改善し、
matched controlを悪化させない。

## 実行範囲

- scientific variant: 1
- 再利用: exp458 v2のfixed4（21,962 rows）
- 新規計算: 残り28 well
- 合計評価: fixed32、156,088 suffix rows
- LightGBM config / fold / booster / fitted model: 0
- PF / Beam / GPU: 0
- Stage 1 / inference / submission: 未承認、未実行

全32 wellのprediction、acceleration posterior、target-free diagnosticを
SHA freezeした後にのみ、truth、role/fold、persistent episode/cause、
保存済みexp209 controlを読みます。判定閾値はexp444 Stage 0Bから変更しません。

## 検証方針

fixed32はmechanism preflightでありCVではありません。posterior accelerationと
future true rate curvatureの符号一致、episode SSE、persistent well/fold改善数、
matched exp209 control safetyを、exp444で事前凍結した閾値によりfail-closedで
判定します。

## 所見

technical 10項目はすべてPASSしました。posterior acceleration nonzero massは
`0.664839`でstate collapseはありません。一方、future curvatureとの符号一致は
`0.500309`、positive foldは`0/5`で、加速度方向を識別できていません。
persistent episode SSEは`-3.6667%`（悪化）、改善は`8/16 wells`・`2/5 folds`、
forward-cause SSE改善も`0.4355%`に留まりました。

matched exp209 controlはpooled RMSE delta `-0.162849 ft`、by-well delta p95
`+0.077808 ft`で安全性2項目だけPASSしました。つまりcontrol改善はあっても、
仮説対象のpersistent acceleration機序は再現せず、このbranchは不採用です。
