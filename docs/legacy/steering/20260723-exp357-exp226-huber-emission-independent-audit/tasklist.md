# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- inference、submissionは未承認。

## 完了

- 2026-07-23: exp357として採番し、exp342 dependencyを削除した。
- 2026-07-23: delta 1.345、score bank、control、stress gate、absolute ceilingを固定した。
- 2026-07-23: exp281/342のnegative evidenceをリスクとして記録した。
- 2026-07-24: ユーザー依頼によりStage 0 compact self-contained train候補と
  fail-closed inference候補を実装した。
- 2026-07-24: Huber piecewise loss、exp280 Gaussian parity、13-shift/block identity、
  deterministic circular control、truth late-join、固定gateの専用test 7件を通過した。
- 2026-07-24: compact候補のJupytext変換・round-trip、py_compile、Ruff、
  strict experiment validationを通過した。正規Notebookは未変更。
- 2026-07-24: ユーザーが正規train Notebook採用とKaggle private CPU Stage 0を承認した。
- 2026-07-24: canonical private CPU kernel version 1（id_no `128448451`）を
  `319.617349 sec`で完了した。
- 2026-07-24: technical / circular / extreme-residual guardはPASSしたが、
  pooled gain、4/5-fold一貫性、stress非悪化をFAILした。
- 2026-07-24: decision=`stage_0_failed_close_without_rescue`として、
  Stage 1、inference、submission、救済調整なしで完了した。
- 2026-07-24: ユーザーがStage 0 FAILを明示overrideし、fixed Huber
  1 variant / 773 HMM runsのStage 1実行を承認した。
- 2026-07-24: canonical private CPU kernel version 2を`9597.242200 sec`で完了し、
  actual HMM RMSEをexp281 `9.827420`から`9.737195`へ`0.090225 ft`改善した。
- 2026-07-24: 4/5 foldsとrequired scopeは改善したが、by-well p95
  `+0.003365 ft`、worst well `+1.403715 ft`、exp226 direct ceiling
  `+0.310086 ft`をFAILした。
- 2026-07-24: decision=`stage_1_failed_close_without_rescue`として、
  救済、再実行、inference、submissionなしでbranchを閉じた。
