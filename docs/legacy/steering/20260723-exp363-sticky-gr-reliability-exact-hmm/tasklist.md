# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- Stage 0 scientific FAILのためStage 1、推論、提出は不適格。

## 完了

- backlog、実験scaffold、steeringを作成した。
- 状態、固定値、Stage 0/1 gate、禁止事項、再現性、実行量を確定した。
- 2026-07-23 の後続承認により、Stage 0 compact self-contained train notebookを実装した。
- block tail、forward filter、posterior集約、circular control、quartile、fold passを
  configとcontract testで固定した。
- placeholder train/inference notebookを正規Notebookへ置換し、inferenceはfail-closedとした。
- 2026-07-24にKaggle private CPU Stage 0 version 1を完了した。
- technical gateはPASS、hidden-like spatial AUCとweak massのscientific gateはFAILした。
- decision `stage_0_failed_close_without_rescue`としてbranchを閉じた。
- Stage 1 decoder、推論、提出は実装・実施していない。
