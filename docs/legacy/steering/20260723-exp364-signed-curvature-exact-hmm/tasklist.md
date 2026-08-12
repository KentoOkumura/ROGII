# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- backlog、実験scaffold、steeringを作成した。
- 状態、固定値、Stage 0/1 gate、禁止事項、再現性、実行量を確定した。
- 2026-07-25にStage 0実装承認を受けた。
- complete 512-row signed-path readout、truth-late-join、16-well resource projectionの
  実装仕様を確定した。
- compact self-contained train / fail-closed inference候補と専用testを実装した。
- Jupytext変換、py_compile、ruff F821、専用9 tests、strict experiment validationを完了した。
- ユーザーの実行承認を記録し、正規train Notebookを採用した。
- private CPU、GPU / TPU / internet offのKaggle packageとbootstrap SHAを監査した。
- canonical kernel version 1（id_no `128529795`）をpushし、`COMPLETE`まで監視した。
- 772 wells / 13,631 blocksのStage 0を`224.737080 sec`で完了した。
- technical gate `12 / 12 PASS`、scientific gate `6 / 9 PASS`を確認した。
- real-minus-circular top1、passing folds、projected runtimeの3 gate FAILにより
  `STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`としてbranchを閉じた。
- Stage 1 exact HMM、inference、submissionを実装・実行しないと確定した。
- Stage 0成果物10件のraw / decompressed content SHAを再検証し、実験記録を更新した。
