# タスクリスト

## TODO

- Stage 0全gate PASS時だけ40 selector modelsの実装・実行量とcontrol再学習0を提示して再承認を得る。
- raw-test prior regeneration、inference、submissionはStage 1 PASS後も別判断とする。

## 進行中

- なし

## ブロック中

- なし。Stage 0 FAILによりStage 1、inference、submissionは不適格。

## 完了

- 2026-07-23: exp311/312/313/315出力を使わない独立後継としてsteeringを確定した。
- 2026-07-23: Stage 0/1のgate、固定family/shrinkage/fallback、実行量、SHA方針を確定した。
- 2026-07-23: ユーザーの実装依頼により、Stage 0 prior generator、stable shuffle、
  compact self-contained train/inference候補、contract testsを実装した。
- 2026-07-23: 合成contract tests 7件、py_compile、ruff、Jupytext変換/`--test`をPASSした。
- 2026-07-23: ユーザーの「実行してください」を、正規train Notebook採用、
  Kaggle CPU package/push/run、Stage 0完了監視までの承認として受領した。
- 2026-07-23: Kaggle CPU version 1（id_no `128363177`）を完了した。
  real-minus-shuffle Spearman `-0.001290 < +0.05`で固定gateをFAILし、
  救済、再実行、Stage 1、inference、submissionなしでbranchを閉じた。
