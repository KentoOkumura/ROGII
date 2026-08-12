# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- Stage 1 GPU学習、raw-test feature regeneration、inference、submissionは未承認。

## 完了

- 2026-07-23: 旧exp314をreopenせず、0-booster preflightを持つ独立後継として設計を固定した。
- 2026-07-23: Stage 0/1のgate、6列schema、15 booster予約、再現性境界を確定した。
- 2026-07-23: exp352の平均signal改善後に次へ進むユーザー指示をStage 0実装とKaggle CPU実行の承認として受領した。
- 2026-07-23: compact self-contained train/inference、正規train Notebook、contract testsを実装した。
- 2026-07-23: exp148 OOF surfaceが水平well全5,092,255行ではなく`TVT_input`欠損3,783,989行であることをtarget-free行数監査で確認し、fold重みを修正した。
- 2026-07-23: Kaggle private CPU version 1（id_no `128362932`）でStage 0を完了した。
- 2026-07-23: coverage/fallback/finite/freeze/4-fold方向はPASSしたが、pooled Spearman、q4-q1、real-minus-shuffleがFAILしたためbranchを閉じた。
- 2026-07-23: Stage 1、列選択、救済調整、再実行、raw-test、inference、submissionを行わないと記録した。
