# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- 後続補正、inference、submissionは未承認。

## 検証

- [x] compact train/inferenceのJupytext変換、`--test`、構文、ruffを確認した。
- [x] exp352専用contract tests 5件と親exp311を含む関連12件を確認した。
- [x] `make validate-exp`と`make validate-template`を確認した。

## 完了

- 2026-07-23: 独立後継としてsteering、固定gate、実行量、fail-closed境界を確定した。
- 2026-07-23: stochastic処理なし、0-model/0-booster/0-HMMの再現性設計を記録した。
- 2026-07-23: compact self-contained train/inference候補とcontract testsを実装した。
- 2026-07-23: exp311保存scoreに合わせ、数値閾値を維持したまま単位をhorizontal GR APIへ訂正した。
- 2026-07-23: compact trainの正規Notebook採用とKaggle CPU Stage 0 version 1実行が承認された。
- 2026-07-23: package config/bootstrap SHAを照合し、Kaggle CPU version 1を完了した。
- 2026-07-23: 8 checks中7 PASS、worst-well safetyだけFAILとしてbranchを閉じた。
- 2026-07-23: availability/fallback、score、surface metrics、gate、summaryのSHAを照合した。
