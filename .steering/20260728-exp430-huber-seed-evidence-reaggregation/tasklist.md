# タスクリスト

## 設計（今回）

- [x] exp430 の experiment/steering 雛形を作成
- [x] 親実験と route を固定
- [x] Gaussian/Huber の式、delta、temperature、共通 trajectory 契約を固定
- [x] preflight/full の実行量と promotion gate を固定
- [x] 再現性・truth-late・SHA 契約を記載
- [x] README、SESSION_NOTES、result、metrics を design-only に更新
- [x] `KAGGLE_DIRECTION.md` と `experiment_summary.md` に登録
- [x] repository validation を通す

## 実装

- [x] 明示実装承認を記録
- [x] Jupytext起点のcompact self-contained train / inferenceを作成
- [x] fixed4 preflight、full4 shard、truth-late mergeを実装
- [x] float64 trajectory freeze、Gaussian/Huber evidence、SHA契約を実装
- [x] 専用contract test、構文、Ruff、Jupytext round-tripを通過
- [x] 正規train / inference notebookへ採用
- [x] Kaggle packageを作成

## 実行以降

- [x] fixed 4-well technical preflight を実行
- [x] technical gate を監査し、実行量を SESSION_NOTES に再確認
- [x] full 4 shardの3+1実行承認を記録
- [x] full trajectory replay を Kaggle へ push
- [x] shard 0--2を先行push
- [x] exp429 merge完了後にshard 3をpush
- [x] full 4 shardのterminal stateとtechnical summaryを監査
- [x] 4 shardのsummary SHAとmerge input rootを固定
- [x] preflight metrics/artifact/SHA を記録
- [x] strict truth-late mergeの実行承認を記録
- [x] promotion gate を判定（technical PASS / scientific FAIL）
- [x] gate FAILによりinference / submissionを無効のままterminal close
