# タスクリスト

## TODO

- TODO
- backlog候補から実験化する場合は、対応する上位仮説、`docs/backlog/<candidate>.md`、根拠ファイルを読む。
- 固定するもの、変更するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を提示し、重要な差分または未決事項を実装前にユーザーへ確認する。
- backlog詳細の上位仮説ID、契約、根拠、判断履歴を`requirements.md`へ欠落なく移し、`config.yaml`の`lineage.hypothesis_id`と`lineage.backlog_candidate`へ同じIDと候補名を設定する。移行確認後に`kaggle-strategy`へ元の詳細ファイルと未着手バックログ行の削除、および「検証中の仮説」の対応候補から対応実験への更新を引き渡す。この手順からbacklogを直接変更しない。
- 依頼原文と一次資料 / 参照実装から手法契約を抽出する。
- `input / target / output / loss / decode / context unit` は`requirements.md`だけに記録する。
- `design.md`には手法契約を複製せず、各契約項目の実装箇所、処理方法、承認済み差分を記録する。
- 実装する処理と省略する処理を具体的に記録し、`docs/glossary.md`に定義した実装区分のいずれかに整理する。
- `proxy` の場合は、省略機構、検証できない主張、追加コストを示し、ユーザー承認を記録するまで実装を開始しない。
- 同じ親 / familyの直近実験を確認し、target、output、decode、context unitを変える案まで見直す必要があるか判断する。
- 実験名が実装機構を過大に主張していないことを確認する。
- 再現性設計を `design.md` に記入する。
- stochastic 処理がある場合は stable seed policy を実装し、global RNG / thread scheduling 依存がないことを確認する。
- Kaggle push 前に`push-kaggle-notebook`のvalidatorを通し、現在の正のNotebook・設定、package、bootstrap ZIPの整合を確認する。
- output取得後にfeature content SHA、対象に応じた`oof_prediction_sha`または`test_prediction_content_sha`、`submission_sha`、model SHAを`metrics.json`の`evidence`へ記録する。
- 実装完了時に手法契約とコードの差分を再監査する。
- negative resultによって否定できる情報源、データ表現、使用方法、融合方法、検証条件、計算条件と、残ったpositive submetric / oracle headroom / coverage / 誤差非相関性を `result.md` に具体的に記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- なし
