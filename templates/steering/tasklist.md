# タスクリスト

## TODO

- TODO
- backlog候補から実験化する場合は、対応する `docs/backlog/<candidate>.md` と根拠ファイルを読む。
- 固定するもの、変更するもの、最小検証、成功条件、停止条件、実行しないこと、未決事項を提示し、重要な差分または未決事項を実装前にユーザーへ確認する。
- backlog詳細をsteering docsへ欠落なく移し、移行確認後に`kaggle-strategy`へ元の詳細ファイルと未着手バックログ行の削除を引き渡す。この手順からbacklogを直接変更しない。
- 依頼原文と一次資料 / 参照実装から手法契約を抽出する。
- `input / target / output / loss / decode / context unit` を `requirements.md` と `design.md` に記録する。
- 実装区分を `faithful` / `staged-faithful` / `proxy` のいずれかに固定する。
- `proxy` の場合は、省略機構、検証できない主張、追加コストを示し、ユーザー承認を記録するまで実装を開始しない。
- 同じ親 / familyの直近実験を分類し、representation auditの発動条件を確認する。
- 実験名が実装機構を過大に主張していないことを確認する。
- 再現性設計を `design.md` に記入する。
- stochastic 処理がある場合は stable seed policy を実装し、global RNG / thread scheduling 依存がないことを確認する。
- Kaggle push 前に metadata と bootstrap 内 config の整合を確認する。
- output 取得後に feature content SHA、prediction SHA、submission SHA、model SHA を記録する。
- 実装完了時に手法契約とコードの差分を再監査する。
- negative resultが閉じるtupleと、残ったpositive submetric / oracle headroom / coverage / 誤差非相関性を `result.md` に記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- なし
