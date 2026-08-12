# 要件

## Backlogからの引き継ぎと承認

- 移行元backlog: TODO (`N/A` または移行した `docs/backlog/<candidate>.md`。内容の移行確認後、元ファイルと未着手行の削除を`kaggle-strategy`へ引き渡す)
- backlog時の状態と実験化承認: TODO (`N/A` または状態、承認日時 / 依頼メッセージ)
- 親実験: TODO
- 根拠 / 一次資料 / 参照実装: TODO
- 固定するもの: TODO
- 変更するもの: TODO
- 最小の反証可能な検証: TODO
- 成功条件: TODO
- 停止条件: TODO
- 実行しないこと: TODO
- 未決事項: TODO（実装開始時は`なし`。残る場合はユーザー確認まで実装しない）
- backlog記録から解釈を変更した箇所とユーザー承認: TODO (`N/A` または承認日時 / 依頼メッセージ)

## 判断履歴

- YYYY-MM-DD: TODO（backlogでの判断、実験化承認、契約変更の承認を時系列で移す）

## 手法契約

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。ユーザーへの説明では、先に実装する処理と省略する処理を具体的に示し、その後に必要ならラベルを添える。

- 依頼原文: TODO
- 期待する成果: TODO
- input: TODO
- target / objective: TODO
- output: TODO
- loss: TODO
- decode: TODO
- context unit: TODO (`row`、`window`、`whole-well`、`set`、`field` など)
- 実装区分: TODO (`faithful`、`staged-faithful`、`proxy`)
- 省略する機構と理由: TODO
- proxyで検証できない主張: TODO
- proxyの場合のユーザー承認: TODO (`N/A` または承認日時 / 依頼メッセージ)
- この実験が支持 / 棄却できる主張: TODO
- この実験では判断できない主張: TODO

## 制約

- Route: TODO (`ml_model`、`pf_beam`、`ensemble`)
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- TODO

## 受け入れ基準

- 手法契約の `input / target / output / loss / decode / context unit` がコードと一致する。
- 実装区分と実験名が実装した機構を正確に表す。
- `proxy` の場合は、省略点、検証不能な主張、ユーザー承認が記録されている。
- backlogから移行した場合は、この文書に根拠、親実験との差分、成功条件、停止条件、禁止する代替実装、判断履歴が欠落していない。
- TODO
- deterministic anchor として扱う場合は、feature content SHA、model SHA、対象に応じた`oof_prediction_sha`または`test_prediction_content_sha`、`submission_sha`、Kaggle kernel versionを`metrics.json`の`evidence`へ記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
