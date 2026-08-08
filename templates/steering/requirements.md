# 要件

## 依頼と手法契約

- 依頼原文: TODO
- 移行元backlog: TODO (`N/A` または移行した `docs/backlog/<candidate>.md`。内容の移行確認後、元ファイルと未着手行の削除を`kaggle-strategy`へ引き渡す)
- backlog時の状態と実験化承認: TODO (`N/A` または状態、承認日時 / 依頼メッセージ)
- 期待する成果: TODO
- 一次資料 / 参照実装: TODO
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

## 制約

- Route: TODO (`ml_model`、`pf_beam`、`ensemble`)
- 固定するもの: TODO
- 実行しないこと: TODO
- 未決事項: TODO（実装開始時は`なし`。残る場合はユーザー確認まで実装しない）
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- TODO

## 受け入れ基準

- 手法契約の `input / target / output / loss / decode / context unit` がコードと一致する。
- 実装区分と実験名が実装した機構を正確に表す。
- `proxy` の場合は、省略点、検証不能な主張、ユーザー承認が記録されている。
- backlogから移行した場合は、根拠、親実験との差分、成功条件、停止条件、禁止する代替実装、判断履歴が欠落していない。
- TODO
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
