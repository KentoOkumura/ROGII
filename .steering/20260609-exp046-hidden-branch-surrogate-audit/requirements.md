# 要件

## 依頼

`hidden_branch_surrogate_audit` を実装する。public sample では visible train well 用処理が発火して `changed_rows=0` になるため、見えない test well 用処理の安全性を train 側 train well の途中以降を隠した疑似 test row で代理検証できるようにする。

## 制約

- Route: `pf_beam`
- 提出用 `submission.csv` は生成しない。
- `exp029` の public sel15 PF/Beam train well の途中以降を隠した疑似 test 生成物を入力にする。
- `exp026` anchor は audit split ごとに validation well を除外して再生成する。
- `exp044` の層化 fold は補助的な危険信号チェックに限定し、採用基準や tuning には使わない。
- 既知の見えない test well 用処理失敗 (`exp031`、`exp033`、`exp035`、`exp045`) を同じ代理面で比較できるようにする。

## 受け入れ基準

- `exp046_hidden_branch_surrogate_audit` が作成され、`config.yaml` の route / lineage / leakage policy が明記されている。
- `hidden_branch_surrogate_audit.py` が overall、distance bucket、split、stratified fold、well、diff/range の監査 CSV と summary JSON を保存する。
- train notebook が監査内容、入力、出力を読めるセル構成で実行できる。
- inference notebook は audit-only guard とし、提出ファイルを生成しない。
- `validate_experiment.py`、py_compile、ruff、smoke 実行が通る。
