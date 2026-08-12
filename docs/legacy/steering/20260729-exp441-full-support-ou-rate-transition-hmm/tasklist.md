# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage 1、rerun、inference、submission。Stage 0 gate FAILにより閉鎖。
- OU parameter、support、emission、grid、gateのsame-OOF救済。

## 次のアクション

- exp441はterminal closeとして保持する。
- 原因確認が必要なら保存済みdiagnosticだけを使う別実験を事前設計する。

## 完了

- 2026-07-29: exp441へ採番し、親、route、single-factor差分を固定した。
- 2026-07-29: exact OU式、全bin積分、境界mass semanticsを固定した。
- 2026-07-29: fixed32 / Stage 1 gate、実行量、truth-late、SHA契約を固定した。
- 2026-07-29: backlog、steering、実験scaffoldとdesign-only記録を作成した。
- 2026-07-29: compact self-contained train/inference候補を実装した。
- 2026-07-29: exact OU bin-integral kernel、position parity、dense
  brute-force、truth-late、SHA contractを実装した。
- 2026-07-29: 専用pytest、Jupytext、構文、Ruff、strict validationをPASSした。
- 2026-07-30: ユーザー承認後に正規train Notebookを採用し、private CPU
  packageのbootstrap/input SHAと実行量を検証した。
- 2026-07-30: Kaggle Stage 0 v1で32 wells / 156,088 rowsを完走した。
- 2026-07-30: technical 16/17、mechanism 2/7で`stage0_fail_closed`とした。
