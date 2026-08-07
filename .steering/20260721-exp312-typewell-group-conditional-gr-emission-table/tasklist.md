# タスクリスト

## 先行条件上書き

- [x] exp311全gate PASS条件をユーザー判断で上書きした。
- [x] exp311 group/fold/summary SHAと固定gate失敗を入力契約へ残した。
- [x] exp293 deployable12を固定評価bankとしてユーザー確認した。

## 実装

- [x] fold-safe emission table、fallback、rank readoutを実装する。
- [x] group shuffleとmatched TVT shift controlを実装する。
- [x] compact self-contained train/inferenceと専用テストを作成する。
- [x] Jupytext、構文、ruff、16 tests、strict experiment/template validationを通す。
- [x] static validation後にユーザーの承認を得た。
- [x] 正規train notebookを採用し、Kaggle private CPU version 1を完了した。

## 完了

- [x] 条件軸、Student-t、df、shrinkage、fallback、gateを固定した。
- [x] decode・ML学習・inference・submissionを禁止した。
- [x] 固定gateを判定し、FAIL時の救済なしでbranchを閉じた。
