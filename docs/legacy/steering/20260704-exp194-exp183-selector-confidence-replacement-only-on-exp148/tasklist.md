# タスクリスト

## TODO

- Kaggle train を実行する場合は、push 前に 15 boosters / control 再学習なしを再確認する。
- train 完了後、CV、feature count、feature schema、model manifest、prediction SHA、Kaggle kernel version を記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- exp194 実験ディレクトリを作成した。
- exp188 の exp183 selector confidence feature builder を exp194 用に移植した。
- active variant を replacement-only feature groups に変更した。
- 再現性設計を `design.md` に記入した。
- GPU train push 前の booster 数と control 再学習なしを `SESSION_NOTES.md` に記録した。
- Jupytext train / inference 変換と `--test` を通した。
- `py_compile` と `ruff --select F821` を通した。
- `make validate-exp EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148` を通した。
- Kaggle train package を prepare し、metadata / bootstrap config を確認した。
