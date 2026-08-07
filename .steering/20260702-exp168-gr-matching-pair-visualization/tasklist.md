# タスクリスト

## TODO

- 必要に応じて対象 well を増やした追加可視化を実行する。

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `exp168_gr_matching_pair_visualization` を作成した。
- train notebook を Jupytext percent 起点で実装した。
- inference notebook は診断専用 stub に置き換えた。
- `py_compile`、`ruff --select F821,F722,F823`、Jupytext 変換 / `--test`、`validate-exp` を通した。
- Kaggle train package を `kentookumura/exp168-gr-matching-pair-visualization-train` として生成した。
- Kaggle v1 の kernelspec metadata 不足を確認し、`python3` kernelspec を付けて v2 を再 push した。
- Kaggle v2 が COMPLETE し、scored pair CSV、selected pair CSV、HTML index、32 PNG を output として取得した。
- Kaggle v3 が COMPLETE し、query vs matched simple overlay 32 PNG と exp098 lgb1 OOF error 付き good/bad HTML を追加取得した。
- Kaggle v5 が COMPLETE し、top-k local minima、true-near minimum、shift-cost curve、全体 GR context、wrong-depth bucket 別 OOF 集計を追加取得した。
