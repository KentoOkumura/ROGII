# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp163 の steering docs を作成し、比較対象を exp092 ではなく exp148 に固定した。
- exp148 ML route anchor に、exp099 row context と exp065 native typewell overlap cluster 由来の fold-safe neighbor prior feature を add-only で追加する設計にした。
- direct selector、soft average、blend、postprocess replacement を入れない feature-only 実装にした。
- exp148 historical control の再学習を無効化し、baseline は保存済み exp148 CV / Public LB を参照する方針にした。
- CPU 前提で `lgb0` / `lgb1` / `lgb2` の train notebook を分割した。
- `py_compile`、`ruff --select F821`、Jupytext 変換、`make validate-exp` のローカル検証を通した。
- Kaggle train を `lgb0` / `lgb1` / `lgb2` に分けて実行し、3 kernels の COMPLETE を確認した。
- 3 split の OOF と feature importance を確認し、exp148 historical baseline と比較した。
- 3-model `lgb_mean` は 8.519739843 で、exp148 historical `lgb_mean` 8.501281182 から +0.018458661 悪化した。
- train-side rejected と判断し、raw test 側 typewell prior parity と inference notebook は実装しないことにした。
- Kaggle output を取得し、prediction SHA proxy と model manifest 生成を記録した。
