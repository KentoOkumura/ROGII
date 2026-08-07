# タスクリスト

## TODO

- Kaggle train package 作成前に GPU cost guard を再確認する。
- Kaggle push 前に metadata と bootstrap 内 config の整合を確認する。
- output 取得後に feature content SHA、prediction SHA、model SHA、Kaggle kernel version を記録する。
- OOF 完了後に near-row、longtail、worst-well、feature importance、exp115 hidden-like stress の要否を判断する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `experiments/exp149_normalized_shape_addonly_features_on_exp092/` を exp130 から派生作成した。
- exp130 の diagnostic score / confidence flag を削除し、normalized shape add-only feature generator に置き換えた。
- config で `exp092_full_row_control` を disabled、`normalized_shape_addonly` を active にした。
