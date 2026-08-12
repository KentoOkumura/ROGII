# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- stochastic PF は per-well/per-seed stable seed policy とし、global RNG / thread scheduling に依存しない設計にした。
- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/` の実装を追加した。
- Jupytext 起点の train / inference notebook を作成した。
- py_compile、ruff、jupytext round-trip、strict validate を通した。
- Kaggle package を prepare し、metadata と bootstrap 内 config の整合を確認した。
- Kaggle train v1 を push / completion 確認した。
- Kaggle output を取得し、row candidate content SHA、metrics SHA、summary SHA を記録した。
- Kaggle train v2 all-well rerun を push / completion 確認した。
- Kaggle v2 output を取得し、row candidate content SHA、metrics SHA、summary SHA を記録した。
- v1/v2 prefix-holdout audit は意図した full replay cache rebuild ではないため superseded と判断した。
- exp072-style full replay train cache rebuild 実装へ差し替えた。
- Kaggle train v3 を push / completion 確認した。
- Kaggle v3 summary/schema を取得した。
- Kaggle CLI の large output OOM を回避するため、Kaggle output URL から train feature gzip を chunk streaming で取得した。
- v3 train feature cache を local artifacts に保存し、raw gzip SHA、decompressed SHA、gzip integrity、row count を記録した。
