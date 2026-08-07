# タスクリスト

## 完了

- [x] steering docs を作成。
- [x] exp235 の config / route / lineage を設定。
- [x] bounded ancestor ring buffer と fixed-lag trace を実装。
- [x] lag 64/128/256 の variant 契約を追加。
- [x] forward fallback、gate diagnostics、interval / worst-well readout を接続。
- [x] train notebook を Jupytext percent 形式で構成。
- [x] static validation を実行。
- [x] full-surface timeout を受け、exact PF semantics を保つ4 deterministic well-shard execution と strict merge utility を追加。

## 次

- [ ] lag64 の4 shard Kaggle CPU train-side audit を実行・strict mergeする。
- [ ] lag128 / lag256 について同じ4 shard audit を実行・strict mergeする。
- [ ] runtime / memory と lag 別 metrics を記録。
- [ ] train guard 通過後に inference / submit を別途判断。
