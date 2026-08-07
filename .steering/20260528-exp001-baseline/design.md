# 設計

## アプローチ

公開 notebook / discussion 調査では、ROGII は row-level regression ではなく well ごとの prefix-conditioned forecasting として扱うべきと整理されている。exp001 では学習器を入れず、既知 prefix の最後の `TVT_input` を tail 全体に保持する `last_anchor` を primary にする。

参考値として、最後の 200 rows から robust recent slope を推定する `recent_linear` も CV に出す。ただし調査では raw extrapolation より drift/residual target が重要とされているため、submission は `last_anchor` のままにする。

## 実験範囲

- 対象実験: `exp001_baseline`
- 親実験: なし
- 変更する変数: 予測戦略 (`last_anchor`, reference `recent_linear`)
- 固定する変数: データ、5-fold GroupKFold、RMSE、seed 42、評価対象 row mask

## リスク

- リークリスク: prefix の最後だけを使うため低い。tail 内 true TVT と train-only formation columns は使わない。
- CV/LB 不一致リスク: visible test は train 由来 3 wells なので、Public LB は過信しない。
- ランタイム/メモリリスク: 全 773 wells を逐次読み込み、OOF 予測は保存しないため低い。
