# exp319 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。exp311/313待ち。実装・実行なし。
- Route: `pf_beam`
- 実行契約: scientific 1 + permutation control 1 / 5 folds / 0 model / 0 booster / 0 decoder。

## 設計契約

- Type Well content descriptor 10列、outer-train robust scale、diagonal Mahalanobis、k=3。
- exp kernel、temperature 1、nearest-distance outer-train p90 cutoff、global fallback。
- descriptor/scaler/neighbor/prior/readoutのschema/content SHAを保存する。
- horizontal suffix、truth/error、well IDをsimilarityへ使わない。

## 次

先行gate PASSと実装承認後にgroup-out readoutを実装する。
