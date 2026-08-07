# exp323 結果

## 状態

旧exp307/308/309 lineage不成立により閉鎖済み。未実装・未実行で、CV、LB、prediction、submissionはない。

## 固定した評価

- Stage 0: rate-change RMSEを定数prior比5%以上改善、累積経路RMSEを0.05 ft以上改善、4/5 folds、1000+・hidden-like 2面非悪化、worst well `<=+0.25 ft`。
- Stage 1: 親exact HMM比0.05 ft以上、4/5 folds、p95/worst/tail/fixed blend guardを全要求する。

## 次

本実験は再開しない。exp338 PASS時だけ、新番号でexp338を親にした新exp323相当を設計する。
