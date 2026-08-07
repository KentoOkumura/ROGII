# exp341 セッションノート

## 2026-07-22 設計確定

- 目的: 補間値を実観測同様に扱う問題を、校正済みsoft varianceだけで分離検証する。
- 依存: exp339 Stage 0全gate通過とtable SHA凍結が必須。
- 実行規模予約: variant 1、fold 5、773 well HMM run、booster 0、control再実行なし。
- 比較: exp281保存済み結果を直接controlとし、exp226もpromotion基準として使う。
- long-tail guard: 4/5 fold、missing率stress、p95、worst wellの悪化上限を固定した。
- 禁止: Student-t/Huber、ACF tempering、sigma再推定、欠損観測の完全無効化。

## 未実施

依存未達のためblocked。コード実装、notebook実行、Kaggle push、成果物生成は行っていない。

