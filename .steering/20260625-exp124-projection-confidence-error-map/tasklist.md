# タスクリスト

## 実装

- [x] `make new-steering EXP=exp124_projection_confidence_error_map` を実行した。
- [x] `make new-exp EXP=exp124_projection_confidence_error_map` を実行した。
- [x] steering requirements / design / tasklist を記入した。
- [x] `config.yaml` に lineage、runtime、入力、gate 候補、再現性方針を記入した。
- [x] `projection_confidence_error_map.py` を追加した。
- [x] train notebook を診断生成物作成用に更新した。
- [x] inference notebook を no-submission summary 用に更新した。
- [x] `py_compile` を通す。
- [x] `make validate-exp EXP=exp124_projection_confidence_error_map` を通す。
- [x] Kaggle train package を strict で生成する。
- [x] debug smoke として `max_rows=20000` の direct script 実行を `/tmp/exp124_projection_confidence_smoke` に完走させた。

## 実行

- [x] Kaggle train を push する。
- [x] output を取得する。
- [x] `SESSION_NOTES.md` / `result.md` / `metrics.json` を実行結果で更新する。
- [x] `KAGGLE_DIRECTION.md` から実装済み backlog を削除し、結果に応じた次候補を追加する。

## 注意

- exp124 は診断専用で、submission は作らない。
- gate 条件は target-free で定義し、`target_tvt` は評価にだけ使う。
- global OOF 改善だけで inference port しない。
