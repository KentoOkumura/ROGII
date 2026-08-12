# タスクリスト

## 実装

- [x] `make new-steering EXP=exp125_confidence_gate_continuity_rawtest_parity` を実行した。
- [x] `make new-exp EXP=exp125_confidence_gate_continuity_rawtest_parity` を実行した。
- [x] steering requirements / design / tasklist を記入した。
- [x] `docs/06_reproducibility.md` を確認した。
- [x] `config.yaml` に lineage、runtime、入力、比較対象 variant、再現性方針を記入した。
- [x] `confidence_gate_continuity_rawtest_parity.py` を追加した。
- [x] train notebook を posthoc audit 実行用に更新した。
- [x] inference notebook を no-submission summary 用に更新した。
- [x] `.venv/bin/python -m py_compile experiments/exp125_confidence_gate_continuity_rawtest_parity/confidence_gate_continuity_rawtest_parity.py` を通す。
- [x] `make validate-exp EXP=exp125_confidence_gate_continuity_rawtest_parity` を通す。
- [x] Kaggle train package を strict で生成する。
- [x] Kaggle inference package を strict で生成する。

## 実行

- [x] Kaggle train を push する。
- [x] output を取得する。
- [x] `SESSION_NOTES.md` / `result.md` / `metrics.json` を実行結果で更新する。
- [x] `KAGGLE_DIRECTION.md` から実装済み backlog を削除し、結果に応じた次候補を追加する。

## 注意

- exp125 は診断専用で、submission は作らない。
- exp112 OOF coverage に合わせた shared surface が採用判断の主対象。
- dense/high-drift gate は optional input がない限り比較対象には入らない。
- raw-test parity checklist は hidden regeneration の代替ではない。
