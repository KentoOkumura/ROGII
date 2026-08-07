# タスクリスト

## 未着手（別途承認が必要）

- 既存バックログの0-booster `exp287_fold_safe_formation_tail_attribution_readout` は再開条件を満たしたが、着手前にユーザー確認を得る。

## 進行中

- なし。

## ブロック中

- exp334のinference、submission、control再学習、追加rerunは未承認かつpromotion guard不通過のため閉じている。

## 完了

- [x] 仮説と単一変更をouter-trainのwell均等化sample weightに固定した。
- [x] valid/early stopping/OOF評価を非加重のまま固定した。
- [x] exp287の421特徴・5 folds・3 configs・seed・GPU設定を固定した。
- [x] 保存済みexp287/exp264 controlを使い、control再学習0に固定した。
- [x] 実行量を1 variant × 3 configs × 5 folds = 15 GPU boostersと記録した。
- [x] pooled、fold、scope、by-well p95、worst-well、悪化well数のAND gateを確定した。
- [x] compact self-contained trainとfail-closed inference候補、専用testを実装した。
- [x] canonical train notebookを採用し、Jupytext/py_compile/ruff/test/strict validationを通した。
- [x] Kaggle T4 preflight version 1を0-boosterで完了し、全contractとSHAを確認した。
- [x] Kaggle T4 train version 2で15/15 boostersを完了した。control再学習は0。
- [x] OOF 3,783,989行、773 wells、5 folds、15 models、主要成果物SHAを監査した。
- [x] CV `8.09349752413077`、exp287比`-0.04321069622868201 ft`、5/5 folds改善を確認した。
- [x] pooled/fold/scope PASS、by-well p95/worst/count FAILを確認し、固定AND gateをFAILと判定した。
- [x] exp334を非昇格として閉じ、inference/submission/追加trainをfalseにした。
- [x] `result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`へ結果を反映した。

## 最終結果

- Global側: pooled、5/5 folds、全scope PASS。
- Tail側: `+1 ft`件数のみPASS。by-well p95、worst-well、`+3/+5 ft`件数はFAIL。
- 総合: `train_complete_guard_failed_closed_no_inference`。推論・提出なし。

## 次のアクション

- exp334は追加実行せずcloseを維持する。
- 0-booster formation tail attribution readoutへ進む場合は、別途ユーザー確認後に設計・実行する。
