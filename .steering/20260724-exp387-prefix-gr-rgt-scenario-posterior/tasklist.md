# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- exp386 Stage 0 FAILにより開始条件が不成立。現設計は再開しない。

## 完了

- 2026-07-24: ユーザー指示によりバックログ化、exp387 scaffold、steering作成を承認。
- route、exp386依存、fixed likelihood、circular control、exact transition/posterior、固定gateを確定。
- exp386 candidate生成とexp387 likelihood/posteriorの責務分離を確定。
- 再現性、leakage、resource、truth-late停止条件を設計。
- template train/inference Notebookは未実装scaffoldのまま保持。
- 2026-07-24: exp386 version 1が空scenario bankとcycle residual gateで
  Stage 0 FAIL_CLOSEとなったため、exp387を未実装・未実行で閉じた。
- 親結果はscenario-bank / finite-path coverage 0、cycle residual p95
  `2.363303 > 0.10`。exp387固有の生成物は実装前閉鎖のため0件。
