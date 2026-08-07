# exp434 タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 対象をexp263 raw-test-ready 12候補に固定した。
- OOF値、candidate ID、pair/fixed formulaを固定した。
- 既存LB 3件の出典と候補同一性gateを固定した。
- 未提出9件を5 pair / 4 primitiveの2 batchに固定した。
- LB結果を見たweight/candidate tuning禁止を固定した。
- 実行量を0 train / 0 model / 0 booster / 通常9 submissionに固定した。
- stable seed、source SHA、prediction/submission SHA、bootstrap方針を固定した。
- backlog、steering、design-only実験scaffoldを作成した。
- exp263 inference v3と4 generator / Stage 0 / Stage 1 source SHAを再監査した。
- K16 / LikPF / fixed既存submissionとのcandidate同一性gateを実装した。
- Jupytext percent形式のcompact self-contained inference候補を別名で作成した。
- 12候補manifest、通常9候補のfail-close選択、float32 formula parityを実装した。
- candidate-version manifest、prediction/submission SHA、fallback row監査を実装した。
- exp434専用test 8件、py_compile、Ruff F821、Jupytext testをPASSした。
- 保存済みexp263 v3 formula bankとの全12候補最大差0、既存fixed / K16
  submissionとの`0.001 ft`同一性gate PASSをread-only確認した。
- compact self-contained実装を正規inference Notebookへ採用した。
- 正規kernel
  `kentookumura/exp434-physics-candidate-lb-audit-infer`のversion 1–9で
  凍結済み5 pair / 4 primitiveを実行し、全件COMPLETEとsubmit-check PASSを確認した。
- K16 existing gateはPASS、LikPF existing gateは最大差`4.7783203125 ft`で
  FAILした。事前登録済みfailure policyに従い、LikPFをversion 10で生成して
  COMPLETE / submit-check PASSを確認した。
- 全10 versionで14,151 rows / 3 wells / fallback 0、parent formula parity
  最大差0.0 ft、candidate bank SHA一致を確認した。
- package / prediction / submission SHAを
  `experiments/exp434_physics_candidate_public_lb_audit/kaggle_run_ledger.json`
  へ記録した。
- version 1–7をcompetition submitし、7件すべてCOMPLETEを確認した。
- version 8–10を凍結順序どおりcompetition submitし、ref
  `55133068 / 55133072 / 55133074`でCOMPLETEとPublic LB
  `12.061 / 15.563 / 9.807`を確認した。
- version 4–7のPublic LB `8.812 / 8.642 / 9.318 / 9.063`とrefを
  `SESSION_NOTES.md`、`result.md`、`metrics.json`、`kaggle_run_ledger.json`、
  `experiment_summary.md`、`submissions/SUBMISSIONS.md`へ記録した。
- 既存2候補を含む採点済み9候補のOOF/LB Spearman `0.750`と順位逆転を
  `physical_model_summary.md`、`KAGGLE_DIRECTION.md`へ記録した。
- 全12候補のPublic LB表、LB - OOF、OOF/LB rankを最終集計し、Spearman
  `0.846154`を記録した。
- `SESSION_NOTES.md`、`result.md`、`README.md`、`metrics.json`、
  `kaggle_run_ledger.json`、`physical_model_summary.md`、提出履歴、全体要約を
  12 / 12採点完了状態へ更新した。
