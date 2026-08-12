# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 13候補 selector 方針の選択。
- 再現性設計を `design.md` に記入。
- exp371 experiment scaffold と fixed13 candidate / feature contractを作成。
- exp333 target-free OOF loader と exp263 fold bundle extensionを実装。
- Stage A + Stage C Jupytext train sourceとcompact candidate notebookを作成。
- inferenceをStage C gate前にfail-closed化。
- exp333 OOF key/fold/SHA、13 candidate、77 compact featureの回帰testを追加。
- py_compile / Ruff / dedicated pytest / Jupytext testを通過。
- 1 variant / 2 objectives / outer 5 × inner 4 / 40 CPU boosters /
  control retraining 0 を `SESSION_NOTES.md` に記録。
- 正規notebook採用とKaggle CPU train v1の明示承認を得た。
- compact train/inference notebookを正規notebookへ採用した。
- 正規採用後のJupytext / py_compile / Ruff / 43 tests / strict validationを通過。
- Kaggle CPU packageのmetadata、bootstrap、scope、SHAを監査した。
- 承認scopeどおりKaggle CPU train v1 version 1をpushした。
- pull-back metadataでprivate / CPU / internet off / source一致を確認した。
- 1回分の実行承認を消費し、ローカル`run_approved=false`へ戻した。
- version 1はpath resolverでfit前停止（0 boosters）。
- absolute patternをroot globから除外し、回帰test追加後に44 testsを通過。
- version 2は入力SHA/allowlist通過後、fold parity guardでfit前停止（0 boosters）。
- exp263 / exp333のfold別row countsと代表wellのfold不一致を確認した。
- ユーザーがglobal key join + exp263 selector-fold repartition方針を承認した。
- version 3でglobal key join / selector-fold repartition、Stage A、40 CPU
  selector boosters、technical / score / leakage auditを完了した。
- fixed13は親fixed12よりpooled `-0.232535 ft`、4/5 folds改善した。
- by-well p95 `+0.861529 ft`、worst well `+10.757997 ft`でscientific gateをFAILした。
- 必要な小容量評価CSV/JSONだけを選択取得し、SHAを確認した。
  - `exp371_fixed13_vs_fixed12_scope_metrics.csv`
  - `exp371_fixed13_candidate_usage.csv`
  - `exp371_fixed13_vs_fixed12_by_well.csv`
  - `exp371_scientific_gate.json`
  - `exp371_summary.json`
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`を更新し、fixed13 branchを閉じた。
- ユーザーがpooled平均改善を根拠に、元のStage C safety gate FAILを保持したまま
  既設計のStage D 15 GPU boosterへ進むことを明示承認した。
- Stage C契約ファイル4件を選択取得し、file/logical SHAを固定した。
- Stage D Jupytext source、config、共通fixed13 compact runner、回帰testを実装し、
  canonical T4 packageの静的監査を完了した。
- 初回Stage D pushは週45時間GPU quota到達でversion作成前に拒否された。
- 2026-07-25にユーザーがquota回復を確認し、同じ15-booster契約の実行を再指示した。
- 前回quota拒否で残ったversion/session/outputのない空shellを対象確認後に削除し、
  同じcanonical slugを再作成した。
- Stage D version 1をT4へpushし、id_no `128524177`、private / T4 /
  internet off / 3 kernel sources、pull-back source一致、RUNNINGを確認した。
- push後にlocal run flagを無効化し、1回分の実行承認を消費した。
- canonical Stage D kernel version 1の`COMPLETE`と15/15 GPU booster完走を確認した。
- output archive全体を取得せず、logsと小容量評価artifactだけを選択取得した。
- fixed13 compact add-onlyは親fixed12比`-0.090815 ft`、3/5 folds、
  near / 1000+ / hidden-like 2面を改善した。
- by-well p95 `+1.179312 ft`、worst well `+4.637599 ft`でStage D gateをFAILした。
- Stage C FAILを再分類せず、inference / submissionなしでbranchを閉じた。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`README.md`、
  `experiment_summary.md`、`KAGGLE_DIRECTION.md`をStage D結果で更新した。
