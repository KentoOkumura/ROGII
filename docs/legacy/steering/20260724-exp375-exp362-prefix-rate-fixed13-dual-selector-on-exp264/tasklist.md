# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- ユーザー判断により別add-one novelty監査を省略し、fixed13 selectorへ直接進む方針を固定した。
- exp362の観測候補をdonor-slopeではなく`prefix_rate_exact_hmm`として定義した。
- fixed12 + 1候補、40 CPU selector boosters、downstreamなしのscopeを固定した。
- 再現性、leakage、fold、raw-test parity、tail safety設計を`design.md`へ記入した。
- steering documentsを作成した。
- design-only experiment scaffoldのstrict validationをPASSした。
- experiment reviewerでcore evidence categoriesが揃っていることを確認した。
- ユーザー指示により実装前で停止する。
- ユーザー指示「exp375を実装してください」によりimplementation-only scopeを再開した。
- exp362 target-free OOF loader、logical/decompressed SHA evidence、global key join、
  selector-fold repartitionを実装した。
- candidate / feature / config契約をfixed12 +
  `prefix_rate_exact_hmm`へ更新した。
- native confidenceの`sigma_tvt`、`source_loglik`、`loglik_per_row`を
  raw-test再生成可能なwell単位定義で実装した。
- Stage A schema refreeze、nested Stage C selector、paired parent readout、
  post-freeze H512 / whole-well novelty診断を実装した。
- compact self-contained trainとfail-closed inference notebook候補を作成した。
- 専用10 tests、関連回帰47 tests PASS（exp373 lifecycle assertion 1件deselect）、
  py_compile、Ruff、Jupytext test、
  strict experiment validationをPASSした。
- 親compact 8章構成との同等性と、exp375 train 540行を確認した。
- ユーザー承認後にcompact版を正規notebookへ採用した。
- 1 variant / 2 objectives / 5 outer / 4 inner / 40 CPU boosters /
  control再学習0を再記録し、metadataとbootstrap内config/SHAを照合した。
- canonical private CPU kernel version 1を実行し、40/40 selector modelsを完了した。
- logsと必要な小型生成物からCV、fold、usage、hidden-like、by-well safety、
  novelty診断、SHAを記録した。
- parent fixed12比pooled・5/5 folds・near・1000+・well tailを悪化させたため、
  decisionを`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`へ固定した。
- downstream TVT、inference、submissionへ進まずbranchを閉じた。
- `result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`を更新した。
