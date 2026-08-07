# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。scientific FAILでterminal close済み。

## 完了

- 2026-07-31: exp495まで使用済みを確認し、exp496を採番した。
- 2026-07-31: `docs/06_reproducibility.md`を確認した。
- 2026-07-31: exp264 fixed12、exp392 fixed13、exp486 Stage 1の結果と契約を確認した。
- 2026-07-31: exp486 Absolute版だけを13本目に追加し、Residual、固定HMM blend、
  候補置換を除外する単一変更を確定した。
- 2026-07-31: route、fold/leakage、候補allowlist、native confidence、実行量、
  scientific AND gate、reranking診断、再現性をsteeringへ固定した。
- 2026-07-31: design-only実験ディレクトリと機械可読contractsを作成した。
- 2026-07-31: canonical train / inference Notebookをplaceholderのまま維持し、
  実装、package、Kaggle run、推論、提出を行っていない。
- 2026-07-31: ユーザーの実装承認を受け、compact self-contained train候補、
  fail-closed inference guard、exp486二入力loader / fixed13 cache、scientific gate、
  post-freeze oracle / reranking、feature importance、再現性summaryを実装した。
- 2026-07-31: candidate / feature / output契約の専用test 10件と関連回帰41件、
  Jupytext roundtrip、py_compile、ruff、strict experiment validationをPASSした。
- 2026-07-31: canonical Notebookは上書きせず、Kaggle package / run / inference /
  submissionは0のまま維持した。
- 2026-07-31: ユーザーが固定Stage A/C runを承認した。`1 variant / 2 objectives /
  outer 5 / inner 4 / 40 CPU boosters / parent-control retraining 0 / PF-HMM-Beam /
  GPU 0`を再記録した。
- 2026-07-31: compact trainをcanonical trainへ採用し、canonical slug/title、
  run-on-push、CPU、offline入力をstrict packageで検証した。
- 2026-07-31: 長slugのSaveKernel 400を記録し、同じ科学契約を短縮canonical
  slugへ再packageした。CPU version 1（id_no `129287597`）をpushし、RUNNINGを確認した。
- 2026-07-31: version 1は40/40 modelsを完了し、technical/leakage/score guardをPASS。
  pooled`-0.191174 ft`、4/5 folds、全7 scope改善に対し、by-well p95
  `+1.109360 ft`、worst`+9.361781 ft`で科学gateをFAILした。
- 2026-07-31: `FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`を記録し、小さい
  metrics / manifest / logsだけを選択取得した。same-OOF rescue、downstream TVT、
  current-test、inference、submissionなしでterminal closeした。
