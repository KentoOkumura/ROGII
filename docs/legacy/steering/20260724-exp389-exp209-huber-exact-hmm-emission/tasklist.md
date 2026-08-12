# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- inference、submissionは未承認。

## 完了

- 2026-07-24: exp357がexp281 residual-offset HMMを親にした誤スコープ実験で、
  本来の依頼がexp209 absolute-TVT exact HMMのGaussian emission単独置換だったと確認した。
- 2026-07-24: exp388まで使用済みであることを確認し、exp389を採番した。
- 2026-07-24: `docs/06_reproducibility.md`を確認した。
- 2026-07-24: exp374のexp209固定HMM契約を構成参照にし、Huber
  `delta=1.345`だけを独立変更する設計を固定した。
- 2026-07-24: 1 variant / 773 HMM runs / parent control rerun 0、
  technical/scientific/tail/fixed-blend gate、no-rescueを固定した。
- 2026-07-24: design-only steering、experiment scaffold、configを作成した。
- 2026-07-24: `KAGGLE_DIRECTION.md`の判断メモと未着手バックログ、
  `experiment_summary.md`へ登録した。
- 2026-07-24: YAML/JSON parse、strict experiment validation、
  template/strict project validation、実験文書reviewをPASSした。
- 2026-07-24: ユーザーの実装承認を受け、exp209 compact self-contained構成を
  基準にfixed Huber emission 1件を実装した。
- 2026-07-24: synthetic Huber formula/boundary、同一emissionでのexp209 exact
  kernel parity、truth-late join、saved-control SHA、全promotion gate、
  run未承認guard、fail-closed inferenceの専用テスト9件をPASSした。
- 2026-07-24: compact self-contained train/inference候補をJupytext変換し、
  round-trip、py_compile、RuffをPASSした。正規Notebookは上書きしていない。
- 2026-07-24: push前に1 variant / 773 HMM / model・fold・booster・control
  rerun 0と、metadata / loose config / bootstrap config / kernel sourcesを
  再確認し、正規train Notebookを採用してKaggle private CPU version 1を開始した。
- 2026-07-24: version 1（id_no `128466838`）を`19,417.246 sec`で完了した。
  technical gate、overall、5/5 folds、全required scope、fixed 50:50はPASS。
- 2026-07-25: by-well p95 `+0.002234 ft`とworst well `00bbac68`
  `+1.750248 ft`で安全gateをFAILした結果、およびcandidate raw/content、
  contract、manifest、metrics、gate SHAを記録した。
- 2026-07-25: decision `huber_exp209_failed_close_without_rescue`として、
  救済、再実行、inference、submissionなしでterminal closeした。
