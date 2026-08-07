# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- inference / submissionは実験範囲外。

## 完了

- 2026-08-01: 最新番号exp497の次としてexp498を採番した。
- 2026-08-01: steeringと標準実験ディレクトリを作成した。
- 2026-08-01: Routeを`pf_beam`、親をexp490、saved-full-OOF diagnosticに固定した。
- 2026-08-01: exp490 merge v1の5入力とscientific contract SHAを固定した。
- 2026-08-01: truth-late順序、7物理量の絶対bucket、単一primary regimeを固定した。
- 2026-08-01: 6項目のphysics-regime all-AND gateとterminal条件を固定した。
- 2026-08-01: planned実行量をreadout 1、HMM / model / prediction / GPU各0とした。
- 2026-08-01: 実装、package、run、inference、submissionを無効のまま維持した。
- 2026-08-01: strict experiment validationをPASSし、実験文書reviewでcore evidenceの
  存在を確認した。train / inference notebookはいずれもcode cell 0のplaceholder。
- 2026-08-01: 追加依頼を実装承認として記録し、Jupytext percent形式のcompact
  self-contained diagnostic train sourceを実装した。
- 2026-08-01: exp490 merge v1の5生成物に加え、4 shard decoder manifest自体のSHAと
  773 wellのraw horizontal / typewell SHA結合をfail-closedにした。
- 2026-08-01: chunked prediction集約、K16集約、visible-prefix GR sigma / information
  ratio、固定7 bucket、単一primary regime、feature contract freezeを実装した。
- 2026-08-01: freeze後だけfold / by-well / episodeを読むtruth-late ledger、6項目
  all-AND、secondary descriptive table、planned 6生成物の保存を実装した。
- 2026-08-01: 合成fixtureの契約test 13件、py_compile、Ruff、Jupytext round-tripを
  PASSした。canonical train notebookは21 cells（Markdown 11 / code 10）、output 0。
- 2026-08-01: exp490 compact train 9章 / 2,399行に対し、exp498 trainはreadoutに必要な
  役割を欠かさない9章 / 1,457行であることを確認した。
- 2026-08-01: 固定入力のread-only SHA解決でmerge 5生成物、4 decoder manifest、
  raw SHA 773 wells、scientific contract一致を確認した。
- 2026-08-01: `make validate-exp` strict、`make validate-template`、実験文書reviewの
  core evidenceをPASSした。Kaggle package / runは行っていない。
- 2026-08-01: ユーザーの「実行してください」でreadout 1 / well aggregation 773 /
  fold readout 5のKaggle private CPU実行を承認した。HMM / prediction / model / booster /
  PF / Beam / GPUは0、親control再実行も0である。
- 2026-08-01: version 1のtruth-late join suffix不整合を修正し、14 testsをPASSした。
- 2026-08-01: Kaggle private CPU version 2（id_no `129328553`）を76.685秒で完了し、
  6生成物のSHA、773 wells、5 folds、technical checks全PASSを確認した。
- 2026-08-01: primary regimeは0 wells、physics checks 6/6 FAIL。固定terminal decision
  `terminate_mean_reversion_tail_regime_cause_tracking`を適用し、backlogから削除した。
