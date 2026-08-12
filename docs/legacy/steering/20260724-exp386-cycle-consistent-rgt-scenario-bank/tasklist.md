# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- Stage 0 FAILのためfull run、Stage 1/2、exp387、inference、current-test、submissionは実行不可。

## 完了

- 2026-07-24: ユーザー指示によりバックログ化、exp386 scaffold、steering作成を承認。
- route、独立family、RGT定義、graph/node/edge、scenario 8--32、diversity、固定gateを確定。
- target GRをexp386で使用せずexp387へ予約する責務分離を確定。
- 再現性、leakage、resource、truth-late停止条件を設計。
- template train/inference Notebookは未実装scaffoldのまま保持。
- 2026-07-24: ユーザーの `exp386を実装してください` を実装承認として記録。
- Jupytext percent形式のcompact self-contained train候補とfail-closed inference候補を
  正規Notebookとは別名で実装した。
- ordered formation interval RGT、64/32 ft node、近傍24 unique well edge、
  LOO q05--q95 stretch、Huber cycle solve、fundamental cycle basis、
  deterministic k-shortest simple route、8--32 scenario、0.5 ft diversityを実装した。
- role-read ledger、target-free logical SHA freeze、512-row rolling-origin、
  truth-late H512 oracle、resource projection、全artifact/SHA manifestを実装した。
- 専用test 11件、py_compile、Ruff、Jupytext train/inference変換とround-tripをPASSした。
- 正規train/inference scaffoldは上書きせず、Kaggle package/push/runは行っていない。
- 2026-07-24: ユーザーの `実行してください` を正規Notebook採用、Kaggle package/push、
  16-well Stage 0 preflight、PASS後full runの承認として記録した。
- push前の実行量を1 variant / 5 graph solves / 773 target-well solves /
  model・HMM・PF・Beam・booster各0 / parent control再実行0に固定した。
- 正規train/fail-closed inferenceを採用し、private CPU / internet off / run-on-pushの
  train package、parent OOF存在、埋め込みZIP/config SHAを監査してPASSした。
- canonical version 1をpushし、pull-back metadataと`RUNNING` statusを確認した。
- canonical version 1（id_no `128478384`）の`COMPLETE`とlogsを確認した。
- 16 wells / 5 foldsでRGT coverage、leakage、runtime、RSSはPASSしたが、
  graph query / scenario-bank / finite-path coverageが0、cycle residual p95が
  `2.363303 > 0.10`でStage 0 FAIL_CLOSEとなった。
- full run、Stage 1/2、edge/stretch/scenario/diversity救済、inference、submissionを
  実行せずbranchを閉じた。
