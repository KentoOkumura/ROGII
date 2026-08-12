# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- current-test実装、inference、submission:
  exp405 scientific FAILのため実施禁止。
- exp406実装・実行:
  Stage 0は解禁済みだが別のユーザー承認待ち。

## 完了

- backlog項目を作成した。
- exp405実験scaffoldを作成した。
- steering requirements / design / tasklistを確定した。
- candidate、semi-Markov、emission、geometry再注入、合否、分岐、再現性をconfigへ固定した。
- ユーザーの明示承認後、別名Jupytext percent形式のcompact self-contained
  train候補と専用contract testを実装した。
- exp293 candidate matrix / manifest / block assignmentのSHA guardと、
  truth-free role-read ledgerを実装した。
- fixed16 preflight、exact semi-Markov正規化、geometry floor、negative control、
  runtime projectionを実装した。
- 実行flagを閉じたimplementation-only状態を確認した。
- fixed16実行量を再提示し、ユーザーの明示承認を得た。
- full saved-OOFを無効のまま、fixed16 preflightの実行flagだけを有効化した。
- 正規train Notebookへcompact self-contained候補を採用した。
- Kaggle package、metadata、bootstrap ZIP、埋め込みconfigの整合性を検証した。
- canonical kernel version 1をKaggle CPUで実行し、fixed16 technical gate
  13項目を全PASSした。
- summary実ファイルを取得してSHAを照合し、input / score / posterior /
  predictionのlogical SHA、runtime、peak RSSを記録した。
- fixed16とKaggle execution flagを閉じ、full saved-OOFを別承認待ちに戻した。
- 実行後のlocal Kaggle packageを`run_on_push: false`かつ全実行flag無効で
  再生成し、誤再pushをfail closedにした。
- full saved-OOFの実行量とfixed16 resource projectionを再提示し、
  ユーザーの明示承認を得た。
- full saved-OOFをsame canonical kernel version 2で完走し、
  prediction freeze後だけtruth / hidden-likeをjoinした。
- technical、constrained-oracle、scientific gateを全AND評価し、
  technical PASS / oracle PASS / scientific FAILを記録した。
- summary / gate / fold / scope / by-well / negative control /
  input-role ledger / SHA manifestを取得して実ファイルSHAを照合した。
- 設計済み分岐どおりexp405をcloseし、exp406 Stage 0を解禁した。
