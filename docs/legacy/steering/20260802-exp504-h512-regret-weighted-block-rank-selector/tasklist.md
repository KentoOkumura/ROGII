# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- exp504番号、名前、route、親、比較対象を確定。
- H512 only、fixed12、candidate順、入力SHAを確定。
- 88列のblock集約、pair表現、label、regret weightを確定。
- 1つのpairwise rank model configとBorda/anchor guardを確定。
- outer 5-fold CV、truth-late順序、成功条件、禁止事項を確定。
- 将来実行量を5 CPU models、親再学習/PF/HMM/Beam/GPU 0で確定。
- backlog、実験scaffold、steeringを設計のみの状態で作成。
- Jupytext percent形式compact self-contained train sourceを実装。
- fixed12/88列/H512集約/pair weight/LightGBM/Borda/anchor guard/評価/SHA生成を実装。
- 正規placeholderを上書きせず、別名compact self-contained候補Notebookを生成。
- 9件のcontract test、構文、F821、Jupytext round-trip、strict experiment validationをPASS。
- 正規train notebook採用とKaggle CPU package/runのユーザー承認を得た。
- canonical CPU/private/run-on-push packageを作成し、metadataとbootstrap manifestを確認した。
- 53文字の初回slugはKaggle `SaveKernel 400`で実行前に拒否されたため、科学契約を変えず
  44文字の`exp504-h512-regret-block-rank-selector-train`へid/titleを同時に短縮した。
- Kaggle private CPU version 1（`id_no=129488458`）で5 CPU modelsを完走した。
- technical gate全PASS、pooled RMSE`8.114276980`、anchor比`-0.124054566 ft`を記録した。
- nonworse`3/5`、hidden-like 2面、by-well p95/worstの固定gate FAILを確認した。
- `FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE`として閉じ、
  result/metrics/summary/backlogを更新した。
