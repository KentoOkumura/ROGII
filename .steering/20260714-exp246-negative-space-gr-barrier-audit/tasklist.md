# タスクリスト

## TODO

- なし。hard-barrier仮説はguard failedでclosed。

## 進行中

- なし。

## ブロック中

- なし

## 完了

- バックログ追加。
- steering requirements/design作成。
- 再現性設計を`design.md`に記入。
- exp246 experiment templateを作成。
- self-contained Jupytext train/inference notebook、config、settings、記録ファイルを実装。
- barrier/corridor/candidate auditのunit-level synthetic assertionsをtrain notebook内に実装。
- Jupytext conversion/test、py_compile、ruff default rule set、strict experiment validationを完了。
- Kaggle train packageを生成し、metadata / config / kernel sources / CPU / internet off / run-on-pushの整合を確認。
- pre-push cost 1 variant / 0 config / 0 fold / 0 booster / parent retrainingなしを再確認。
- Kaggle CPU v1のgzip SHA失敗を診断し、科学設定を変えずfile buffer closeだけを修正。
- 同じcanonical kernel IDのv2を773 wells / 3,783,989 rowsで完走。
- metrics、5 safety guards、input/output SHA、hidden-like、by-well結果を記録。
- 5 guardsすべてfailを根拠にhard barrier、threshold grid、inference、submissionをclosed。
