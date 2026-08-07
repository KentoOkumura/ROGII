# タスクリスト

## TODO

- shard完了後にaggregateを1回だけ実行する。
- 実行後にshard feature content SHA、aggregate prediction SHA、kernel version、診断結果を記録する。

## 進行中

- Kaggle CPU shard 0/1 version 1を監視する（kernel id no `127592526` / `127592528`）。

## ブロック中

- なし

## 完了

- backlog、exp209/242/243/266、`docs/06_reproducibility.md`を確認した。
- `exp268_multi_scale_initial_rate_candidates`のsteeringとexperiment scaffoldを作成した。
- HMM 4候補、2 target-free well shard、saved tail30 control、診断専用oracleという設計を固定した。
- compact self-contained generator 2本、aggregate train、disabled inferenceを実装した。
- exp268固有contract test 6件を実装し、PASSを確認した。
- Jupytext round-trip、py_compile、Ruff F821、strict experiment validationを通した。
- private CPU Kaggle package 4本をprepareし、metadataとbootstrap内config/sourceのbytes一致を確認した。
- ユーザーの実行指示後、shard 0/1をcanonical IDへversion 1としてpushした。
- `experiment_summary.md`と`KAGGLE_DIRECTION.md`を実装済み・Kaggle CPU待ちへ更新した。
