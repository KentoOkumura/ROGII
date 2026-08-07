# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 依頼範囲、route、親、変更点、禁止事項をrequirements/designへ固定した。
- `docs/06_reproducibility.md`を確認し、PF replayは`n_jobs=1`、stable request id、SHA記録を必須化した。
- compact self-contained Jupytext train / inference notebookとconfigを実装した。
- 単一backlog項目として `KAGGLE_DIRECTION.md` に登録した。
- train / inference notebookの`py_compile`、Ruff `F821`、Jupytext round-trip testを通した。
- strict experiment / repository template validationを通した。
- canonical Kaggle packageをprepareし、metadata、bootstrap config、kernel sourceを確認した。
- Stage 0 v1の96/96 request、error 0と、exp072 cacheの`likpf_mean` schema不一致をKaggle logsで特定した。
- exp072 v2 feature schemaを取得し、`likpf_mean_d`から絶対値を復元する修正とpre-replay schema validationを実装した。
- Stage 0 v2を同じcanonical kernelで完了し、32 wells / 96 requests / technical checks 7/7 passを確認した。
- overall -0.198601、long-tail -0.226789に対し、worst well +1.124800でadoption guardがfailした。
- 固定Stage 1でも同じworst well予測が残ることを確認し、Stage 1 / inference / submissionを不実行とした。
- input/cache/prefix score/controller OOFのSHAと取得した小生成物のSHAを監査した。
- ユーザー判断によりworst-wellを拒否条件から監視指標へ変更し、Stage 1を再開した。
- stable SHA256 well modulo 4のshard notebookとglobal aggregate notebookを実装した。
- 4つのStage 1 CPU shardを完了し、773 wells / 2,319 requests / error 0 / technical passを確認した。
- 4 shardのrow-level OOFをaggregateし、773 wells / 3,783,989 rowsのtechnical checks 9/9 passを確認した。
- overall +0.268755、1000+ +0.307983、hidden-like +0.282873 / +0.267543、改善fold 0/5でglobal adoption guard不通過を確定した。
- worst-wellをmonitor-onlyとしても不採用であり、inference / submissionを禁止してbranchを完了した。
