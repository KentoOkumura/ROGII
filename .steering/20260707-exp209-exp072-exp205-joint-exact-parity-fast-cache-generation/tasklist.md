# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` scaffold を作成。
- exp072 full replay helper と public replay implementation を exp209 に同梱。
- exp072 generation result DataFrame を optional に返す `return_frame` path を実装。
- exp205 direct comparison に in-memory exp072 baseline frame 入力を追加。
- HMM generator に optional well-level outer parallelism を追加。
- joint orchestrator、train notebook source、inference no-op notebook source を実装。
- 再現性設計を `design.md` に記入。
- Jupytext 変換 / `--test`、`ruff --select F821`、`validate_experiment.py` を通過。
- Kaggle train package を prepare し、metadata の CPU / internet off / kernel sources を確認。
- Kaggle train v3 を完了し、exp072/exp205 reference parity、comparison metric、wall time を記録。
- exp072 full cache parity FAIL、HMM decompressed SHA parity PASS、metric RMSE は近似一致として ACCEPT、runtime target FAIL と判定。
- 当初は strict serial parity 未達のため `feature_cache.hmm.outer_workers=2` follow-up を保留したが、ユーザー確認により RMSE 近似一致を許容し、v5 以降の runtime 探索へ進めた。
- v4 用に `numba.get_num_threads()` の effective thread count 記録を実装し、`runtime.numba_num_threads=null` に変更。
- v4 Kaggle train package を prepare し、kernel version 4 を push。
- v4 Kaggle train を完了し、effective Numba threads `4`、HMM elapsed `19,749.099 sec`、HMM decompressed SHA PASS、best RMSE `10.269696146642758` を記録。
- v4 は v3 より `1,400.882 sec` 速いが、`numba_num_threads=null` でも実効 thread count は `4` のままで、all-core 化による HMM 並列度増加は確認できないと判定。
- v5 用に `feature_cache.hmm.outer_workers=2`、`runtime.numba_num_threads=2` へ変更し、static checks / Jupytext / validate を通過。
- Kaggle train kernel version 5 を push し、`KernelWorkerStatus.RUNNING` を確認。
- Kaggle train v5 を完了し、effective Numba threads `2`、HMM elapsed `11,285.868 sec`、HMM decompressed SHA PASS、best RMSE `10.269696146642758` を記録。
- v5 は total elapsed `20,203.290 sec` で v4 から `12,580.970 sec` 短縮し、6h 未満 runtime target を達成した。
- v6 用に `feature_cache.hmm.outer_workers=4`、`runtime.numba_num_threads=1` へ変更し、static checks / Jupytext / validate を通過。
- Kaggle train kernel version 6 を push し、`KernelWorkerStatus.RUNNING` を確認。
- Kaggle train v6 を完了し、effective Numba threads `1`、HMM elapsed `14,627.100 sec`、HMM decompressed SHA PASS、best RMSE `10.269696146642758` を記録。
- v6 は total elapsed `28,768.406 sec` で v5 より `8,565.116 sec` 遅く、HMM elapsed も v5 より `3,341.232 sec` 遅いため不採用と判定。
- 既定設定を v5 の `feature_cache.hmm.outer_workers=2`、`runtime.numba_num_threads=2` に戻し、exp209 の best runtime として記録。
