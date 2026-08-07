# SESSION_NOTES

## 2026-07-02

- `bimodal_posterior_pfbeam_candidate_audit` backlog を `exp171_bimodal_posterior_pfbeam_candidate_audit` として実験化した。
- Route: `pf_beam`
- GPU 学習: なし
- active variant 数: 0
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし

### 実装内容

- raw train well の known prefix から deterministic slope prior を作り、typewell GR shift-scan surface を構成する。
- top1 mode と、6-30ft 離れた top2 local minimum を抽出する。
- 固定温度 `[2, 4, 6, 8, 12, 16]` から posterior mean 候補を作る。
- hard commit、midpoint、posterior mean、fixed exp072 PF/Beam candidates を同じ sampled rows で比較する。
- direct replacement、selector、ML feature 化、inference、submit は未実施。

### 再現性メモ

- 新規乱数は使わない。sample rows は deterministic linspace。
- posterior temperature は config 固定で、same-OOF truth から選ばない。
- upstream exp072 cache は stochastic component として扱い、実行時に SHA を summary へ記録する。
- gzip 出力は decompressed content SHA を記録する。
- train-side diagnostic のため deterministic submission anchor ではない。

### 次アクション

1. Kaggle train notebook を準備し、CPU で実行する。
2. Kaggle train 結果から commit / posterior / likPF の bucket 別差分を記録する。

### 検証

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb exp171_bimodal_posterior_pfbeam_candidate_audit_train.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb exp171_bimodal_posterior_pfbeam_candidate_audit_inference.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test exp171_bimodal_posterior_pfbeam_candidate_audit_train.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test exp171_bimodal_posterior_pfbeam_candidate_audit_inference.py`
- `.venv/bin/python -m py_compile ...` 通過
- `.venv/bin/ruff check experiments/exp171_bimodal_posterior_pfbeam_candidate_audit` 通過
- `make validate-exp EXP=exp171_bimodal_posterior_pfbeam_candidate_audit` 通過
- `make update-summary` 実行済み

### Kaggle package / train v1

```bash
make prepare-kaggle-notebooks EXP=exp171_bimodal_posterior_pfbeam_candidate_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp171-bimodal-posterior-pfbeam-train --title 'exp171 bimodal posterior pfbeam train' --run-on-push --strict"
make push-kaggle-train EXP=exp171_bimodal_posterior_pfbeam_candidate_audit
```

- 生成先: `experiments/exp171_bimodal_posterior_pfbeam_candidate_audit/kaggle/train`
- kernel id: `kentookumura/exp171-bimodal-posterior-pfbeam-train`
- URL: https://www.kaggle.com/code/kentookumura/exp171-bimodal-posterior-pfbeam-train
- GPU: false
- internet: false
- kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- result: Kernel version 1 pushed successfully.
- `kaggle kernels pull kentookumura/exp171-bimodal-posterior-pfbeam-train -p /tmp/kaggle-pull/exp171-bimodal-posterior-pfbeam-train -m` 成功。
- `kaggle kernels status kentookumura/exp171-bimodal-posterior-pfbeam-train`: `KernelWorkerStatus.RUNNING`
- ユーザー指示により監視は停止。完了連絡後に logs / result / metrics を取得して記録する。

### Kaggle train v1 結果

- status: COMPLETE
- kernel: `kentookumura/exp171-bimodal-posterior-pfbeam-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp171-bimodal-posterior-pfbeam-train
- runtime: 889.676942 sec
- rows: 1,187,328 row-context rows
- wells: 773
- full output: `/tmp/kaggle-output/exp171_bimodal_posterior_pfbeam_candidate_audit/train`
- copied small aggregate outputs to `artifacts/`: summary, candidate metrics, bucket metrics, gain-vs-commit
- `row_context.csv.gz` は 290MB のため実験配下へ常設しない。

主要結果:

- `likpf_mean` all/hidden-tail RMSE 11.471434 / MAE 6.989252 / within10 0.775439。
- best posterior all は `rolling_median_11__bimodal_posterior/posterior_mean_t16` RMSE 76.698097 / MAE 39.759649 / within10 0.224286。
- hidden_tail best posterior は `savgol_31_p2__bimodal_posterior/posterior_mean_t16` RMSE 102.301054 / MAE 50.511337 / within10 0.207900。
- hidden_tail で posterior / midpoint は hard commit より mean abs-error を最大 +1.328095ft 改善したが、`likpf_mean` には大きく届かない。
- prefix_backtest best は `rolling_median_11/midpoint` RMSE 35.809334。posterior は midpoint を超えない。

判断:

- `bimodal_posterior_pfbeam_candidate_audit` は完了/不採用。
- posterior candidate の direct replacement、PF/Beam likelihood 変更、inference port、submit はしない。
- `p`、entropy、mode separation、top2 gap も現状の実装では exp148 add-only feature へ進める根拠が弱い。

### 生成物 SHA

- `row_context.csv.gz` raw SHA256: `1f210ec405be93cec466e0094480f19d149c0582d056651abb34d17f5b80b14a`
- `row_context.csv.gz` decompressed SHA256: `8899ead616b05182788f77581097f606f1b15984ab036dc8bab8a75f1b70ebbd`
- `candidate_metrics.csv` SHA256: `5d67c63c0c61d8bc635c1a7de84752dd568c910e590b7fb9b4d3ebbd694a74eb`
- `bucket_metrics.csv` SHA256: `43ce84f003b655285ca468857c38105bab5bfbf5df3111a60867501ee0223cdc`
- `gain_vs_commit.csv` SHA256: `7c8c7bc9c5eae2597bea18e3a13975ccfe084b9514f47485b4fab9e7ff4c3c0b`
