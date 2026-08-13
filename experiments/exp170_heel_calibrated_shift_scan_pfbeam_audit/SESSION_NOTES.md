# SESSION_NOTES

## 2026-07-02

- `heel_calibrated_shift_scan_pfbeam_audit` backlog を `exp170_heel_calibrated_shift_scan_pfbeam_audit` として実験化した。
- Route: `pf_beam`
- GPU 学習: なし
- active variant 数: 0
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし

### 実装内容

- exp167 の shift-scan diagnostic をベースに、known-prefix affine GR calibration を追加した。
- `raw`、`flat_calibrated`、`heel_calibrated` の surface を比較する。
- fixed exp072 PF/Beam candidates は observation-cost readout にだけ使う。
- direct replacement、selector、ML feature 化、inference、submit は未実施。

### 再現性メモ

- 新規乱数は使わない。sample rows は deterministic linspace。
- upstream exp072 cache は stochastic component として扱い、実行時に SHA を summary へ記録する。
- gzip 出力は decompressed content SHA を記録する。
- train-side diagnostic のため deterministic submission anchor ではない。

### 次アクション

### 検証

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb exp170_heel_calibrated_shift_scan_pfbeam_audit_train.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb exp170_heel_calibrated_shift_scan_pfbeam_audit_inference.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test exp170_heel_calibrated_shift_scan_pfbeam_audit_train.py`
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test exp170_heel_calibrated_shift_scan_pfbeam_audit_inference.py`
- `.venv/bin/python -m py_compile ...` 通過
- `.venv/bin/ruff check experiments/exp170_heel_calibrated_shift_scan_pfbeam_audit` 通過
- `make validate-exp EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit` 通過
- `make update-summary` 実行済み

### Kaggle package

```bash
make prepare-kaggle-notebooks EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp170-heel-calibrated-shift-scan-pfbeam-audit-train --title 'exp170 heel calibrated shift scan pfbeam audit train' --run-on-push --strict"
```

- 生成先: `experiments/exp170_heel_calibrated_shift_scan_pfbeam_audit/kaggle/train`
- kernel id: `kentookumura/exp170-heel-calibrated-shift-scan-pfbeam-audit-train`
- GPU: false
- internet: false
- kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

Initial push with the long slug failed:

```bash
make push-kaggle-train EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit
```

- failure: `400 Client Error: Bad Request ... SaveKernel`
- id/title slug matched, exp072 kernel source resolved by `kaggle kernels pull`.
- likely cause: Kaggle-side slug/title constraint with 52-char slug.

Re-prepared with shorter same-exp kernel id/title:

```bash
make prepare-kaggle-notebooks EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp170-heel-calib-shift-scan-pfbeam-train --title 'exp170 heel calib shift scan pfbeam train' --run-on-push --strict"
make push-kaggle-train EXP=exp170_heel_calibrated_shift_scan_pfbeam_audit
```

- result: Kernel version 1 pushed successfully.
- URL: https://www.kaggle.com/code/kentookumura/exp170-heel-calib-shift-scan-pfbeam-train

### 次アクション

### Kaggle train v1 結果

- status: COMPLETE
- kernel: `kentookumura/exp170-heel-calib-shift-scan-pfbeam-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp170-heel-calib-shift-scan-pfbeam-train
- logs: 773 wells の audit が完走し、生成物 path を出力した。
- `kaggle kernels output` は大きい `row_context.csv.gz` 取得中に中断し、集計 CSV だけ取得した。

主要結果:

- `raw__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.058258
- `rolling_median_11__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.076816
- `savgol_31_p2__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.784873
- `raw__flat_calibrated` hidden_tail mean abs-error gain vs raw: -27.570070
- `rolling_median_11__raw` hidden_tail mean abs-error gain vs raw: +0.209137
- `likpf_mean` hidden_tail RMSE: 11.471434
- `raw__heel_calibrated` `likpf_mean` observation rank: mean 19.346994 / top1 rate 0.045445 / top5 rate 0.215344
- `raw__raw` `likpf_mean` observation rank: mean 18.163254 / top1 rate 0.052105 / top5 rate 0.233162

判断:

- heel calibration は shift-scan top1 を改善しない。
- PF/Beam observation cost の mean gap は少し下がるが、rank/top1/top5 が悪化するため採用しない。
- inference port / submit はしない。

### 次アクション

1. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` を更新する。
2. GR alignment 系は heel calibration ではなく、rolling/savgol smoothing の小 scoped audit または bimodal posterior diagnostics に寄せる。
