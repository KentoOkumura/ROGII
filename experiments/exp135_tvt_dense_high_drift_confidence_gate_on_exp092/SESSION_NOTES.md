# exp135_tvt_dense_high_drift_confidence_gate_on_exp092 セッションノート

## 現在の状態

- status: `completed_train_side_rejected_no_submit`
- route: `ml_model`
- parent: `exp092_u_projection_correction_disagreement_fullrun`
- CV: exp092 base 9.322479896、best gate 9.874846008
- LB: -
- inference: disabled diagnostic only
- blocked: none

## 実装内容

- `.steering/20260626-exp135-tvt-dense-high-drift-confidence-gate-on-exp092/` を作成。
- `experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/` を exp126 から作成。
- 実装本体を `tvt_dense_high_drift_confidence_gate_on_exp092.py` に置き換えた。
- `config.yaml` を exp092 base / exp073 reference / exp072 dense feature cache の no-training posthoc audit に更新した。
- `README.md`、`result.md`、`metrics.json` を未実行状態に更新した。

## 設計メモ

- LightGBM の新規学習は行わない。
- `target_tvt` は scoring と oracle coverage readout だけに使う。
- gate 条件は `dense_std_abs`、`tvt_dense_d_abs`、`pf_dense_abs_diff`、`exp092_dense_abs_diff`、`pf_beam_abs_diff`、`tail_rank` のみ。
- scope は `segment` と `well` を比較する。
- OOF 改善だけでは提出候補にせず、common worst、near-row、path continuity、worst-well regression、raw-test parity を合否条件にする。

## 実行予定

```bash
uv run python -m py_compile experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/tvt_dense_high_drift_confidence_gate_on_exp092.py experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/settings.py
uv run ruff check experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/tvt_dense_high_drift_confidence_gate_on_exp092.py experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/settings.py
uv run ruff format --check experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/tvt_dense_high_drift_confidence_gate_on_exp092.py experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/settings.py
make validate-exp EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092
make prepare-kaggle-notebooks EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp135-tvt-dense-gate-train --title 'exp135 tvt dense gate train' --run-on-push --strict"
make push-kaggle-train EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092
```

## Kaggle train v1

```bash
make prepare-kaggle-notebooks EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp135-tvt-dense-gate-train --title 'exp135 tvt dense gate train' --run-on-push --strict"
make push-kaggle-train EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092
kaggle kernels status kentookumura/exp135-tvt-dense-gate-train
kaggle kernels logs kentookumura/exp135-tvt-dense-gate-train
kaggle kernels output kentookumura/exp135-tvt-dense-gate-train -p experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/kaggle/output/train_v1
```

- kernel: `kentookumura/exp135-tvt-dense-gate-train`
- version: 1
- status: `KernelWorkerStatus.ERROR`
- URL: `https://www.kaggle.com/code/kentookumura/exp135-tvt-dense-gate-train`
- output: `experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/kaggle/output/train_v1`
- failure: notebook cell 2 failed with `ImportError: cannot import name 'paths' from 'settings'`.
- cause: copied notebook used `from settings import paths`, but template `settings.py` exposes `ExperimentPaths` and does not instantiate `paths`.
- fix: train / inference notebooks now import `ExperimentPaths` and instantiate `paths = ExperimentPaths()`.
- post-fix validation: `python3 -m json.tool` for train/inference notebooks PASS, `make validate-exp EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092` PASS.

## Kaggle train v2

```bash
make prepare-kaggle-notebooks EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp135-tvt-dense-gate-train --title 'exp135 tvt dense gate train' --run-on-push --strict"
make push-kaggle-train EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092
```

- kernel: `kentookumura/exp135-tvt-dense-gate-train`
- version: 2
- push: success
- URL: `https://www.kaggle.com/code/kentookumura/exp135-tvt-dense-gate-train`
- monitoring: stopped per user request; user will report completion.

## Kaggle train v2 completion

```bash
kaggle kernels status kentookumura/exp135-tvt-dense-gate-train
kaggle kernels logs kentookumura/exp135-tvt-dense-gate-train
kaggle kernels output kentookumura/exp135-tvt-dense-gate-train -p experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/kaggle/output/train_v2
```

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/kaggle/output/train_v2`
- rows / wells: 3,783,989 / 773
- runtime in summary: 246.318 sec
- base `exp092_lgb1`: RMSE 9.322479896 / MAE 5.980980396 / within10 0.822047051
- best non-oracle: base `exp092_lgb1`; all configured gates are worse overall.
- best gate by global RMSE: `seg_dense50_q75_tail1000_min100_clip20_a050`, RMSE 9.874846008, delta +0.552366112, within10 0.788303296, gate rate 0.320801, gate wells 575, max well regression +9.535752.
- all candidate oracle RMSE: 3.421443766.
- dense candidate oracle RMSE: 10.971366039.
- PF `likpf_mean` worst50: `single_tvt_densew` improves exp092 by -4.652228 RMSE; best clipped segment gate improves by about -3.13 RMSE.
- common PF+ML worst26: `single_tvt_densew` improves exp092 by -6.432887 RMSE, but configured gates still break global OOF.
- distance buckets: best clipped dense50 gate worsens `000_050` by +0.082539, `050_100` by +0.172969, and `1000_plus` by +0.554533.
- worst by-well regressions for best clipped dense50 gate: `389ae58f` +9.535752, `059c8f24` +9.300540, `99529c45` +9.278271, `071d7b45` +9.055897, `b0d42b0d` +8.985595.
- raw-test parity checklist: required columns present, gate conditions target-free, no LightGBM training; inference port not applicable.
- decision: reject direct dense gate. Do not inference-port or submit exp135.

## Code follow-up

- v2 completed with pandas `FutureWarning` about assigning float64 values into a float32 prediction series.
- Local code was updated after v2 to cast `pred` to float64 before gated assignment. This does not change recorded v2 metrics and is only to avoid future pandas incompatibility.
