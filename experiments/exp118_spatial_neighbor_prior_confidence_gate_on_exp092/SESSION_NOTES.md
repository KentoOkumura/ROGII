# exp118_spatial_neighbor_prior_confidence_gate_on_exp092 セッションノート

## 目的

exp114 spatial neighbor prior を exp092 OOF prediction に直接補正する場合、target-free confidence gate で悪化 well を抑制できるか train-side に診断する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train v1 完了、review 候補、提出なし
- CV: best RMSE 9.321625436
- LB: なし

## コマンドログ

### 2026-06-24 実装

```bash
make new-steering EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092
make new-exp EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092 SOURCE=experiments/exp114_spatial_neighbor_prior_signal_audit
```

- `.steering/20260624-exp118-spatial-neighbor-prior-confidence-gate-on-exp092/` を作成。
- exp114 から実験を派生し、実装を posthoc confidence gate audit に差し替えた。
- 入力は exp114 OOF prior と exp092 OOF predictions。
- inference notebook は no-submission guard。

### 予定

```bash
.venv/bin/python -m py_compile experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092/spatial_neighbor_prior_confidence_gate_on_exp092.py
.venv/bin/ruff check experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092/spatial_neighbor_prior_confidence_gate_on_exp092.py
make validate-exp EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092
make prepare-kaggle-notebooks EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --run-on-push --strict"
```

- `py_compile` / `ruff` / `validate-exp` は成功。
- 初回 package は長い slug で `kaggle kernels push` が SaveKernel 400 になったため、kernel id を `kentookumura/exp118-spatial-gate-exp092-train`、title を `exp118 spatial gate exp092 train` に短縮して再生成した。

### 2026-06-24 Kaggle train v1

```bash
make prepare-kaggle-notebooks EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp118-spatial-gate-exp092-train --title 'exp118 spatial gate exp092 train' --run-on-push --strict"
make push-kaggle-train EXP=exp118_spatial_neighbor_prior_confidence_gate_on_exp092
kaggle kernels status kentookumura/exp118-spatial-gate-exp092-train
kaggle kernels logs kentookumura/exp118-spatial-gate-exp092-train
kaggle kernels output kentookumura/exp118-spatial-gate-exp092-train -p experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092/kaggle/output/train_v1
timeout 600 kaggle kernels output kentookumura/exp118-spatial-gate-exp092-train -p experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092/kaggle/output/train_v1
```

- push は成功。Kernel URL: `https://www.kaggle.com/code/kentookumura/exp118-spatial-gate-exp092-train`
- status は `KernelWorkerStatus.COMPLETE`。
- runtime は summary 上 2536.951905 秒。
- best policy: `lgb1__xy_only_k8__std_q50_distance_q50__a0p05__c5`
  - RMSE 9.321625436
  - baseline exp092 `lgb1` RMSE 9.322479896
  - delta RMSE -0.000854460
  - MAE 5.979567419、delta -0.001412906
  - within10 0.822200857、delta +0.000153806
  - gate rate 0.257382355
  - correction abs mean 0.189312、p95 0.25、max 0.25
- by-well:
  - 192 改善 / 195 悪化 / 386 同値
  - max regression +0.208085 RMSE
  - max improvement -0.208868 RMSE
  - mean delta -0.001259 RMSE
- bucket delta:
  - 000_050: -0.023932
  - 050_100: -0.019074
  - 100_250: -0.007222
  - 250_500: +0.001466
  - 500_1000: +0.004397
  - 1000_plus: -0.001158
- path continuity:
  - baseline step >=10: 1 / step >=25: 0
  - best step >=10: 1 / step >=25: 0
  - correction step >=5: 0
  - correction step max: 0.5 ft
- output 取得:
  - `gate_metrics`、`by_well`、`by_well_delta`、`bucket_metrics`、`path_continuity`、`feature_schema`、`summary` は取得済み。
  - `top_gated_predictions.csv.gz` は Kaggle summary に SHA があるが、ローカル output download が 600 秒で timeout し、ローカルファイルは 0 byte。

## 判断

confidence gate は review 候補として支持。exp114 direct correction の worst-well regression は大きく抑えられ、path continuity も維持した。一方で改善幅は RMSE -0.000854 と小さく、250-1000 ft bucket はわずかに悪化するため、direct inference port / submit はまだしない。提出候補化するなら raw-test/full-train parity と exp115 hidden-like stress readout を先に確認する。

## 変更点

- exp092 `lgb1` OOF prediction を baseline とする。
- exp114 の `xy_plus_trajectory_shape_k8` と `xy_only_k8` prior を評価する。
- correction は `exp092_pred + alpha * clip(prior_tvt - exp092_pred, -clip, clip)`。
- gate は prior std、neighbor distance、neighbor wells、azimuth mismatch、abs delta cap を使う。

## 再現性メモ

- seed policy: deterministic_posthoc_grid_no_model_rng
- stochastic components: upstream exp092 / exp114 の生成物のみ
- submission: なし
