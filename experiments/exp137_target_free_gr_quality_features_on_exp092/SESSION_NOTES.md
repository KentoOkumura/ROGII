# exp137_target_free_gr_quality_features_on_exp092 セッションノート

## 状態

- 2026-06-26: 実装済み。Kaggle train は未実行。
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- quality context 親: `exp065_typewell_supertype_cluster_cv_audit`
- submission: なし

## 実装メモ

- `backlog/KAGGLE_DIRECTION.md` の `target_free_gr_quality_features_on_exp092` を実験化する。
- exp130 を派生元にし、exp092 相当の U-projection correction / disagreement surface と LightGBM train loop を再利用する。
- 追加特徴量は GR 品質 / coverage のみに限定する。
  - prefix/eval/full GR missing rate
  - prefix/eval missing run max
  - row-level missing flag、missing run length、nearest finite GR gap、finite GR bracket flag
  - prefix/eval GR median shift と robust scale ratio
  - exp065 native overlap / exact hash / shifted NCC / DTW group context
- 生 GR 値、row-wise NCC/DTW score、GR 由来 candidate TVT、hard switch は入れない。
- 初回 Kaggle train は CPU deterministic mode とし、GPU quota には依存しない。

## コマンド

```bash
make new-steering EXP=exp137_target_free_gr_quality_features_on_exp092
make new-exp EXP=exp137_target_free_gr_quality_features_on_exp092 SOURCE=experiments/exp130_pfbeam_normalized_diagnostic_score
python3 -m py_compile experiments/exp137_target_free_gr_quality_features_on_exp092/target_free_gr_quality_features_on_exp092.py experiments/exp137_target_free_gr_quality_features_on_exp092/settings.py
python3 -m json.tool experiments/exp137_target_free_gr_quality_features_on_exp092/exp137_target_free_gr_quality_features_on_exp092_train.ipynb
python3 -m json.tool experiments/exp137_target_free_gr_quality_features_on_exp092/exp137_target_free_gr_quality_features_on_exp092_inference.ipynb
.venv/bin/ruff check experiments/exp137_target_free_gr_quality_features_on_exp092/target_free_gr_quality_features_on_exp092.py experiments/exp137_target_free_gr_quality_features_on_exp092/settings.py
.venv/bin/ruff format experiments/exp137_target_free_gr_quality_features_on_exp092/target_free_gr_quality_features_on_exp092.py experiments/exp137_target_free_gr_quality_features_on_exp092/settings.py
make validate-exp EXP=exp137_target_free_gr_quality_features_on_exp092
make prepare-kaggle-notebooks EXP=exp137_target_free_gr_quality_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp137-gr-quality-train --title 'exp137 gr quality train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp137_target_free_gr_quality_features_on_exp092 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp137-gr-quality-infer --title 'exp137 gr quality infer' --run-on-push --strict"
make update-summary
make push-kaggle-train EXP=exp137_target_free_gr_quality_features_on_exp092
kaggle kernels pull kentookumura/exp137-gr-quality-train -p /tmp/kaggle-pull/exp137-gr-quality-train-v1 -m
kaggle kernels status kentookumura/exp137-gr-quality-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp137-gr-quality-train
```

## 検証

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS
- synthetic frame による `build_target_free_gr_quality_features()` smoke: PASS。24 rows / 32 GR quality features / 3 summary rows。
- `make validate-exp EXP=exp137_target_free_gr_quality_features_on_exp092`: PASS
- train Kaggle package: `experiments/exp137_target_free_gr_quality_features_on_exp092/kaggle/train`
  - kernel id: `kentookumura/exp137-gr-quality-train`
  - title: `exp137 gr quality train`
  - GPU: disabled
  - internet: disabled
  - kernel sources: `kentookumura/exp072-exp063-full-replay-feature-cache-train`, `kentookumura/exp065-typewell-supertype-cluster-cv-audit-train`
- inference Kaggle package: `experiments/exp137_target_free_gr_quality_features_on_exp092/kaggle/inference`
  - kernel id: `kentookumura/exp137-gr-quality-infer`
  - submission generation disabled

## Kaggle train v1

- canonical kernel id: `kentookumura/exp137-gr-quality-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp137-gr-quality-train`
- push: `Kernel version 1 successfully pushed`
- existence check: `kaggle kernels pull kentookumura/exp137-gr-quality-train -p /tmp/kaggle-pull/exp137-gr-quality-train-v1 -m`: PASS
- status after push: `KernelWorkerStatus.RUNNING`
- normal `logs`: empty while running
- `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp137-gr-quality-train`: no log output before timeout
- probe output path: `/tmp/kaggle-output/exp137_target_free_gr_quality_features_on_exp092/train_v1_probe`; no files while running
- 2026-06-27 status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
- failure mode: Kaggle runtime timeout/cancel after about 42,949 seconds.
- v1 partial log:
  - full cache rows: 3,783,989
  - `exp092_full_row_control`: all 3 LGBM configs completed.
  - `target_free_gr_quality_addonly`: `lgb0` and `lgb1` completed, `lgb2` did not complete before timeout.
  - `target_free_gr_quality_addonly` `lgb0` fold RMSE: 8.993835, 9.576391, 8.485171, 10.797023, 10.590807.
  - `target_free_gr_quality_addonly` `lgb1` fold RMSE: 8.486715, 9.519227, 8.396092, 10.082622, 10.412099.
- interpretation: Implementation reached model training, but the full 2 variants x 3 configs plan is too heavy for the CPU notebook limit.
- partial output retrieval:
  - command: `kaggle kernels output kentookumura/exp137-gr-quality-train -p experiments/exp137_target_free_gr_quality_features_on_exp092/kaggle/output/train_v1`
  - result: manually interrupted after 15 files because final metrics were not present and the download was slow.
  - downloaded: diagnostic feature summary and partial `exp092_full_row_control` model files.
  - not downloaded / not available from partial run: final metrics CSV, prediction CSV, model manifest, summary JSON.

## Kaggle train v2

- change: add `model.training.active_model_indices: [0]` and pass it from the train notebook into the training function.
- intent: keep the full-row `exp092_full_row_control` vs `target_free_gr_quality_addonly` comparison, but run only the `lgb0` config so the Kaggle CPU run can finish inside the runtime limit.
- static checks:
  - `python3 -m py_compile ...`: PASS
  - train notebook JSON: PASS
  - `ruff check`: PASS
  - `ruff format --check`: PASS
  - `make validate-exp EXP=exp137_target_free_gr_quality_features_on_exp092`: PASS
- package:
  - `make prepare-kaggle-notebooks EXP=exp137_target_free_gr_quality_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp137-gr-quality-train --title 'exp137 gr quality train' --run-on-push --strict"`: PASS
- push:
  - `make push-kaggle-train EXP=exp137_target_free_gr_quality_features_on_exp092`
  - result: `Kernel version 2 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp137-gr-quality-train`
  - status after push: `KernelWorkerStatus.RUNNING`
  - `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp137-gr-quality-train`: no log output during the first 3 minutes.
  - status after short log watch: `KernelWorkerStatus.RUNNING`
- completion:
  - status after user completion notice: `KernelWorkerStatus.COMPLETE`
  - log status: `train_completed`
  - elapsed seconds: 16038.686
  - active model indices: `[0]`
  - generated target-free GR quality features: 32
  - output command: `kaggle kernels output kentookumura/exp137-gr-quality-train -p experiments/exp137_target_free_gr_quality_features_on_exp092/kaggle/output/train_v2`
  - output path: `experiments/exp137_target_free_gr_quality_features_on_exp092/kaggle/output/train_v2`
  - source feature SHA256: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- v2 pooled metrics:
  - `exp092_full_row_control` `lgb0`: RMSE 9.535793274086194, 3,783,989 rows, 240 features.
  - `target_free_gr_quality_addonly` `lgb0`: RMSE 9.72965651223404, 3,783,989 rows, 272 features.
  - delta addonly - control: +0.19386323814784667.
- v2 bucket deltas addonly - control:
  - distance `000_050`: -0.007979
  - distance `050_100`: -0.021365
  - distance `100_250`: +0.037757
  - distance `250_500`: +0.103499
  - distance `500_1000`: +0.064636
  - distance `1000_plus`: +0.215175
  - tail rank `500_999`: +0.123502
  - tail rank `1000_plus`: +0.193867
- v2 by-well delta addonly - control:
  - count: 773
  - mean: +0.084110
  - median: +0.043437
  - min: -8.368730
  - max: +12.713932
- top GR quality importances:
  - `grq_prefix_eval_gr_scale_ratio`: 1666.0
  - `grq_prefix_eval_gr_median_shift_norm`: 1407.2
  - `grq_prefix_eval_missing_rate_gap`: 1340.6
  - `grq_prefix_gr_missing_rate`: 1171.2
- decision: train-side audit rejected. Do not port to inference or submit.

## 次アクション

1. `backlog/KAGGLE_DIRECTION.md` から実装済み backlog を削除する。
2. `experiment_summary.md` を更新する。
3. GR quality 系の後続は、全 row add-only ではなく near-prefix guard / segment verifier の限定用途に寄せる。
