# exp151_tvt_dense_addonly_confidence_features_on_exp092 セッションノート

## 2026-06-27 実装

- `KAGGLE_DIRECTION.md` の `tvt_dense_addonly_confidence_features_on_exp092` を実験化した。
- `docs/legacy/steering/20260627-exp151-tvt-dense-addonly-confidence-features-on-exp092/` を作成。
- `experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/` を `exp149_normalized_shape_addonly_features_on_exp092` から作成し、実装を dense confidence 用に差し替えた。
- 親実験は `exp092_u_projection_correction_disagreement_fullrun`、cache は `exp072_exp063_full_replay_feature_cache`。
- 追加特徴は `tvt_dense` / `tvt_densew` / `tvt_dense50` の drift、slope、roughness、dense family disagreement、PF/Beam/likPF-vs-dense 差、near/longtail interaction に限定する。
- Dense candidate の prediction replacement、hard switch、oracle label、true error rank、target 変更は入れない。

## GPU コストガード

- active variant 数: 1 (`tvt_dense_confidence_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15
- exp092 control 再学習: なし (`exp092_full_row_control.enabled=false`)
- baseline は保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を参照する。

## 検証ログ

- `python3 -m py_compile experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/tvt_dense_addonly_confidence_features_on_exp092.py experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/settings.py`: PASS
- `python3 -m json.tool experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/exp151_tvt_dense_addonly_confidence_features_on_exp092_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/exp151_tvt_dense_addonly_confidence_features_on_exp092_inference.ipynb`: PASS
- `make validate-exp EXP=exp151_tvt_dense_addonly_confidence_features_on_exp092`: PASS
- `.venv/bin/ruff check experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/tvt_dense_addonly_confidence_features_on_exp092.py experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/settings.py`: PASS
- synthetic frame smoke for `build_tvt_dense_confidence_features()`: PASS。30 features、feature groups は `dense_confidence_geometry` 4、`dense_candidate_path` 12、`dense_candidate_disagreement` 14、finite check PASS。
- `make prepare-kaggle-notebooks EXP=exp151_tvt_dense_addonly_confidence_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp151-tvt-dense-addonly-exp092-train --title 'exp151 tvt dense addonly exp092 train' --run-on-push --strict"`: PASS
- `make prepare-kaggle-notebooks EXP=exp151_tvt_dense_addonly_confidence_features_on_exp092 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp151-tvt-dense-addonly-exp092-infer --title 'exp151 tvt dense addonly exp092 infer' --run-on-push --strict"`: PASS

## Kaggle package

- train kernel id: `kentookumura/exp151-tvt-dense-addonly-exp092-train`
- train title: `exp151 tvt dense addonly exp092 train`
- inference kernel id: `kentookumura/exp151-tvt-dense-addonly-exp092-infer`
- inference title: `exp151 tvt dense addonly exp092 infer`
- train/inference metadata は `enable_gpu=true`、`enable_internet=false`、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`。

## 2026-06-28 Kaggle train v1

- `make push-kaggle-train EXP=exp151_tvt_dense_addonly_confidence_features_on_exp092`: PASS
- Kaggle kernel: `kentookumura/exp151-tvt-dense-addonly-exp092-train`
- Kernel version: 1
- URL: `https://www.kaggle.com/code/kentookumura/exp151-tvt-dense-addonly-exp092-train`
- push output: `Kernel version 1 successfully pushed`
- `kaggle kernels pull kentookumura/exp151-tvt-dense-addonly-exp092-train -p /tmp/kaggle-pull/exp151-tvt-dense-addonly-exp092-train-v1 -m`: PASS
- `kaggle kernels status kentookumura/exp151-tvt-dense-addonly-exp092-train`: `KernelWorkerStatus.RUNNING`
- `kaggle kernels logs kentookumura/exp151-tvt-dense-addonly-exp092-train`: empty log
- `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp151-tvt-dense-addonly-exp092-train`: timeout with empty log
- `kaggle kernels output kentookumura/exp151-tvt-dense-addonly-exp092-train -p /tmp/kaggle-output/exp151_train_probe`: no files yet
- checked at `2026-06-28 00:21:31 UTC`: still running. Do not re-push under a different slug; continue monitoring the same kernel id.

## 2026-06-28 Kaggle train v1 result

- User reported completion; `kaggle kernels status kentookumura/exp151-tvt-dense-addonly-exp092-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp151-tvt-dense-addonly-exp092-train`: PASS。Notebook completed with `train_completed`.
- `kaggle kernels output kentookumura/exp151-tvt-dense-addonly-exp092-train -p experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/kaggle/output/train_v1`: PASS
- output: `experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/kaggle/output/train_v1`
- runtime in summary: 15307.819 sec
- rows: 3,783,989
- wells: 773
- generated dense features: 30
- total features: 270
- saved model count: 15 boosters
- pooled OOF:
  - `lgb0`: 9.597243443, delta vs exp092 `lgb0` +0.064117005
  - `lgb1`: 9.355161771, delta vs exp092 `lgb1` +0.032681876
  - `lgb2`: 9.375504593, delta vs exp092 `lgb2` +0.037312188
  - `lgb_mean`: 9.388714996, delta vs exp092 `lgb_mean` +0.045650930
- best model: `lgb1`, but it is worse than exp092 `lgb1` 9.322479896.
- prediction SHA:
  - `lgb0`: `6e15218026021241b50b6a9a46fefb55001ef3e40ca1df06f445eece1e2a6cfa`
  - `lgb1`: `167f9404376e9d742c3a9e04ef8e3330af6156779938ed2d13a8e07d5fb73b9a`
  - `lgb2`: `5fdd97d392b3bbd1c0ab0e3bd73ba230d1dba9c1cbf9b8bb68b056545f7beb78`
  - `lgb_mean`: `475d8334cbf4adfea839ec5cb287873f8c78e6f6d8dae94946a53955bd7c8933`
- decision: completed train-side rejected. Do not inference-port or submit.
