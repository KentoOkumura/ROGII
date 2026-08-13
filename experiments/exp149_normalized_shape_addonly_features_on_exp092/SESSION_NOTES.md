# exp149_normalized_shape_addonly_features_on_exp092 セッションノート

## 状態

- 2026-06-27: 実装済み。Kaggle train は未実行。
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`

## 実装メモ

- `backlog/KAGGLE_DIRECTION.md` の `normalized_shape_addonly_features_on_exp092` を実験化した。
- exp130 を派生元にしたが、`normalized_diagnostic_score`、shape score probability、high/low confidence flag は削除した。
- 追加特徴は shape 表現に限定する。
  - `md_since_norm`、`tail_md_scale`、target-free `u_scale`
  - `z_rel_norm`、`z_rel_abs_norm`
  - candidate ごとの `u_norm`、last anchor からの normalized drift、prefix line drift
  - candidate path の gradient slope / curvature / roughness
  - candidate path の well-local polynomial residual / slope / curvature
  - candidate 間 normalized U diff / std / range
- Inference notebook は no-submission summary のみを書き、`submission.csv` は作らない。
- GPU cost guard: `exp092_full_row_control` は `enabled: false`。初回 train 対象は 1 variant x 3 LightGBM configs x 5 folds = 15 boosters。

## コマンド

```bash
make new-steering EXP=exp149_normalized_shape_addonly_features_on_exp092
make new-exp EXP=exp149_normalized_shape_addonly_features_on_exp092 SOURCE=experiments/exp130_pfbeam_normalized_diagnostic_score
```

## 検証

- `python3 -m py_compile experiments/exp149_normalized_shape_addonly_features_on_exp092/normalized_shape_addonly_features_on_exp092.py experiments/exp149_normalized_shape_addonly_features_on_exp092/settings.py`: PASS
- `python3 -m json.tool experiments/exp149_normalized_shape_addonly_features_on_exp092/exp149_normalized_shape_addonly_features_on_exp092_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp149_normalized_shape_addonly_features_on_exp092/exp149_normalized_shape_addonly_features_on_exp092_inference.ipynb`: PASS
- `.venv/bin/ruff check experiments/exp149_normalized_shape_addonly_features_on_exp092/normalized_shape_addonly_features_on_exp092.py experiments/exp149_normalized_shape_addonly_features_on_exp092/settings.py`: PASS
- `make validate-exp EXP=exp149_normalized_shape_addonly_features_on_exp092`: PASS
- 合成 frame 40 rows で `build_normalized_shape_features()` smoke: PASS。94 features、feature groups は `normalized_shape_geometry` 7、`normalized_candidate_shape` 65、`normalized_shape_disagreement` 22、finite check PASS。
- `make prepare-kaggle-notebooks EXP=exp149_normalized_shape_addonly_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp149-normalized-shape-addonly-features-on-exp092-train --title 'exp149 normalized shape addonly features on exp092 train' --run-on-push --strict"`: PASS
- `make prepare-kaggle-notebooks EXP=exp149_normalized_shape_addonly_features_on_exp092 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp149-normalized-shape-addonly-features-on-exp092-inference --title 'exp149 normalized shape addonly features on exp092 inference' --run-on-push --strict"`: PASS
- `kaggle/train/kernel-metadata.json`: GPU true、internet false、competition source `rogii-wellbore-geology-prediction`、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`。
- `kaggle/train/config.yaml`: `exp092_full_row_control.enabled=false`、`normalized_shape_addonly.enabled=true`、active mode `gpu_repro_guard_dp_threads8`。

## 次アクション

validation が通れば Kaggle train package を作成する。push 前に、active variant 数 1、LightGBM config 数 3、fold 数 5、合計 booster 数 15、control 再学習なしであることを再確認する。

## Kaggle train push

- 2026-06-27: ユーザー依頼により Kaggle train を実行する。
- 実行予定:
  - active variant 数: 1 (`normalized_shape_addonly`)
  - LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
  - fold 数: 5
  - 合計 booster 数: 15
  - 親実験 / control 再学習: なし (`exp092_full_row_control.enabled=false`)
- kernel id: `kentookumura/exp149-normalized-shape-addonly-features-on-exp092-train`
- title: `exp149 normalized shape addonly features on exp092 train`
- 初回 push:
  - command: `make push-kaggle-train EXP=exp149_normalized_shape_addonly_features_on_exp092`
  - result: failed
  - message: `400 Client Error: Bad Request ... SaveKernel`
  - note: id/title slug は一致していたが長すぎる可能性があるため、同じ実験フォルダのまま短い意味付き slug に再生成する。
- 失敗 id の存在確認:
  - `kaggle kernels pull kentookumura/exp149-normalized-shape-addonly-features-on-exp092-train -p /tmp/kaggle-pull/exp149-normalized-shape-addonly-features-on-exp092-train -m`: `403 Forbidden`
  - `kaggle kernels list --mine --search exp149-normalized-shape-addonly`: `Not found`
- 再 prepare:
  - command: `make prepare-kaggle-notebooks EXP=exp149_normalized_shape_addonly_features_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp149-nshape-addonly-exp092-train --title 'exp149 nshape addonly exp092 train' --run-on-push --strict"`
  - kernel id: `kentookumura/exp149-nshape-addonly-exp092-train`
  - title: `exp149 nshape addonly exp092 train`
- 再 push:
  - command: `make push-kaggle-train EXP=exp149_normalized_shape_addonly_features_on_exp092`
  - result: `Kernel version 1 successfully pushed.`
  - URL: `https://www.kaggle.com/code/kentookumura/exp149-nshape-addonly-exp092-train`
- push 後確認:
  - `kaggle kernels pull kentookumura/exp149-nshape-addonly-exp092-train -p /tmp/kaggle-pull/exp149-nshape-addonly-exp092-train-v1 -m`: PASS
  - `kaggle kernels status kentookumura/exp149-nshape-addonly-exp092-train`: `KernelWorkerStatus.RUNNING`
  - `kaggle kernels logs kentookumura/exp149-nshape-addonly-exp092-train`: empty log
  - `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp149-nshape-addonly-exp092-train`: empty log, timeout
  - second status check: `KernelWorkerStatus.RUNNING`

## Kaggle train v1 result

- 2026-06-28: ユーザーから完了連絡を受け、Kaggle train v1 を確認。
- `kaggle kernels status kentookumura/exp149-nshape-addonly-exp092-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels output kentookumura/exp149-nshape-addonly-exp092-train -p experiments/exp149_normalized_shape_addonly_features_on_exp092/kaggle/output/train_v1`: PASS
- output: `experiments/exp149_normalized_shape_addonly_features_on_exp092/kaggle/output/train_v1`
- Kaggle summary:
  - status: `train_completed`
  - elapsed seconds: 16187.913
  - rows: 3,783,989
  - features: 334
  - added normalized shape features: 94
  - model count: 15
- pooled metrics:
  - `lgb0`: RMSE 9.558981894、exp092 `lgb0` から +0.025855456 悪化
  - `lgb1`: RMSE 9.315846067、exp092 `lgb1` から -0.006633828 改善、exp139 `lgb1` から -0.009061573 改善
  - `lgb2`: RMSE 9.327179161、exp092 `lgb2` から -0.011013243 改善、exp139 `lgb2` から -0.010399150 改善
  - `lgb_mean`: RMSE 9.341688371、exp092 `lgb_mean` から -0.001375695 改善、exp139 `lgb_mean` から -0.028895854 改善
- prediction SHA:
  - `lgb0`: `dff1bff1f3997cc8db3e545bbe3174ebd06913d681a0c5d2e26ef53e7c273980`
  - `lgb1`: `2eed044e805e2b6bffe172b57f91e3002ab396a7ae92c556bca3228aa6fe3e9c`
  - `lgb2`: `ef29c5396effed936cb1d77b93f6a7c8740548ad959f7f486222af1204a18c96`
  - `lgb_mean`: `c98c59ae6e084beff4c9129288c052d7b3eae30c8efa8159baf9dee2e0b57611`
- artifact SHA:
  - predictions decompressed SHA256: `af7191254d1b618aceb0fc9d43bf4061b22f240ad55429cb00d98d0cd00fe561`
  - feature schema SHA256: `720c7b54038c24ba257ea337161633e1839e53e736d3e7e0c9f86f938d4be573`
  - model manifest SHA256: `638dfe06ea40fa568b00a6fd522a075b667a03b5276e4b4b2d8dde6fd1500c5e`
- feature importance:
  - normalized shape features が上位に入った。
  - 主な上位: `nshp_likpf_mean_poly_curvature_norm`、`nshp_pf_z_poly_curvature_norm`、`nshp_pf_ancc_poly_curvature_norm`、candidate polynomial slope 系。
- 判断:
  - train-side OOF は positive。`lgb1` / `lgb2` / `lgb_mean` は exp092 同 model を改善。
  - raw-test feature parity と exp115 hidden-like stress は未確認のため、現時点では inference port / submit はしない。
