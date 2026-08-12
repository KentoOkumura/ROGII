# exp114_spatial_neighbor_prior_signal_audit セッションノート

## 目的

X/Y 近傍に加えて掘削方向・軌跡形状が似た train wells から作る spatial neighbor prior が、PF/Beam/likPF の TVT 誤差方向を説明できるか fold-safe に診断する。

## 現在の状態

- Route: ensemble
- 状態: Kaggle train v1 完了、direct submit なし
- CV: best RMSE 11.151818387
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 2026-06-23 実装

```bash
make new-steering EXP=exp114_spatial_neighbor_prior_signal_audit
make new-exp EXP=exp114_spatial_neighbor_prior_signal_audit
.venv/bin/python -m py_compile experiments/exp114_spatial_neighbor_prior_signal_audit/spatial_neighbor_prior_signal_audit.py
make validate-exp EXP=exp114_spatial_neighbor_prior_signal_audit
make prepare-kaggle-notebooks EXP=exp114_spatial_neighbor_prior_signal_audit EXTRA_ARGS="--notebook train --run-on-push --strict"
.venv/bin/ruff check experiments/exp114_spatial_neighbor_prior_signal_audit/spatial_neighbor_prior_signal_audit.py
```

- `docs/legacy/steering/20260623-exp114-spatial-neighbor-prior-signal-audit/` を作成し、requirements / design / tasklist を記入した。
- `config.yaml` に `ensemble` route、exp099 parent、fold-safe leakage policy、4 つの spatial neighbor variant、出力契約を追加した。
- `spatial_neighbor_prior_signal_audit.py` を追加した。
- train notebook を入力確認、audit 実行、生成物確認セルに更新した。
- inference notebook は no-submission guard に変更した。
- 構文チェック、実験 validation、Kaggle train package 生成、対象ファイル lint は通過。
- 生成 metadata は `kentookumura/exp114-spatial-neighbor-prior-signal-audit-train`、CPU、internet off、run_on_push true、kernel sources は exp099 / exp065。

### 予定

```bash
task push-kaggle-train EXP=exp114_spatial_neighbor_prior_signal_audit
task kaggle-status KERNEL=<username>/<train-kernel-slug>
```

### 2026-06-23 Kaggle train push

```bash
make push-kaggle-train EXP=exp114_spatial_neighbor_prior_signal_audit
make prepare-kaggle-notebooks EXP=exp114_spatial_neighbor_prior_signal_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp114-spatial-neighbor-prior-signal-audit-train --title 'exp114 spatial neighbor prior signal audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp114_spatial_neighbor_prior_signal_audit
kaggle kernels pull kentookumura/exp114-spatial-neighbor-prior-signal-audit-train -p /tmp/kaggle-pull/exp114-spatial-neighbor-prior-signal-audit-train -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
kaggle kernels logs kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
kaggle kernels output kentookumura/exp114-spatial-neighbor-prior-signal-audit-train -p experiments/exp114_spatial_neighbor_prior_signal_audit/kaggle/output/train_v1
kaggle kernels status kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
```

- 初回 push は Kaggle 400 で失敗。原因は generated metadata の title `ROGII - Wellbore Geology Prediction exp114_spatial_neighbor_prior_signal_audit train` から生成される slug が `kentookumura/exp114-spatial-neighbor-prior-signal-audit-train` と一致しないこと。
- `--kernel-id kentookumura/exp114-spatial-neighbor-prior-signal-audit-train --title 'exp114 spatial neighbor prior signal audit train'` で train package を再生成した。
- 再 push は成功し、Kernel version 1 が作成された。
- Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp114-spatial-neighbor-prior-signal-audit-train`
- `kernels pull -m` は成功し、Kaggle 側の存在を確認した。
- `logs -f` と通常 `logs` は空。`output` も実行中時点では空。
- `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。
- ユーザー指示により監視を停止。Kaggle kernel 自体は running のまま。

### 2026-06-24 Kaggle train v1 output 取得

```bash
kaggle kernels status kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
kaggle kernels logs kentookumura/exp114-spatial-neighbor-prior-signal-audit-train
kaggle kernels output kentookumura/exp114-spatial-neighbor-prior-signal-audit-train -p experiments/exp114_spatial_neighbor_prior_signal_audit/kaggle/output/train_v1
```

- status は `KernelWorkerStatus.COMPLETE`。
- Kaggle train v1 runtime は summary 上 2770.882 秒。Notebook log の best print は 2833 秒付近。
- 取得先: `experiments/exp114_spatial_neighbor_prior_signal_audit/kaggle/output/train_v1/`
- best candidate: `xy_plus_trajectory_shape_k8_likpf_mean_corr_a0p2_c40`
  - RMSE 11.151818387 / MAE 7.062013290 / within10 0.779284506 / rows 3,783,989 / wells 773
  - `likpf_mean` RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479
  - delta RMSE vs `likpf_mean`: -0.443079285
- Variant best:
  - `xy_plus_trajectory_shape_k8_likpf_mean_corr_a0p2_c40`: RMSE 11.151818387
  - `xy_only_k8_likpf_mean_corr_a0p2_c40`: RMSE 11.157375869
  - `xy_plus_direction_and_typewell_k8_likpf_mean_corr_a0p2_c40`: RMSE 11.200408
  - `xy_plus_azimuth_k8_likpf_mean_corr_a0p2_c40`: RMSE 11.203437
- 全 distance bucket で `likpf_mean` より RMSE 改善:
  - 000_050: -0.200437
  - 050_100: -0.269320
  - 100_250: -0.261381
  - 250_500: -0.193208
  - 500_1000: -0.139782
  - 1000_plus: -0.491153
- by-well は 416 改善 / 357 悪化 / 0 同値、最大悪化 +6.508121 RMSE、最大改善 -6.606688 RMSE。
- signal metric:
  - `xy_plus_trajectory_shape_k8` vs `likpf_mean`: correlation 0.273578、sign-match 0.582275、prior beats base rate 0.314600。
  - `xy_plus_trajectory_shape_k8` vs `pf_ancc`: correlation 0.399221、sign-match 0.599769。
- 生成物 SHA:
  - `candidate_metrics`: `846331be7e9d207687e6998ea7a53f00fd5c4484d224f4efd0e924b01e5d5de5`
  - `signal_metrics`: `e21268970a1f532864bff302614406ecae72b0c7dff820bd834254044191f9d8`
  - `bucket_metrics`: `3f17da6953cb859b0b7ff935b6567d741ae923889fd6b4ed0eb5068e976a8497`
  - `by_well`: `5268d90e0ad5f97365a5e05517df7ed9e1c867036f108c6d465cd2cb258ccd87`
  - `oof_predictions_raw`: `7a328efd941b4acce476622d3e65e775c65bc9a385c600cdfed9efe3f0d75aa0`
  - `oof_predictions_decompressed`: `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`
- 判断: global と bucket は強いが、worst-well regression が大きいため direct inference port / submit はしない。これは「prior は強いが補正すると悪化 well が出る」分岐なので、spatial prior は confidence / gate の材料として扱う。ML に特徴量として入れる評価は別 backlog として扱う。

## 変更点

- `xy_only_k8`: centroid X/Y だけの control。
- `xy_plus_azimuth_k8`: centroid X/Y に signed azimuth を追加。
- `xy_plus_trajectory_shape_k8`: start/end、bbox、local azimuth、md/z span、dZ/dMD、tortuosity、prefix TVT range を追加。
- `xy_plus_direction_and_typewell_k8`: shape + direction に native overlap 0.999 same-cluster 制約を追加。
- `candidate_metrics`、`signal_metrics`、`bucket_metrics`、`by_well`、`neighbor_summary`、`well_geometry_summary`、`oof_predictions`、`summary.json` を保存する。

## 再現性メモ

- seed policy: deterministic_groupkfold_fixed_neighbor_rules_no_model_rng
- stochastic components: この実験固有の stochastic 処理なし。upstream exp099 / exp065 cache は固定入力として扱う。
- CPU/GPU runtime: CPU、GPU なし。
- Kaggle kernel id / version: `kentookumura/exp114-spatial-neighbor-prior-signal-audit-train` v1。
- input / feature schema SHA: summary JSON に記録済み。
- feature content SHA: OOF decompressed `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`。
- model manifest / model SHA: モデルなし。
- prediction SHA: OOF raw gzip `7a328efd941b4acce476622d3e65e775c65bc9a385c600cdfed9efe3f0d75aa0`。
- submission SHA: submission は作らない。
- rerun check: 未実施。

## 次のアクション

1. direct correction は閉じる。
2. `spatial_neighbor_prior_confidence_gate_on_exp092` として、spatial prior を信用してよい row/well を判定する confidence / gate follow-up を検討する。
3. ML に特徴量として入れる評価は `spatial_neighbor_prior_ml_features_on_exp092` として別 backlog に分ける。
