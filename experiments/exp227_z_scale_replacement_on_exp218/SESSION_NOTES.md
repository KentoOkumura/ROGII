# exp227_z_scale_replacement_on_exp218 セッションノート

## 2026-07-09 実装

`z_scale_replacement_on_exp218` を実装する。exp218 の GRWR add-only ML submitted anchor に対し、exp224 の target-free well-scaled `z` / `dz` / `dzdmd` / `slp_z` feature builder を再利用する。ただし exp224 の add-only ではなく、raw 4 列を model feature list から落として replacement-only として評価する。

### 根拠

- parent: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- exp218 `lgb_mean` CV: 8.475793752
- exp218 Public LB: 7.843
- exp224 add-only `lgb_mean` CV: 8.538687042
- exp224 delta vs exp218: +0.062893290

### 実装

- `exp224_well_scaled_z_dz_features_on_exp218` から `exp227_z_scale_replacement_on_exp218` を作成。
- helper を `z_scale_replacement_on_exp218.py` にリネーム。
- active variant: `z_scale_replacement`
- `drop_base_columns: [z, dz, dzdmd, slp_z]`
- `feature_columns_for_variant()` に `drop_base_columns` 処理を追加。
- train index / `train_lgb0` / `train_lgb1` / `train_lgb2` / inference scaffold を exp227 名へ更新。

### Kaggle train cost guard

- runtime: CPU
- active variant: 1 (`z_scale_replacement`)
- active mode: 1 (`cpu_deterministic_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- boosters per split: 5
- total planned boosters: 15
- control / parent retraining: なし

### 禁止事項

- target-derived scaler
- direct z correction
- candidate replacement
- blend
- postprocess
- hard selector
- sample-weight 変更
- raw 4 列を残す add-only 再試行

### 再現性メモ

`docs/06_reproducibility.md` を確認済み。CPU deterministic LightGBM、seed 42、fixed threads、Kaggle bootstrap 再生成、split train output の model manifest / feature schema / prediction SHA 記録を前提にする。train 完了後、必要な場合だけ Kaggle output を取得して decompressed prediction SHA と aggregate readout を保存する。

### 次

Jupytext 変換、静的チェック、`validate_exp`、Kaggle package prepare を通した後、`train_lgb0` / `train_lgb1` / `train_lgb2` を push する。

## 2026-07-09 Kaggle train push

以下の CPU split notebook を Kaggle に push し、3 本とも version 1 として作成された。metadata pull も成功し、status は `KernelWorkerStatus.RUNNING`。

```bash
kaggle kernels push -p experiments/exp227_z_scale_replacement_on_exp218/kaggle/train_lgb0
kaggle kernels push -p experiments/exp227_z_scale_replacement_on_exp218/kaggle/train_lgb1
kaggle kernels push -p experiments/exp227_z_scale_replacement_on_exp218/kaggle/train_lgb2
kaggle kernels pull kentookumura/exp227-zscale-exp218-lgb0 -p /tmp/kaggle-pull/exp227-zscale-exp218-lgb0 -m
kaggle kernels pull kentookumura/exp227-zscale-exp218-lgb1 -p /tmp/kaggle-pull/exp227-zscale-exp218-lgb1 -m
kaggle kernels pull kentookumura/exp227-zscale-exp218-lgb2 -p /tmp/kaggle-pull/exp227-zscale-exp218-lgb2 -m
kaggle kernels status kentookumura/exp227-zscale-exp218-lgb0
kaggle kernels status kentookumura/exp227-zscale-exp218-lgb1
kaggle kernels status kentookumura/exp227-zscale-exp218-lgb2
```

- `train_lgb0`: `kentookumura/exp227-zscale-exp218-lgb0` v1 / https://www.kaggle.com/code/kentookumura/exp227-zscale-exp218-lgb0 / `RUNNING`
- `train_lgb1`: `kentookumura/exp227-zscale-exp218-lgb1` v1 / https://www.kaggle.com/code/kentookumura/exp227-zscale-exp218-lgb1 / `RUNNING`
- `train_lgb2`: `kentookumura/exp227-zscale-exp218-lgb2` v1 / https://www.kaggle.com/code/kentookumura/exp227-zscale-exp218-lgb2 / `RUNNING`

次は Kaggle UI または `kaggle kernels logs` / `status` で完了を確認する。CV 評価だけなら output archive は取得せず、logs / notebook output を根拠に記録する。split OOF aggregate、feature importance、SHA 確認が必要な場合だけ output を取得する。

## 2026-07-10 Kaggle CPU split train 完了 / OOF 集計

status:

- `train_lgb0`: `KernelWorkerStatus.COMPLETE`
- `train_lgb1`: `KernelWorkerStatus.COMPLETE`
- `train_lgb2`: `KernelWorkerStatus.COMPLETE`

logs / output:

- `train_lgb0`: RMSE TVT 8.665460822415、RMSE target 8.665460907285、elapsed 16469.808 sec、full coverage pass
- `train_lgb1`: RMSE TVT 8.578969199552、RMSE target 8.578969207254、elapsed 12905.693 sec、full coverage pass
- `train_lgb2`: RMSE TVT 8.599021782864、RMSE target 8.599021940658、elapsed 11728.436 sec、full coverage pass
- coverage は 3 split すべて 3,783,989 rows / 773 wells、dropped rows 0

downloaded outputs:

- `kaggle/output/train_lgb0_v1/artifacts/`
- `kaggle/output/train_lgb1_v1/artifacts/`
- `kaggle/output/train_lgb2_v1/artifacts/`

split OOF aggregate:

- script: `aggregate_split_oof.py`
- output: `kaggle/output/train_split_aggregate_v1/artifacts/`
- 3-config `lgb_mean` RMSE TVT: 8.561884246773
- RMSE target: 8.561884363571
- prediction SHA256: `38f7632b3e0442d1e667bff6b866cc3613775826e01ce22d76c6f395dac7c460`
- exp218 parent `lgb_mean` 8.475793751657 から +0.086090495116 悪化
- exp148 feature surface `lgb_mean` 8.501281181896 から +0.060603064877 悪化
- exp224 add-only `lgb_mean` 8.538687041980 から +0.023197204793 悪化

readout:

- top importance: `slp_b_d_50`, `spatial_knn_dist`, `wsz_dz_over_likpf_tvt_p05_p95_range`, `ll_learned_pred_abs_error_beam_mean`, `grwr_fft_rotation_ratio_x_log1p_md_since`
- worst wells top3: `86454a6f` RMSE 48.296673、`1b1eba53` RMSE 45.455573、`fb03ae90` RMSE 45.418222
- aggregate bucket は `md_since` が split prediction archive にないため tail-rank bucket のみ厳密再計算。distance bucket は各 split の既存 CSV を `split_individual_bucket_metrics.csv` に連結した。

判定:

- CV が exp218 / exp148 / exp224 add-only のすべてより悪いため不採用。
- inference port、saved-booster aggregate manifest、submit は行わない。
- raw `z` / `dz` / `dzdmd` / `slp_z` を well-scaled z 系へ置き換える仮説は反証。exp224 add-only よりさらに悪いため、この z scale 系は直接 correction / replacement ではなく readout または限定的な confidence feature に留める。
