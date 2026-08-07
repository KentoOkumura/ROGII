# SESSION_NOTES

## 2026-07-07

### 実装

- `affine_shift_landscape_ruler_readout` バックログ向けに `exp216_affine_shift_landscape_ruler_readout` を実装。
- `scan_filter_for_region` を更新し、ruler readout 列を追加。
  - `best_delta`
  - `second_shift_ft`
  - `second_delta`
  - `second_delta_vs_best`
  - `second_cost`
  - `margin`
  - `zero_shift_ft`
  - `zero_cost`
  - `zero_rank`
  - `secondary_mode_shift_ft`
  - `secondary_mode_gap`
  - `bimodal_flag`
  - `prefix_holdout_error`
  - `prefix_holdout_abs_error`
  - `calibration_residual_scale`
- 集約 shift curve、distance bucket 別 error correlation を生成物として追加。
- `second_delta` は second-best の絶対 shift とし、best との差分は `second_delta_vs_best` に分離。
- `config.yaml` と notebook / inference の識別子を `exp216` に統一。

### 再現性メモ

- `cfg.experiment.status`: `completed_train_side_rejected_no_submit`
- 乱数依存は deterministic なインデックス抽出に限定（`max_eval_rows_per_region_per_well`）
- upstream exp072 cache は固定 readout 固定参照のみに使用
- gzip 生成物は raw 圧縮 SHA と decompressed SHA を記録

### Kaggle 運用ノート

- 現在はローカルでコードの一貫化が完了。
- 実行結果の取得前の状態。
- Kaggle train 実行予定:
  - active variant 数: 0
  - model/config 数: 0
  - fold 数: 0
  - 合計 booster 数: 0
  - GPU: false
  - control / parent 再学習: なし
  - fixed exp072 cache: observation-cost readout のみ
- kernel id/title は slug 制約を避けるため短縮した同一実験名系にする。
- 参考コマンド（運用ルート）:

```bash
make prepare-kaggle-notebooks EXP=exp216_affine_shift_landscape_ruler_readout \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp216-affine-ruler-readout-train --title 'exp216 affine ruler readout train' --run-on-push --strict"
```

### 次アクション

- Kaggle train を実行して実アーティファクトを取得。
- `affine_shift_landscape_ruler_readout` の row_context / gain / shift curve / error correlation 表を確認し、zero-shift順位と二峰性が悪化する場合は即却下。

### Kaggle train v1 push

```bash
make prepare-kaggle-notebooks EXP=exp216_affine_shift_landscape_ruler_readout \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp216-affine-ruler-readout-train --title 'exp216 affine ruler readout train' --run-on-push --strict"
make push-kaggle-train EXP=exp216_affine_shift_landscape_ruler_readout
```

- result: Kernel version 1 pushed successfully.
- kernel: `kentookumura/exp216-affine-ruler-readout-train`
- URL: https://www.kaggle.com/code/kentookumura/exp216-affine-ruler-readout-train

### Kaggle train v1 result

- status: COMPLETE
- kernel: `kentookumura/exp216-affine-ruler-readout-train` v1
- rows: 3,561,984 row-context rows
- wells: 773
- runtime: 1264.923 sec
- output: aggregate CSV / summary を取得し、`artifacts/` に同期した。大きい `row_context.csv.gz` はローカル取得していない。

取得コマンド:

```bash
kaggle kernels output kentookumura/exp216-affine-ruler-readout-train \
  -p /tmp/exp216_affine_ruler_output --force \
  --file-pattern '.*(summary\.json|surface_metrics\.csv|bucket_metrics\.csv|well_metrics\.csv|gain_vs_raw\.csv|shift_curve_metrics\.csv|error_correlation_metrics\.csv|pfbeam_candidate_metrics\.csv|pfbeam_observation_metrics\.csv|well_input_summary\.csv|metrics\.json)$'
```

主要結果:

- best all: `savgol_31_p2__raw`, RMSE 108.534313, MAE 69.576973
- hidden_tail best: `rolling_median_11__raw`, RMSE 125.707127, MAE 76.419067
- prefix_backtest best: `savgol_31_p2__raw`, RMSE 87.718421, MAE 62.469505
- `raw__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.056464
- `rolling_median_11__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.075805
- `savgol_31_p2__heel_calibrated` hidden_tail mean abs-error gain vs raw: -2.783559
- `savgol_31_p2__raw` prefix_backtest mean abs-error gain vs raw: +0.661982
- `rolling_median_11__raw` hidden_tail mean abs-error gain vs raw: +0.207899
- `likpf_mean` hidden_tail RMSE: 11.471434
- `raw__raw` `likpf_mean` observation rank: mean 18.163254 / top1 0.052105 / top5 0.233162
- `raw__heel_calibrated` `likpf_mean` observation rank: mean 19.346994 / top1 0.045445 / top5 0.215344

判断:

- affine / heel calibration は train-side localization と PF/Beam observation rank を改善しない。
- raw smoothing は surface sharpness と prefix_backtest で小改善するが、direct candidate replacement の根拠には弱い。
- `zero_rank` / entropy / bimodal signal は uncertainty / fallback 診断材料に限定する。
- inference port / submit はしない。

検証:

- `.venv/bin/python -m py_compile ...` 通過
- `.venv/bin/ruff check ...` 通過
- `make validate-exp EXP=exp216_affine_shift_landscape_ruler_readout` 通過
