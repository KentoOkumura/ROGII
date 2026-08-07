# exp169_tvt_input_pfbeam_offset_calibration セッションノート

## 目的

`tvt_input_pfbeam_offset_calibration` を実装する。known prefix の `TVT_input` 末尾を holdout として PF/Beam candidate を再生成し、candidate と `TVT_input` の well 固有 offset を推定して、exp072 fixed tail candidate に posthoc 補正として使えるかを監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_diagnostic_no_submit
- CV: 11.594897672217703 (`likpf_mean` baseline)
- LB: なし
- 提出: なし

## コマンドログ

### 2026-07-02 JST 実装

```bash
uv run python scripts/new_steering.py --experiment exp169_tvt_input_pfbeam_offset_calibration
uv run python scripts/new_experiment.py --name exp169_tvt_input_pfbeam_offset_calibration --source experiments/exp140_z_slope_posthoc_correction_on_pfbeam_candidates
```

実装内容:

- `.steering/20260702-exp169-tvt-input-pfbeam-offset-calibration/` を作成し、requirements / design / tasklist を記入した。
- exp140 を親に実験ディレクトリを作成し、exp169 用に config / settings / notebooks を更新した。
- exp072 の `public_notebook_replay_audit.py` をコピーし、prefix holdout replay に再利用する。
- `tvt_input_pfbeam_offset_calibration.py` を実装した。
- train notebook は設定確認、入力確認、prefix replay、offset summary、bucket/group metrics 表示の構成にした。
- inference notebook は diagnostic no-op として、submission を生成しない。

実行予定:

- active variant 数: 3 base candidates x 2 offset sources x 1 estimator x 2 alphas x 2 clip x 1 near guard x 1 fade end x 2 max IQR x 1 min rows x 2 slope modes = 96 variants
- model/config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- 親実験/control 再学習: なし
- GPU: なし
- PF/Beam replay: known prefix holdout rows のみ。`pf_seeds=32`、`pf_particles=300`。

## 生成予定ファイル

- `exp169_tvt_input_pfbeam_offset_calibration_candidate_metrics.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_bucket_metrics.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_by_well.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_group_metrics.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_prefix_offsets.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_prefix_status.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_oof_predictions.csv.gz`
- `exp169_tvt_input_pfbeam_offset_calibration_feature_schema.csv`
- `exp169_tvt_input_pfbeam_offset_calibration_summary.json`

## 次

1. Jupytext 変換、構文チェック、ruff、validate-exp を通す。
2. Kaggle train package を作成する。
3. Kaggle train を実行し、結果に基づいて `result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`、`experiment_summary.md` を更新する。

### 2026-07-02 JST Kaggle train v1 push / running

```bash
kaggle kernels push -p experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/train
kaggle kernels pull kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train -p /tmp/kaggle-pull/exp169-tvt-input-pfbeam-offset-calibration-train -m
kaggle kernels logs kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
kaggle kernels status kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
- metadata pull 成功。
- `status`: `KernelWorkerStatus.RUNNING`
- CLI logs は実行中空。ユーザー指示によりローカル監視だけ停止した。Kaggle kernel 自体は停止していない。

### 2026-07-03 JST Kaggle train v1 complete / output retrieved

```bash
kaggle kernels status kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
kaggle kernels logs kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train
kaggle kernels output kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train -p experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/output/train_v1
kaggle kernels output kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train -p experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/output/train_v1 --file-pattern '.*(summary|prefix).*' -o
```

結果:

- `status`: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/output/train_v1`
- full output 取得は OOF gzip で長時間停止したため中断し、candidate / bucket / by-well / group / schema を取得した。
- 追加で `--file-pattern '.*(summary|prefix).*'` により summary / prefix offsets / prefix status を取得した。
- Kaggle 側 summary には OOF gzip SHA が記録されている。ローカルの不完全 0-byte OOF は削除した。
- rows / wells: 3,783,989 / 773
- prefix replay: 773 wells ok、197,888 prefix rows、5,411 offset rows
- runtime: 26,208.918837 sec
- variant count: 96
- primary baseline `likpf_mean`: RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479
- best offset variant: `off_likpf_mean_self_median_a0p5_c10_g50_f250_iqr20_n32_const`
- best offset: RMSE 11.580455166 / MAE 7.097507839 / within10 0.772440935
- delta vs baseline: -0.014442507 RMSE
- max well regression vs baseline: +4.173820317 RMSE

解釈:

- Global RMSE は小幅改善したが、MAE と within10 が悪化し、worst-well regression も大きい。
- `1000_plus` は RMSE 12.704015 -> 12.690640 へ改善したが、MAE は 7.999678 -> 8.033676 へ悪化した。
- prefix offset 自体は安定しており、median prefix RMSE は `pf_ancc` 1.047701、`likpf_mean` 1.574280、`beam_mean` 1.862985。
- direct correction / inference port / submit はしない。
- offset diagnostics は exp148 系 confidence feature の低優先候補に下げる。

### 2026-07-03 JST all-interval PF/Beam visualization guard complete

```bash
kaggle kernels status kentookumura/exp169-all-interval-pfbeam-visualization-guard
kaggle kernels logs kentookumura/exp169-all-interval-pfbeam-visualization-guard
kaggle kernels output kentookumura/exp169-all-interval-pfbeam-visualization-guard -p experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/output/all_interval_viz_v1
```

結果:

- `status`: `KernelWorkerStatus.COMPLETE`
- URL: https://www.kaggle.com/code/kentookumura/exp169-all-interval-pfbeam-visualization-guard
- output: `experiments/exp169_tvt_input_pfbeam_offset_calibration/kaggle/output/all_interval_viz_v1`
- runtime: 720.004 sec
- 対象 well: `91b301ce`, `ba48188d`, `fef8af96`, `1b1eba53`, `86454a6f`, `4e050c92`
- plot modes: `exp169_holdout_tail`, `full_known_backtest`
- plot count: 12
- PNG: `artifacts/all_interval_pfbeam_plots/*.png`
- HTML index: `artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_index.html`
- manifest: `artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_manifest.csv`
- summary: `artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_summary.json`
- manifest SHA: `3a5ceb801cca2a4470cd4c5cfb458ef7ae40babea17a0ab693f83f0389af52c8`
- HTML index SHA: `27fd6b56b17eb3b7cdfdc5fec341165ab27a725547b8ef3ef2649399ec10105b`

確認:

- `file artifacts/all_interval_pfbeam_plots/*.png` で全 12 枚が `2100 x 1470` PNG と確認できた。
- manifest はヘッダ込み 13 行で、HTML index は全 12 PNG を相対パス参照している。
- 代表画像 `full_known_backtest__91b301ce.png` と `exp169_holdout_tail__fef8af96.png` を目視し、上段 TVT/PF/Beam、中段 candidate error、下段 GR/Z が描画されていることを確認した。
