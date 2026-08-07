# exp169_tvt_input_pfbeam_offset_calibration 結果

## 状態

Kaggle train v1 完了。train-side diagnostic として完了。direct correction は不採用。

## 仮説

known prefix の `TVT_input` に対する PF/Beam candidate offset が tail でも保たれるなら、candidate-specific な capped offset correction で `likpf_mean` や Beam/PF candidate を改善できる可能性がある。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 検証: train well prefix holdout offset backtest
- 主比較: `likpf_mean`
- base candidates: `likpf_mean`, `beam_mean`, `pf_ancc`, `sc_ens`, `hyb`
- prefix holdout rows: 256
- min known prefix rows: 80
- min calibration rows: 32

## 結果

- 実行: `kentookumura/exp169-tvt-input-pfbeam-offset-calibration-train` v1
- 行数 / well 数: 3,783,989 rows / 773 wells
- prefix replay: 773 wells ok、197,888 prefix rows、5,411 offset rows
- variant 数: 96
- runtime: 26,208.919 sec
- baseline `likpf_mean`: RMSE 11.594897672、MAE 7.067632584、within10 0.772807479
- best offset variant: `off_likpf_mean_self_median_a0p5_c10_g50_f250_iqr20_n32_const`
- best offset RMSE: 11.580455166
- delta vs `likpf_mean`: -0.014442507 RMSE
- best offset MAE: 7.097507839
- best offset within10: 0.772440935
- max well regression vs `likpf_mean`: +4.173820317 RMSE

global RMSE は小幅に改善したが、MAE と within10 は悪化した。`1000_plus` は 12.704015 から 12.690640 へ改善した一方、MAE は 7.999678 から 8.033676 へ悪化した。near `000_050` は fade guard により no-op。

representative wells では `fef8af96` と `ba48188d` は小改善したが、`91b301ce`、`1b1eba53`、`86454a6f` は悪化した。worst well `4e050c92` は +4.173820 RMSE regression で、direct correction の guard 条件を満たさない。

prefix offset 自体は安定しており、median prefix RMSE は `pf_ancc` 1.047701、`likpf_mean` 1.574280、`beam_mean` 1.862985。`pfbeam_median` offset も IQR median 1.125244 と低い。ただし、この prefix offset は tail 全体へ直接転写すると MAE / within10 / worst-well を壊す。

## 再現性

- deterministic anchor: false
- seed policy: public replay helper の stable per-well seed
- model SHA / manifest SHA: model なし
- submission SHA: submission なし
- candidate metrics SHA: `b47c564ade4b8ad52ccc077709faadad7506bb4e1ee55b0de517941bb1abbfef`
- prefix offsets SHA: `784833c4fcd699562b82a67c7ec264cd1efd2b036d196064377d9d369c1f3c59`
- OOF gzip raw SHA: `91c11c2a58c53110081d8919d284e5f6d5aa92b4fc6d452850ba6af57769bddf`
- OOF gzip decompressed SHA: `c93e152e72c7224a69aee53fb4744e0471d71d3853375ea7d29a169165145219`

## 次

inference port / submit は行わない。offset を TVT 候補へ直接補正するのではなく、prefix offset median / IQR / prefix RMSE / candidate disagreement を exp148 系の confidence feature として使う低優先候補に下げる。

## 全区間 PF/Beam 可視化

known `TVT_input` 区間でも PF/Beam を生成した結果を確認するため、exp083 系 EDA を拡張した guard notebook を Kaggle で実行した。

- 実行: `kentookumura/exp169-all-interval-pfbeam-visualization-guard` v1
- status: `KernelWorkerStatus.COMPLETE`
- runtime: 720.004 sec
- output: `kaggle/output/all_interval_viz_v1`
- plot count: 12
- HTML index: `kaggle/output/all_interval_viz_v1/artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_index.html`
- manifest: `kaggle/output/all_interval_viz_v1/artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_manifest.csv`

対象は 6 well x 2 mode。`exp169_holdout_tail` は exp169 と同じ prefix holdout 条件、`full_known_backtest` は early known prefix だけを残して既知区間の大部分まで PF/Beam を通した条件である。代表画像を確認し、TVT path、candidate error、GR/Z context が全区間で描画されている。
