# exp169_tvt_input_pfbeam_offset_calibration

## 状態

- ルート: pf_beam
- 状態: completed_train_side_diagnostic_no_submit
- CV: 11.594897672217703 (`likpf_mean` baseline)
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-02
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

PF/Beam / `likpf_mean` 候補は well によって真の TVT からほぼ定数 offset している可能性がある。hidden test でも observed prefix の `TVT_input` は使えるため、known prefix 上で同じ candidate 生成を backtest し、`candidate_tvt - TVT_input` の robust offset を測れば tail candidate の補正や confidence diagnostic に使えるかもしれない。

## 変更点

- known prefix 末尾を一時的に NaN にして PF/Beam replay を行う。
- prefix holdout 上で candidate 別 offset median / Huber / IQR / slope / prefix RMSE を計算する。
- exp072 fixed tail candidate cache に対して capped / fade-in offset correction を grid 比較する。
- true TVT は tail candidate RMSE / bucket / by-well metrics の scoring にだけ使う。
- supervised model、inference port、提出は行わない。

## 検証方針

- 検証面: exp072 train well pseudo-tail candidate cache
- 主比較: `likpf_mean`
- 追加確認: near `000_050`、`1000_plus`、candidate disagreement top quartile、representative wells、最大 well regression

## 実行入口

- 学習 notebook: `exp169_tvt_input_pfbeam_offset_calibration_train.ipynb`
- 推論 notebook: `exp169_tvt_input_pfbeam_offset_calibration_inference.ipynb`
- 可視化 guard notebook: `exp169_tvt_input_pfbeam_offset_calibration_guard.ipynb`

## 結果

Kaggle train v1 を完了した。best offset variant `off_likpf_mean_self_median_a0p5_c10_g50_f250_iqr20_n32_const` は RMSE 11.580455166 で `likpf_mean` 11.594897672 から -0.014442507 改善した。

ただし MAE は 7.067632584 から 7.097507839 へ悪化し、within10 も 0.772807479 から 0.772440935 へ悪化した。max well regression も +4.173820317 と大きいため、direct correction / inference port / submit はしない。

## 所見

prefix offset 自体は well ごとに安定しているが、tail へ直接転写すると一部 well の片側 bias を増やす。今後使う場合は TVT 候補補正ではなく、exp148 系 ML の confidence / risk feature に限定する。

## 全区間 PF/Beam 可視化

exp083 系 EDA を拡張し、known `TVT_input` 区間にも PF/Beam を再生成した結果を全区間で確認する guard notebook を実行した。

- 実行: `kentookumura/exp169-all-interval-pfbeam-visualization-guard` v1
- output: `kaggle/output/all_interval_viz_v1`
- plot count: 12
- 対象 well: `91b301ce`, `ba48188d`, `fef8af96`, `1b1eba53`, `86454a6f`, `4e050c92`
- modes: `exp169_holdout_tail`, `full_known_backtest`
- HTML index: `kaggle/output/all_interval_viz_v1/artifacts/exp169_tvt_input_pfbeam_offset_calibration_all_interval_plot_index.html`
- PNG dir: `kaggle/output/all_interval_viz_v1/artifacts/all_interval_pfbeam_plots/`

`exp169_holdout_tail` は exp169 と同じ known prefix 末尾 256 rows + original tail の replay、`full_known_backtest` は early known prefix だけを anchor にして、残りの known 区間と original tail をまとめて replay する図である。
