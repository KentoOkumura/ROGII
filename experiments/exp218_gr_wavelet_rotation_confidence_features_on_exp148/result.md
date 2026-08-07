# exp218_gr_wavelet_rotation_confidence_features_on_exp148 結果

## 状態

Kaggle train v1 完了。train-side は exp148 GPU CV に対して positive。Kaggle inference v1 も完了し、`submission.csv` は submit-check pass。提出 ref `54457577` は Public LB 7.843 で完了し、exp148 CPU runtime Public LB 7.921 を上回ったため ML route submitted anchor に更新する。

## 仮説

GR wavelet detail energy、FFT rotation-band energy、raw-vs-denoised local agreement、candidate observation-cost entropy は、exp148 の learned likelihood confidence とは別系統の target-free 不確実性信号になる可能性がある。

## 評価設計

- `gr_wavelet_rotation_confidence_addonly`: exp148 の feature surface に `gr_wavelet_rotation_confidence` feature group を追加する。
- `exp148_fulltrain_control`: 再学習しない。保存済み exp148 metrics を historical baseline として参照する。
- GroupKFold 5 folds、well group、metric は RMSE。
- GPU runtime、3 LightGBM configs、5 folds、15 boosters。

## Kaggle 実行

### Train v1

- kernel: `kentookumura/exp218-gr-wavelet-rotation-exp148-train`
- version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp218-gr-wavelet-rotation-exp148-train
- status: `COMPLETE`
- output: `/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/train_v1`
- elapsed: 14335.658 sec

### Inference v1

- kernel: `kentookumura/exp218-gr-wavelet-rotation-exp148-inference`
- version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp218-gr-wavelet-rotation-exp148-inference
- status: `COMPLETE`
- output: `/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1`
- elapsed: 108.982 sec

## CV

| model | RMSE TVT | delta vs exp148 same model |
| --- | ---: | ---: |
| lgb0 | 8.557165712 | -0.042620148 |
| lgb1 | 8.512227651 | -0.051743470 |
| lgb2 | 8.524447601 | +0.014627882 |
| lgb_mean | 8.475793752 | -0.025487430 |

exp148 GPU `lgb_mean` 8.501281182 からは改善した。一方、exp160 `lgb_mean` 8.463718774 よりは +0.012074978 弱い。exp160 は train-side positive でも Public LB 8.061 で exp148 より悪化したが、exp218 は提出後 Public LB 7.843 で exp148 CPU runtime 7.921 を上回った。

## Coverage / artifacts

- rows: 3,783,989
- wells: 773
- features: 380
- GRWR generated features: 86
- feature groups: `projection_correction,u_disagreement,learned_likelihood_confidence,gr_wavelet_rotation_confidence`
- feature join coverage: pass、drop rows 0、drop wells 0
- prediction SHA (`lgb_mean`): `6ad4f96c8ed3cb10c301fd17bf02bf2da6363ddc6e0f96649c9738a09c3b11cf`
- model manifest SHA: `904570def0d6ad0140f3df95c8bb38f31823295fd191206290e3833b5b2cc237`
- feature schema SHA: `aaf5f13f1e7c5236cd332dcebfdbf98e9c08247465833232e79ce3ff56362b49`

## Inference / submission artifact

- selected: `gr_wavelet_rotation_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`
- model count: 15
- feature count: 380
- test rows: 14,151
- submission rows: 14,151
- predicted rows: 14,151
- fallback rows: 0
- prediction min / max: 11590.199219 / 12239.539062
- prediction mean / std: 11905.245072 / 278.608569
- prediction SHA: `483845c8969e99e8d12c9dfcbe43bb8dfc727a1df8905ef045f02e35ebdcbff1`
- submission SHA: `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`
- summary SHA: `4c6cecf940b5793efe3b496d2e668f8d2116fea5299d7663e90d4939da3368c0`
- feature schema SHA: `e70f65f05b865cc77f426b1671a01714e597a1795c857eef4b7c17323e1344d6`
- raw-test learned likelihood decompressed SHA: `8d1146ac1e68da67a2c8d2d00788c1593fc99654b949e0a5ac065cf781344e13`
- raw-test long likelihood decompressed SHA: `92ae5e9328073ac2727fa18dc2e03025a557e9646b8b9be57fd051d1ae86c612`
- anchor T0 vs `last_known_tvt` max abs diff: 0.0
- known prefix rows min / max: 1442 / 2083
- submit-check: pass。`sample_submission.csv` と header / row count が一致し、ID順一致、重複 ID 0、欠損/Inf 0。

## Submission

- version: `v059`
- ref: `54457577`
- submitted at: 2026-07-08 09:47:59.040000 UTC / 2026-07-08 18:47:59.040000 JST
- status: `SubmissionStatus.COMPLETE`
- Public LB: 7.843
- delta vs exp148 CPU runtime Public LB 7.921: -0.078
- delta vs exp148 GPU inference v7 Public LB 7.960: -0.117
- delta vs exp198 Public LB 7.930: -0.087
- delta vs exp160 Public LB 8.061: -0.218
- delta vs exp082 ensemble Public LB 7.601: +0.242

exp218 は exp148 CPU runtime anchor を上回ったため ML route submitted anchor として採用する。overall では exp082 ensemble Public LB 7.601 が引き続き最良。

## Bucket / by-well

exp148 OOF viewer と exp218 OOF を id で streaming join して同一行比較した。

| bucket | rows | exp218 RMSE | exp148 RMSE | delta |
| --- | ---: | ---: | ---: | ---: |
| 000_050 | 38,650 | 0.957634 | 0.978726 | -0.021092 |
| 050_100 | 38,650 | 1.310175 | 1.316981 | -0.006806 |
| 100_250 | 115,950 | 2.094431 | 2.084639 | +0.009791 |
| 250_500 | 193,157 | 3.315459 | 3.298294 | +0.017165 |
| 500_1000 | 385,911 | 4.800747 | 4.792035 | +0.008711 |
| 1000_plus | 3,011,671 | 9.295198 | 9.325405 | -0.030207 |

by-well は 413 wells 改善、360 wells 悪化。median delta は -0.026294 で全体としては改善側だが、最大悪化 `f88ddb26` は +4.075520 RMSE。worst wells では `86454a6f` -1.020068、`1b1eba53` -0.553343、`81bf5923` -1.412768 は改善した一方、`fb03ae90` +0.982819、`efe96181` +1.434352、`708caea9` +1.090246 は悪化した。

## Feature importance

GRWR block はモデルに使われた。`grwr_fft_rotation_ratio_x_log1p_md_since` は mean importance 3654.4 で全体 4 位。GRWR features は 86 列中 85 列で mean importance > 0。上位には `grwr_gr_missing_rate`、`grwr_fft_dominant_frequency_norm`、`grwr_fft_high_frequency_ratio`、`grwr_fft_dominant_energy_ratio`、`grwr_fft_rotation_energy_ratio` が入った。

## 判断

train-side の仮説は支持。特に 1000+ long-tail では exp148 より改善し、GRWR feature の重要度も十分に出た。Inference v1 は current-test GRWR feature generation と saved-booster inference が成立し、提出形式も pass。Public LB 7.843 は exp148 CPU runtime 7.921 を -0.078 上回り、exp193 7.946 / exp198 7.930 も上回ったため、exp218 は ML route submitted anchor として採用する。一方で、100-1000 bucket の小幅悪化、worst-well の一部悪化、overall best の exp082 ensemble 7.601 との差は残る。
