# exp218_gr_wavelet_rotation_confidence_features_on_exp148

## 概要

exp148 の learned-likelihood ML anchor に、target-free な GR wavelet / FFT rotation-denoise confidence feature を add-only で追加する実験。

## Route

- route: `ml_model`
- status: `ml_route_anchor_public_lb_7p843`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- variant: `gr_wavelet_rotation_confidence_addonly`

## 状態

Kaggle train v1 は `kentookumura/exp218-gr-wavelet-rotation-exp148-train` で COMPLETE。Kaggle inference v1 も `kentookumura/exp218-gr-wavelet-rotation-exp148-inference` で COMPLETE。提出 ref `54457577` は Public LB 7.843 で、現行の ML route submitted anchor。

## 結果

- rows / wells / features: 3,783,989 / 773 / 380
- GRWR generated features: 86
- boosters: 15
- `lgb_mean` RMSE: 8.475793752
- delta vs exp148 GPU `lgb_mean` 8.501281182: -0.025487430
- delta vs exp160 `lgb_mean` 8.463718774: +0.012074978
- feature join coverage: pass、drop rows 0、drop wells 0
- `lgb_mean` prediction SHA: `6ad4f96c8ed3cb10c301fd17bf02bf2da6363ddc6e0f96649c9738a09c3b11cf`

## Inference

- kernel: `kentookumura/exp218-gr-wavelet-rotation-exp148-inference`
- version: 1
- output: `/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1`
- selected model: `gr_wavelet_rotation_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`
- model count: 15
- test rows / submission rows / fallback rows: 14,151 / 14,151 / 0
- prediction range: 11590.199219 - 12239.539062
- prediction mean / std: 11905.245072 / 278.608569
- prediction SHA: `483845c8969e99e8d12c9dfcbe43bb8dfc727a1df8905ef045f02e35ebdcbff1`
- submission SHA: `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`
- submit-check: pass。行数、header、ID順、重複なし、欠損なしを確認。

## Submission

- version: `v059`
- ref: `54457577`
- submitted at: 2026-07-08 09:47:59.040000 UTC / 2026-07-08 18:47:59.040000 JST
- status: `SubmissionStatus.COMPLETE`
- Public LB: 7.843
- delta vs exp148 CPU runtime Public LB 7.921: -0.078
- delta vs exp148 GPU inference v7 Public LB 7.960: -0.117
- delta vs exp198 Public LB 7.930: -0.087
- overall ensemble anchor exp082 Public LB 7.601 には +0.242 届かない。

## 仮説

GR wavelet detail energy、FFT rotation-band energy、raw-vs-denoised local agreement、candidate observation-cost entropy は、exp148 の learned likelihood confidence とは別系統の target-free 不確実性信号になりうる。

## 追加特徴

- DWT db4 approximation/detail residual energy
- FFT dominant / rotation-band / high-frequency energy ratio
- raw-vs-rolling / raw-vs-SG / raw-vs-DWT local agreement
- existing candidate の GR observation-cost entropy / rank / zero-candidate rank proxy
- md_since、candidate spread、learned likelihood entropy との interaction

## 検証方針

GroupKFold by well、5 folds、LightGBM 3 configs の `lgb_mean` OOF を exp148 の保存済み CV / Public LB と比較する。worst-well、near-row、1000+ longtail、feature importance を確認し、global RMSE だけでは採用しない。train-side positive の場合も、raw-test parity と inference artifact 検証なしに submit しない。

## 所見

train-side は exp148 GPU CV に対して positive。`grwr_fft_rotation_ratio_x_log1p_md_since` は全体 feature importance 4 位に入り、GRWR block は実際に使われた。Inference artifact と `submission.csv` は生成済みで submit-check も pass。提出後の Public LB 7.843 は exp148 CPU runtime 7.921 を -0.078 上回ったため、exp218 を ML route submitted anchor に更新する。一方、100-1000 tail bucket と一部 worst wells の OOF 悪化は残り、overall では exp082 ensemble Public LB 7.601 が引き続き最良。

## 注意

DWT/FFT/denoised GR や candidate observation cost は、予測値の直接置換、blend、postprocess、hard gate には使わない。Inference v1 と提出は saved LightGBM booster inference のみ。
