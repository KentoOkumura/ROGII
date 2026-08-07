# exp217_grcal_public_raw_pf_confidence_features_on_exp158 結果

## 現在の状態

Kaggle train v3 は `KernelWorkerStatus.COMPLETE`。`pubraw_` 生成を cache 化したことで、selector train は完走した。

Kaggle kernel: `kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`  
URL: https://www.kaggle.com/code/kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train

v3 best は `viterbi_sw050_bias000_jw050_jf025_d0075_std999999_md0000_seg012`。RMSE は `10.669620824`、MAE は `6.453232089`、within 10ft は `0.788548804`、oracle label accuracy は `0.284127940`。path switch は `14,599`、`3.858098 / 1000 rows`。

exp158 continuity RMSE `10.789163253` からは `-0.119542429` 改善。best OOF model の `lgb_candidate_error_ranker` RMSE `10.695686205` からも Viterbi で `-0.026065381` 改善した。一方で exp184 heatmap add-only `10.560650325` には `+0.108970499`、exp191 typewell continuity `10.598006880` には `+0.071613944` 悪い。

判断: `pubraw_` confidence features は exp158 selector への add-only signal としては有効。ただし現行 PF/Beam route の強い reference を更新しないため、inference / submit へは進めない。

## Cache

2026-07-13 に `pubraw_` 生成を cache stage に分離した。cache notebook は `kentookumura/exp217-pubraw-cache-v1` / `exp217 pubraw cache v1` として CPU・internet off・run-on-push true で実行し、`KernelWorkerStatus.COMPLETE`。生成物は `exp217_grcal_public_raw_pf_confidence_features_on_exp158_pubraw_features.csv.gz`。通常 train はこの cache kernel を input にして `pubraw_` 再生成を skip する構成に変更済み。

cache v1 は 3,783,989 rows / 773 wells / 25 `pubraw_` features を生成。runtime は 23,098.604 秒。`pubraw_features` SHA256 は `63ea14c78f980f1c18060923585797da86e73d3700a270afd340ea3a8be2381d`、decompressed SHA256 は `1c7c8717740d696bf28be7bf78e8fac9f33957957886dafff86c438a7e030e7d`。

## 過去実行

Kaggle train v1 の status は `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。logs は `pubraw` feature generation 完了後、fold 0/1 の 6 モデル保存まで進み、fold 2 の学習開始直後で止まっている。最終 summary / metrics / OOF / Viterbi 結果は生成されていないため、v1 は CV / LB / 採用判断に使わない。

v2 は version 2 として `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, `--accelerator NvidiaTeslaT4` で push したが、status は `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。ログでは `GPU enabled: True` の後、`[pubraw] 1/773` までしか進んでいない。`pubraw` 生成は CPU/Numba のままなので、T4 runtime は notebook hardware の切り替えであり、この stage を GPU-native にする変更ではなかった。

v3 実行内容は active selector variant 1、LightGBM 3 configs x 5 folds = 15 boosters、control / parent retraining なし。`pubraw` 生成は train 内では行わず、`kentookumura/exp217-pubraw-cache-v1` から cache 読み込み。v3 runtime は `28,635.531` 秒。

## Readout

- distance bucket:
  - `000_050`: RMSE `0.622860`
  - `050_100`: RMSE `1.368190`
  - `100_250`: RMSE `2.570571`
  - `250_500`: RMSE `4.278296`
  - `500_1000`: RMSE `6.140488`
  - `1000_plus`: RMSE `11.693821`
- worst wells:
  - `86454a6f`: RMSE `57.240032`
  - `1b1eba53`: RMSE `56.474652`
  - `5f4d2a52`: RMSE `44.241381`
  - `fef8af96`: RMSE `38.689825`
  - `91db7070`: RMSE `35.966960`
- `pubraw_` feature importance:
  - `lgb_candidate_error_ranker`: `pubraw_gr_sigma` rank 4、`pubraw_seed_mean_minus_candidate` rank 6、`pubraw_seed_mean_minus_scale5` rank 7
  - `lgb_multiclass`: `pubraw_gr_sigma` rank 3、`pubraw_seed_weight_max` rank 6、`pubraw_seed_weight_entropy` rank 8
  - `lgb_candidate_binary`: `pubraw_gr_sigma` rank 5、`pubraw_seed_mean_minus_candidate` rank 6

## 注意

`pubraw_pf_scale5/12` は selector feature であり、selectable candidate、direct replacement、blend、postprocess、PF weight replacement、inference、submit には使わない。

## 次の方針

exp217 は train-side positive vs exp158 だが、PF/Beam route anchor は更新しない。追加の inference port / submit は行わない。`pubraw_` 系は selector confidence feature として候補に残せるが、次に使う場合は exp184/exp191 以上の feature surface へ add-only するか、high spread / high likpf gap bucket の限定 guard として検証する。
