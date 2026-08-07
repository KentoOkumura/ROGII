# exp221_lgb_oof_gaussian_emission_hmm_on_exp148 結果

## 状態

Kaggle train v3 は完走。`sigma=20/lambda=0.50` の single HMM+LGB variant が train-side OOF で exp148 / exp193 の LGB mean を上回った。

Kaggle inference v1 も完了。hidden-safe 形で exp148 saved LightGBM boosters から current-test `lgb_mean` 予測を生成し、その予測を exp221 HMM の Gaussian emission center として使って `submission.csv` を作成した。

Code submission v1 は ref `54490473` で Public LB 7.953。exp148 GPU inference v7 7.960 は小さく上回ったが、exp193 7.946、exp148 CPU runtime 7.921、exp218 ML anchor 7.843、exp082 ensemble anchor 7.601 には届かないため採用しない。

## 仮説

exp148 / exp193 の LGB OOF 点予測を HMM の Gaussian emission として使うことで、点予測の強さを保ちながら HMM posterior の系列制約で worst-well / step-delta を改善できるか確認する。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- LGB source: exp148 `lgb_mean` OOF、比較 baseline として exp193 `lgb_mean` OOF
- Route: `ensemble`
- 実行 variant: `hmm_lgb_exp148_lgb_mean_s2000_l0500`
- 学習 / Booster: なし / 0
- GPU / internet: false / false
- Kernel: `kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train` v3
- Runtime: 17,827.454 sec、773 wells、3,783,989 rows

## Inference v1

- Kernel: `kentookumura/exp221-lgb-hmm-exp148-infer` v1
- Kaggle status: COMPLETE
- Metadata: GPU true、internet false、kernel sources は exp072 train / exp148 train / exp099 train / exp111 train / exp112 train
- exp148 proxy: `learned_likelihood_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`、15 boosters、294 features、fallback 0
- HMM candidate: `hmm_lgb_exp148_lgb_mean_s2000_l0500`
- Output: 14,151 rows / 3 wells、submission rows 14,151、fallback rows 0、extra prediction ids 0
- Submission validation: PASS
- Submission SHA256: `d90926bc87268285640863ddc3e24fbaa4d715c1b7394f7410a2d4f6d13b7cc3`
- Prediction range: min 11598.3203125、max 12235.1181641、mean 11904.820939、std 277.524177
- HMM posterior std: mean 0.696918、p90 1.243711
- Submission: ref `54490473`、Public LB 7.953、Private LB 未公開

見えている test は sample なので、この row/well count は形式確認であり score evidence ではない。重要な確認点は、current-test exp148 予測が notebook 内で生成され、`sample_submission.csv` の全 ID に strict に map され、fallback が 0 だったこと。

## 結果

Best candidate は `hmm_lgb_exp148_lgb_mean_s2000_l0500`。

| candidate | RMSE | MAE | within10 | delta RMSE vs exp148 | delta RMSE vs exp193 | delta RMSE vs exp072 likPF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HMM+LGB sigma20 lambda0.50 | 8.327736951 | 4.811969897 | 0.858810108 | -0.173554033 | -0.128939102 | -3.267160717 |
| exp193 lgb_mean | 8.456676053 | 5.318167527 | 0.857385421 | -0.044614931 | 0.000000000 | -3.138221615 |
| exp148 lgb_mean | 8.501290984 | 5.335657607 | 0.856319350 | 0.000000000 | +0.044614931 | -3.093606684 |
| exp072 likpf_mean | 11.594897668 | 7.067632583 | 0.772802194 | +3.093606684 | +3.138221615 | 0.000000000 |

Distance bucket は全 bucket で exp148 / exp193 比改善。`1000_plus` は RMSE 9.130481727 で exp148 比 -0.194179177、exp193 比 -0.142626253。near bucket も `000_050` -0.198198870 vs exp148、`050_100` -0.158874698 vs exp148。

exp115 hidden-like は両 subgroup で改善。`verification_like_spatial` は RMSE 9.572220447、exp148 比 -0.229604714、exp193 比 -0.114087659。`verification_like_typewell_purged` は RMSE 9.545366073、exp148 比 -0.231809550、exp193 比 -0.115029762。

By-well では全体として改善寄りだが、悪化 well は残る。exp148 比は 509 wells 改善 / 264 wells 悪化、median delta -0.311669714、最大悪化は `2e63d9de` +4.981191458 RMSE。exp193 比は 495 wells 改善 / 278 wells 悪化、最大悪化も `2e63d9de` +5.628247836 RMSE。

Step-delta は HMM smoothing が効いており、HMM+LGB の `abs_step_delta_mean` は 0.011034786、p99 は 0.065000000、`>5/10/25` rates はすべて 0。exp148 / exp193 の p99 2.013 / 1.986 よりかなり滑らか。

HMM std calibration は単調な error calibration ではない。最低 std bin RMSE 8.985650889、中央 bins 3-7 は 7.66-7.80、最高 std bin RMSE 9.997108771。posterior std はそのまま calibrated confidence としては使わない。

## 解釈

固定 sigma の LGB Gaussian emission を HMM に足す方向は train-side では支持される。exp148 / exp193 の LGB mean より overall、全 distance bucket、hidden-like subgroup、step-delta が改善しており、単純な point prediction replacement より sequence posterior として扱う利点が出た。

ただし well 単位の悪化は小さくない。特に `2e63d9de` は exp148 / exp193 比で最大悪化しており、raw-test inference 化する場合は by-well regression の guard と visualization を確認してから進める。

## 次

Public LB は CV 改善ほど伸びず、既存 anchor には届かなかったため採用しない。次に進めるなら、fixed sigma の点予測 emission ではなく、quantile band / uncertainty-calibrated sigma で hidden-side の悪化を抑えられるかを別実験で見る。
