# exp160_sp45_bimodal_selector_confidence_features_on_exp148 結果

## 状態

Kaggle train v2 / inference v1 / scoring 完了。train-side OOF は exp148 baseline から小幅改善し、inference v1 の `submission.csv` は submit-check を通過した。Public LB は 8.061。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Route: `ml_model`
- variant: `sp45_bimodal_selector_confidence_addonly`
- baseline: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 学習: 3 LightGBM configs x 5 folds = 15 boosters
- control 再学習: なし

## 結果

### Train

- Kaggle kernel: `kentookumura/exp160-sp45-bimodal-exp148-train` version 2
- status: `COMPLETE`
- rows / wells: 3,783,989 / 773
- features: 372
- feature join coverage: pass、dropped rows 0、dropped wells 0
- elapsed: 14,573.844 sec

| model | pooled RMSE |
| --- | ---: |
| lgb0 | 8.582750400 |
| lgb1 | 8.458535254 |
| lgb2 | 8.502983731 |
| lgb_mean | 8.463718774 |

exp148 `lgb_mean` 8.501281182 から、exp160 `lgb_mean` は -0.037562408 改善した。

### Inference

- Kaggle kernel: `kentookumura/exp160-sp45-bimodal-exp148-inference` version 1
- status: `COMPLETE`
- output: `/tmp/kaggle-output/exp160_sp45_bimodal_selector_confidence_features_on_exp148/inference_v1/`
- selected: `sp45_bimodal_selector_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`
- models / features: 15 boosters / 372 features
- test rows / submission rows: 14,151 / 14,151
- fallback rows: 0
- elapsed: 141.319 sec、feature generation 102.769 sec
- prediction min / max / mean / std: 11590.324219 / 12240.247070 / 11905.439880 / 278.695602
- prediction sha256: `cb85e56ed032e3f5c0577c1928272e9f1621da9e9b356932caac20fc0d1c03d2`
- submission sha256: `366543ab052b98afec8c61f020c6eccc84c751fd734262dd9913bbb53fab354b`
- submit-check: PASS。重複 ID なし、empty/NaN/Inf-like なし、header と row count は `sample_submission.csv` と一致。

### Submission

ユーザー確認により exp160 の Public LB は ref `54183128` の 8.061 と確定した。

| ref | submitted | Public LB | delta vs exp148 7.960 | memo |
| ---: | --- | ---: | ---: | --- |
| 54183128 | 2026-06-29 23:36:23.280000 | 8.061 | +0.101 | `monitor_submission.py --once` で記録。 |

## 判定

SP45 / bimodal selector confidence features は train-side では positive。直接置換や hard gate ではなく add-only ML feature として効いている。

Train-side CV は exp148 から改善したが、Public LB は exp148 7.960 から 8.061 へ +0.101 悪化した。exp160 は提出候補として採用しない。
