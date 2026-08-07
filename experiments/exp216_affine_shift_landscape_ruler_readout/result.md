# exp216_affine_shift_landscape_ruler_readout 結果

## 仮説

prefix-only affine fit で作る GR shift landscape の ruler readout（best/second/zero/secondary-mode/scale）を収集し、PF/Beam candidate の採否判断や fallback 条件に使えるかを診断する。

## 設定

- 親/参照: `affine_shift_landscape_ruler_readout` backlog、`exp167_fft_denoised_gr_matching_audit`、`exp072_exp063_full_replay_feature_cache`
- route: `pf_beam`
- Kaggle kernel: `kentookumura/exp216-affine-ruler-readout-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp216-affine-ruler-readout-train
- status: COMPLETE
- rows: 3,561,984 row-context rows
- wells: 773 train wells
- runtime: 1264.923 sec
- GPU: false
- PF/Beam 再生成: なし
- ML 学習: なし
- inference / submit: なし

## 結果

overall best は `savgol_31_p2__raw` で、raw typewell calibration の smoothing 診断だけが小幅に良かった。affine/heel calibration は overall、hidden_tail、prefix_backtest のいずれでも raw 系より悪化した。

| surface | region | RMSE | MAE | within10 | margin mean | entropy mean | zero rank mean | bimodal rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| savgol_31_p2__raw | all | 108.534313 | 69.576973 | 0.165121 | 1.343931 | 0.693299 | 24.666847 | 0.483518 |
| rolling_median_11__raw | hidden_tail | 125.707127 | 76.419067 | 0.153122 | 1.289864 | 0.696782 | 32.398373 | 0.508490 |
| savgol_31_p2__raw | prefix_backtest | 87.718421 | 62.469505 | 0.177393 | 1.352752 | 0.694301 | 16.912889 | 0.470205 |

raw `raw__raw` との mean abs-error gain は、raw smoothing だけが小さく改善し、heel/flat calibration は悪化した。

| surface | region | mean abs-error gain vs raw | improved rate | gap gain | entropy reduction | decoy gap gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_median_11__raw | hidden_tail | 0.207899 | 0.186055 | 0.133844 | 0.015125 | 0.506772 |
| savgol_31_p2__raw | hidden_tail | -0.057475 | 0.203928 | 0.179089 | 0.019608 | 0.673629 |
| raw__heel_calibrated | hidden_tail | -2.056464 | 0.203191 | -0.008637 | -0.018298 | -0.185166 |
| rolling_median_11__heel_calibrated | hidden_tail | -2.075805 | 0.270507 | 0.121226 | -0.006411 | 0.254398 |
| savgol_31_p2__raw | prefix_backtest | 0.661982 | 0.203004 | 0.176596 | 0.018778 | 0.674403 |
| rolling_median_11__raw | prefix_backtest | 0.446634 | 0.181785 | 0.130796 | 0.014422 | 0.504657 |
| raw__heel_calibrated | prefix_backtest | -2.160580 | 0.195115 | -0.049655 | -0.020143 | -0.262186 |

fixed exp072 PF/Beam candidates の hidden_tail RMSE は calibration mode に依存せず同じで、best は既存 `likpf_mean`。

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| likpf_mean | 11.471434 | 6.989252 | 0.775439 | -1.036218 |
| pf_ancc | 14.106718 | 8.641896 | 0.701377 | -1.060010 |

`likpf_mean` observation readout は raw surface の方が heel calibration より良い。`raw__raw` は mean rank 18.163254 / top1 0.052105 / top5 0.233162、`raw__heel_calibrated` は mean rank 19.346994 / top1 0.045445 / top5 0.215344。

distance bucket 別の error correlation では、flat calibration の `zero_rank` が最大で abs corr 約 0.37 を示した。ただし flat calibration 自体が大きく悪化しており、直接候補改善ではなく不確実性・fallback 診断材料に限定する。

## 再現性

- deterministic anchor: false。診断生成物であり submission anchor ではない。
- seed policy: `no_rng_deterministic_linspace_sampling`
- upstream stochastic component: exp072 PF/Beam/likelihood-PF train feature cache
- Kaggle kernel version: v1
- exp072 cache gzip SHA256: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 cache decompressed SHA256: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- `surface_metrics.csv` SHA256: `fe48aefc4bcf0a20692336062683fcd8880de5683bd56ce91022a3737ead7e21`
- `gain_vs_raw.csv` SHA256: `86646b2d08071297b365af330e50a7a9ed54b7fc5889c573dd346dac259b447b`
- `shift_curve_metrics.csv` SHA256: `8efc7ba509fe6fd85bccb551b704d77a09b42543b67ea2bbc8b18176dc2069bb`
- `error_correlation_metrics.csv` SHA256: `3dd6894b53ed1b9f013e84100ec18e86f9511f6a5b17e04852f6106d42b0c0d7`
- `pfbeam_candidate_metrics.csv` SHA256: `f9700ad384db6040ab2650ee712a0b892edec82e740b02abfb3dd8f1e707545e`
- `pfbeam_observation_metrics.csv` SHA256: `dcbbda6623dfbcca2cad605405c88b9204f07575f17908971eebc6cf205b3721`
- `row_context.csv.gz` decompressed SHA256: `488850f4927e6cd50e02ad75ab1c55924c76501a9bec06ae516f61a87e76d3fd`

## 解釈

affine/heel calibration は direct PF/Beam generation 変更や ML feature 化へ進める根拠にならない。raw smoothing は gap / entropy / decoy gap では小さく改善するが、hidden_tail の TVT 誤差改善は rolling median で +0.207899 ft 程度に留まる。

`zero_rank`、entropy、secondary-mode/bimodal signals は error correlation の診断材料にはなるが、予測値や posterior mean として直接使うのは避ける。後続へ回す場合は P2 `topk_path_confidence_features` の confidence / uncertainty / fallback 特徴量候補に限定する。

結論として、`affine_shift_landscape_ruler_readout` は train-side diagnostic として完了。inference port / submit はしない。
