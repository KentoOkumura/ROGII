# 設計

## 方針

exp167 の typewell GR shift-scan diagnostic を土台に、typewell GR 側へ affine calibration を加える。calibration の fit source は known-prefix rows に限定し、tail truth は scoring のみに使う。

比較する calibration mode:

- `raw`: typewell GR をそのまま使う。
- `flat_calibrated`: known prefix の MD-linear TVT prior で typewell GR を sample し、horizontal GR へ robust affine fit する。
- `heel_calibrated`: known prefix の `TVT_input` で typewell GR を sample し、horizontal GR へ robust affine fit する。

比較する GR filter:

- `raw`
- `rolling_median_11`
- `savgol_31_p2`

## 評価

- shift-scan top1 TVT RMSE / MAE / within2 / within5 / within10
- top1-top2 cost gap、entropy、decoy gap
- raw vs calibrated gain
- distance bucket: `000_050`、`050_100`、`100_250`、`250_500`、`500_1000`、`1000_plus`
- well別 worst regression
- exp072 fixed candidates の RMSE と observation cost / rank / top1 gap

## 入出力

入力:

- `data/raw/train/*__horizontal_well.csv`
- `data/raw/train/*__typewell.csv`
- `experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`

出力:

- row context CSV gzip
- surface metrics CSV
- bucket metrics CSV
- well metrics CSV
- gain vs raw CSV
- PF/Beam candidate metrics CSV
- PF/Beam observation metrics CSV
- summary JSON
- `metrics.json`

## 再現性

- 新規乱数は使わない。row sampling は deterministic linspace。
- upstream exp072 PF/Beam cache は stochastic 由来の既存生成物として SHA を記録する。
- gzip 生成物は decompressed content SHA を主証拠にする。
- train-side diagnostic なので deterministic submission anchor とは呼ばない。
