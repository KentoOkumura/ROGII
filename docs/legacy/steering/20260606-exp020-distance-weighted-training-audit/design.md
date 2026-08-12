# 設計

## アプローチ

1. `exp013` の row OOF artifact を読み、`raw_lightgbm_no_gr` を基準に距離 bucket 別の候補比較を行う。
2. train CSV から fold-safe な prefix-only `recent_linear` を再計算し、near-row の anchor 系 baseline と raw model を比較する。
3. raw residual prediction と true residual の差を bucket 別に集計し、bias、分散、target/pred residual scale を確認する。
4. 同じ `baseline.py` の feature builder と LightGBM no-GR 設定を使い、training-fold wells のみで sample-weight variants を fit する。
5. `near_mid_far_segmented_lightgbm` は距離 segment ごとに model を分け、valid rows は該当 segment の model で予測する。

## 実験範囲

- 対象実験: `exp020_distance_weighted_training_audit`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: row distance による sample weight、near/mid/far model segmentation
- 固定する変数: feature set `no_gr_signal`、LightGBM hyperparameters、GroupKFold by well、last-anchor residual target、row sampling cap

## 成功条件

- full CV が raw anchor 13.549257 を改善する。
- 改善が rows 0-249 の過剰最適化だけでなく、全 row RMSE で確認できる。
- `exp014` held-out postprocess 13.535596 と比較して、training 側の改善候補として次に進める価値がある。

## リスク

- リークリスク: valid well の rows や `TVT` は training-fold model fit に入れない。OOF artifact audit は診断用に限定する。
- CV/LB 不一致リスク: public visible wells は near-row 比率や hidden tail 長が private と異なる可能性があるため、LB ではなく well-level CV と距離 bucket を主判断にする。
- ランタイム/メモリリスク: segment model は fold あたり最大 3 model になる。row sampling cap は exp013 と同じ 300k/fold、800/well を維持する。
