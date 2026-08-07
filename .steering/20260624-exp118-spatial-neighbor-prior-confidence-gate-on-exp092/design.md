# 設計

## 背景

exp114 では spatial / trajectory-shape neighbor prior が `likpf_mean` に対して global RMSE と全 distance bucket を改善した。一方で worst-well regression が最大 +6.508 RMSE 残ったため、direct correction / submit は閉じた。

次段階では、ML route submitted anchor である exp092 `lgb1` OOF prediction を固定し、spatial prior を小さく補正として足す row / well を confidence gate で制限できるか確認する。

## 入力

- exp114 OOF prior:
  - `exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz`
  - `prior_tvt`, `prior_delta`, `prior_std`, `neighbor_wells`, `distance_mean`, `azimuth_mismatch`
- exp092 OOF prediction:
  - `exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz`
  - 対象は `u_projection_correction_plus_disagreement` / `gpu_repro_guard_dp_threads8` / `lgb1`

## 方式

1. exp092 OOF prediction と exp114 OOF prior を `id`, `well` で merge する。
2. `target_tvt` と `true_tvt` の一致を確認する。
3. correction は `exp092_pred + alpha * clip(prior_tvt - exp092_pred, -clip, clip)` とする。
4. gate は target-free diagnostic のみで作る。
   - valid prior
   - prior std quantile
   - neighbor distance quantile
   - neighbor well count
   - azimuth mismatch
   - abs correction delta cap
5. policy ごとに RMSE / MAE / within10 / gate rate / correction magnitude を集計する。
6. 上位 policy だけ by-well、distance bucket、path continuity、wide prediction を保存する。

## 非対象

- inference notebook で `submission.csv` は作らない。
- 新規モデル学習はしない。
- exp092 への ML add-only feature 化は別 backlog とする。

## 判定

- global RMSE が exp092 より改善し、かつ worst-well regression が小さい場合だけ review 候補。
- global 改善があっても worst-well warning が残る場合は、direct correction は閉じる。
- 改善しない場合は、spatial prior は hard gate / postprocess ではなく ML feature 候補へ戻す。
