# exp130_pfbeam_normalized_diagnostic_score

## 概要

exp092 の U-projection correction / disagreement LightGBM surface に、PF/Beam 候補を well-local normalized U/MD 空間で採点した target-free confidence features を add-only で入れる実験。

PF/Beam 候補を直接置換したり hard switch したりせず、`pf_ancc`、`pf_z`、`beam_mean`、`beam_med`、`likpf_mean` の path smoothness、prefix slope residual、候補間 disagreement、shape score margin を LightGBM の追加特徴量として渡す。

## 仮説

PF/Beam 候補の raw disagreement feature は単純追加で悪化したが、well-local normalized U/MD 空間なら path の滑らかさや prefix slope 整合をスケール非依存に表せる。これを confidence feature として使うと、PF/Beam 候補を直接選ばずに exp092 surface の弱点を補える可能性がある。

## 検証方針

GroupKFold by well で `exp092_full_row_control` と `pfbeam_normalized_diagnostic_addonly` を同一 row 上で比較する。pooled RMSE、well-level regression、distance bucket、feature importance を確認し、改善しても raw-test parity と hidden-like stress を確認するまで submission candidate にはしない。

## 比較

- `exp092_full_row_control`: exp092 相当の projection correction + U disagreement features。
- `pfbeam_normalized_diagnostic_addonly`: control に normalized diagnostic / candidate shape / normalized disagreement features を追加。

## 生成物

- `exp130_pfbeam_normalized_diagnostic_score_metrics.csv`
- `exp130_pfbeam_normalized_diagnostic_score_by_well.csv`
- `exp130_pfbeam_normalized_diagnostic_score_bucket_metrics.csv`
- `exp130_pfbeam_normalized_diagnostic_score_projection_feature_summary.csv`
- `exp130_pfbeam_normalized_diagnostic_score_diagnostic_feature_summary.csv`
- `exp130_pfbeam_normalized_diagnostic_score_feature_importance.csv`
- `exp130_pfbeam_normalized_diagnostic_score_feature_importance_mean.csv`
- `exp130_pfbeam_normalized_diagnostic_score_feature_importance_mean_top.png`
- `exp130_pfbeam_normalized_diagnostic_score_predictions.csv.gz`
- `exp130_pfbeam_normalized_diagnostic_score_feature_schema.csv`
- `exp130_pfbeam_normalized_diagnostic_score_lgb_models/manifest.json`
- `exp130_pfbeam_normalized_diagnostic_score_summary.json`

## 状態

実装済み。Kaggle train は未実行。

## 所見

未実行。Kaggle train 完了後に更新する。
