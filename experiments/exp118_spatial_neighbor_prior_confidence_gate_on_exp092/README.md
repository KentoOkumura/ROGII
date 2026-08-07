# exp118_spatial_neighbor_prior_confidence_gate_on_exp092

## 目的

exp114 の spatial neighbor prior を exp092 OOF prediction に対して小さな補正として使う場合、target-free な confidence gate で worst-well regression を抑えられるか診断する。

## 状態

- Route: `ml_model`
- 状態: Kaggle train v1 完了、review 候補、提出なし
- 提出: なし

## 仮説

exp114 spatial neighbor prior は direct correction としては worst-well regression が大きいが、exp092 の強い OOF prediction に対して小さく gated correction として使えば、一部 row / well で改善が残る可能性がある。

## 検証方針

exp092 `lgb1` OOF prediction と exp114 OOF prior を `id`, `well` で merge し、prior std、neighbor distance、neighbor count、azimuth mismatch、abs delta cap による gate grid を train-side に評価する。新規モデル学習、inference port、submit はしない。

## 入力

- exp114 OOF prior: `exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz`
- exp092 OOF prediction: `exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz`

## 出力

- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_gate_metrics.csv`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_by_well.csv`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_by_well_delta.csv`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_bucket_metrics.csv`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_path_continuity.csv`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_top_gated_predictions.csv.gz`
- `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_summary.json`

## 判断方針

この実験は調査結果を出す。global RMSE が改善しても worst-well regression や path continuity が悪ければ inference port / submit には進めない。

## 所見

Kaggle train v1 は complete。best は `lgb1__xy_only_k8__std_q50_distance_q50__a0p05__c5` で、exp092 `lgb1` RMSE 9.322479896 から 9.321625436 へ -0.000854460 改善した。within10 は +0.000153806、MAE は -0.001412906。

max well regression は +0.208085 RMSE で、設定した warning threshold 0.25 未満。path continuity は exp092 baseline と同等で、step >=10 は 1、step >=25 は 0。ただし改善幅は非常に小さく、250-500 / 500-1000 ft bucket はわずかに悪化するため、この実験単独では submit しない。

## ファイル

- 学習 notebook: `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_train.ipynb`
- 推論 notebook: `exp118_spatial_neighbor_prior_confidence_gate_on_exp092_inference.ipynb`
- 実装: `spatial_neighbor_prior_confidence_gate_on_exp092.py`
