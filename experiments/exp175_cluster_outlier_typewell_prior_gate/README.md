# exp175_cluster_outlier_typewell_prior_gate

## 目的

exp109 / exp114 の typewell / spatial neighbor prior を、native typewell cluster の空間中心から外れた well だけに弱く適用する posthoc gate を診断する。

## 状態

- Route: `ml_model`
- 状態: Kaggle train v2 完了、不採用
- 提出: なし

## 仮説

typewell / spatial prior は全体では強い信号を持つが、direct correction では worst-well regression が大きい。cluster 中心から外れ、近傍 well の majority cluster ともずれる well に限定すれば、prior correction の悪化を抑えつつ一部の longtail / high-disagreement row を改善できる可能性がある。

## 検証方針

exp065 cluster assignment と exp114 well geometry summary から `own_cluster_dist_z`、`nearest_other_closer`、`nearby_majority_diff_k8` を作る。これを exp109 typewell prior、exp114 spatial prior、exp148 / exp092 OOF prediction に join し、`alpha=0.05/0.10/0.20`、clip `5/10/20/40ft` の弱い correction grid を train-side に評価する。

## 入力

- exp065 cluster assignment: `common_typewell_cluster_assignments.csv`
- exp109 typewell prior: `exp109_typewell_neighbor_prior_features_oof_predictions.csv.gz`
- exp114 spatial prior / geometry: `exp114_spatial_neighbor_prior_signal_audit_oof_predictions.csv.gz`, `exp114_spatial_neighbor_prior_signal_audit_well_geometry_summary.csv`
- base OOF prediction: exp148 `lgb_mean` または exp092 `lgb1`
- exp115 hidden-like split: optional stress subgroup

## 出力

- `exp175_cluster_outlier_typewell_prior_gate_gate_metrics.csv`
- `exp175_cluster_outlier_typewell_prior_gate_by_well.csv`
- `exp175_cluster_outlier_typewell_prior_gate_by_well_delta.csv`
- `exp175_cluster_outlier_typewell_prior_gate_bucket_metrics.csv`
- `exp175_cluster_outlier_typewell_prior_gate_subgroup_metrics.csv`
- `exp175_cluster_outlier_typewell_prior_gate_path_continuity.csv`
- `exp175_cluster_outlier_typewell_prior_gate_cluster_outlier_well_features.csv`
- `exp175_cluster_outlier_typewell_prior_gate_top_gated_predictions.csv.gz`
- `exp175_cluster_outlier_typewell_prior_gate_summary.json`

## 判断方針

global RMSE が改善しても、max well regression、near bucket、`1000_plus`、cluster-outlier subset、exp115 stress を確認する。未実行の raw-test/full-train parity があるため、この実験単独では inference port / submit に進めない。

## 所見

Kaggle train v2 完了。best は補正なしの exp148 `lgb_mean` baseline RMSE 8.501281182 で、ML output への posthoc correction は baseline を上回れなかった。best non-baseline は `typewell_native_overlap_0p999__own_z_gt2p0__std_le20__a0p05__c5` だが RMSE 8.501592821、baseline から +0.000311639 悪化した。near `000_050` や `nearby_majority_diff_k8` subgroup には小さい改善があるが、global / exp115 stress では支持されないため inference port / submit はしない。なお、exp109/114 と同じ PF/Beam/likPF 候補への cluster-outlier gated prior correction はこの実験では未検証。

## ファイル

- 学習 notebook: `exp175_cluster_outlier_typewell_prior_gate_train.ipynb`
- 推論 notebook: `exp175_cluster_outlier_typewell_prior_gate_inference.ipynb`
- 実装: `cluster_outlier_typewell_prior_gate.py`
