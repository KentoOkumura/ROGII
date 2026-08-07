# exp175_cluster_outlier_typewell_prior_gate 結果

## 仮説

exp109 / exp114 の prior correction は全体信号がある一方、worst-well regression が大きい。native typewell cluster の空間中心から外れ、近傍 well の majority cluster とも食い違う well に限定すれば、prior を弱く足す価値が残る可能性がある。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- fallback 比較: `exp092_u_projection_correction_disagreement_fullrun`
- prior 親: `exp109_typewell_neighbor_prior_features`, `exp114_spatial_neighbor_prior_signal_audit`
- cluster 親: `exp065_typewell_supertype_cluster_cv_audit`
- 検証: 固定 OOF prediction への cluster-outlier gated posthoc audit
- 提出: なし

## 現在の結果

Kaggle train v2 完了。3,783,989 rows / 773 wells の OOF を、exp148 `lgb_mean` と exp092 `lgb1` の 2 source で評価した。

| 項目 | 値 |
| --- | --- |
| kernel | `kentookumura/exp175-cluster-outlier-typewell-prior-gate-train` v2 |
| status | 完了 / 不採用 |
| exp148 baseline | RMSE 8.501281182 / MAE 5.335650953 / within10 0.856332035 |
| exp092 baseline | RMSE 9.322479896 / MAE 5.980980 / within10 0.822047 |
| best policy | exp148 `lgb_mean` baseline |
| decision | `cluster_outlier_prior_gate_not_supported` |

best non-baseline は exp148 `lgb_mean` に `typewell_native_overlap_0p999` を `own_z_gt2p0`、`std_le20`、`alpha=0.05`、clip 5ft で入れる policy だったが、RMSE 8.501592821 で baseline から +0.000311639 悪化した。exp092 側の best correction も RMSE 9.322923 で +0.000443 悪化した。

by-well では best non-baseline が 29 wells 改善 / 33 wells 悪化 / 711 wells 同値、最大悪化 +0.181173 RMSE、最大改善 -0.210044 RMSE。worst-well regression は小さく抑えられたが、global RMSE と MAE は改善しなかった。

bucket / subgroup では近距離 `000_050` が -0.004189 RMSE、`nearby_majority_diff_k8` subgroup が -0.002714 RMSE と局所改善した。一方で `250_500`、`500_1000`、`1000_plus`、exp115 hidden-like subgroup、`own_z_gt1p5` は悪化し、global 改善に届かなかった。

## 解釈

cluster-outlier gate で exp109/114 prior の大きな worst-well regression は抑えられるが、exp148 / exp092 の ML output に対する posthoc correction としては信号が弱すぎる。発火範囲を絞っても、ML output に typewell / spatial prior を直接足す方針は採用しない。

ただし、これは exp109/114 と同じ「PF/Beam/likPF 候補に prior を補正する」設計を cluster-outlier well だけに限定した検証ではない。今回の実験は補正対象も ML output に変えているため、exp109/114 の直接 follow-up としては軸がずれていた。PF/Beam/likPF 候補を対象に、今回の cluster-outlier gate だけを追加する no-training audit は未検証として残す。

## 次

inference port / submit はしない。ML output への `cluster_outlier_typewell_prior_gate` は完了/不採用として閉じる。一方、PF/Beam/likPF 候補への cluster-outlier gated prior correction は別 backlog として確認する価値がある。
