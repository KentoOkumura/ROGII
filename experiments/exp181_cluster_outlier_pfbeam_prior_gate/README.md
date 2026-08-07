# exp181_cluster_outlier_pfbeam_prior_gate

## 目的

exp109 / exp114 で改善した typewell / spatial neighbor prior correction を、PF/Beam/likPF OOF 候補に対して cluster-outlier well だけへ限定した場合に、global 改善と worst-well regression のバランスが改善するか診断する。

## 状態

- Route: `pf_beam`
- 状態: Kaggle train v1 完了、train-side audit として完了
- 提出: なし

## 仮説

exp109 / exp114 の prior は global には強いが、全 row に correction すると worst-well regression が大きい。native typewell cluster の空間外れ well だけに限定すれば、global 改善を大きく落とさず regression を抑えられる可能性がある。

## 検証方針

exp175 と同じ `own_cluster_dist_z`、`nearest_other_closer`、`nearby_majority_diff_k8` 系 gate を作る。補正対象は exp148 / exp092 ML output ではなく、exp109 OOF 内の `likpf_mean`、`pf_ancc`、`beam_mean` に限定する。

補正式は exp109 / exp114 と同じ `base + alpha * clip(prior - base)`。`alpha=0.05/0.10/0.20`、clip `5/10/20/40ft`、prior std / neighbor count gate を比較し、exp109 / exp114 の global best も reference policy として同じ表に出す。

## 入力

- exp109 OOF: typewell prior と `likpf_mean` / `pf_ancc` / `beam_mean`
- exp114 OOF: spatial prior
- exp114 well geometry: cluster-outlier gate 用 geometry
- exp065 cluster assignment: native typewell cluster
- exp115 fold assignments: hidden-like stress subgroup

## 出力

- `exp181_cluster_outlier_pfbeam_prior_gate_gate_metrics.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_by_well.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_by_well_delta.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_bucket_metrics.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_subgroup_metrics.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_path_continuity.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_cluster_outlier_well_features.csv`
- `exp181_cluster_outlier_pfbeam_prior_gate_top_gated_predictions.csv.gz`
- `exp181_cluster_outlier_pfbeam_prior_gate_summary.json`

## 判断方針

baseline は `likpf_mean` / `pf_ancc` / `beam_mean`、reference は exp109 global best と exp114 global best。global RMSE、distance bucket、cluster-outlier subset、exp115 hidden-like stress、changed rows/wells、max well regression、path continuity を見る。positive でも raw-test/full-train parity 前に inference port / submit へ進めない。

## 所見

Kaggle train v1 では、best gated policy `any_outlier_signal_k8/std_le20/a0.2/c40` が `likpf_mean` baseline RMSE 11.594897672 を 11.479140438 へ改善した。exp109 global reference は 11.143359414 まで改善するが、max well regression は +6.594183。best gated は max well regression を +4.359666 まで下げる一方、まだ direct correction としては大きい。clip20 の guarded policy は RMSE 11.497560716、max well regression +3.032388。

distance bucket と exp115 hidden-like stress は壊れていないため、prior signal と cluster-outlier gate 自体は有効。ただし inference port / submit には進めず、今後使う場合は direct posthoc correction ではなく selector / confidence feature / candidate scoring の材料に限定する。

## ファイル

- 学習 notebook: `exp181_cluster_outlier_pfbeam_prior_gate_train.ipynb`
- 推論 notebook: `exp181_cluster_outlier_pfbeam_prior_gate_inference.ipynb`
- 実装: `cluster_outlier_pfbeam_prior_gate.py`
