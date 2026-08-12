# exp120 design

## 実験名

`exp120_typewell_geology_neighbor_prior_pf_likelihood_probe`

## route

`pf_beam`

PF/Beam candidate surface に対する likelihood 診断が主目的。ML route への特徴量投入は後続候補に分ける。

## 入力

- Raw train horizontal/typewell CSV: `data/raw/train` または Kaggle input の train directory。
- exp099 v2 feature cache:
  - `exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz`
  - candidate columns: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `pf_z` が存在すれば使用。
- exp065 cluster assignments:
  - `common_typewell_cluster_assignments.csv`
  - `native_overlap_0p999` / `native_overlap_1` / `exact_hash` などを自動検出。

## Fold

- well grouped 5 folds。
- sklearn が使える場合は `GroupKFold(n_splits=5)`。
- sklearn がない場合は stable SHA256 hash fold fallback。
- valid fold の rows は query と scoring のみに使い、neighbor source は train folds の同 group wells だけに限定する。

## Prior

### neighbor drift prior

同じ native overlap group の train-fold wells から、`md_since` bin ごとに `true_tvt - last_known_tvt` の weighted mean / std / count / source well count を作る。query row では:

`neighbor_prior_tvt = last_known_tvt + mean_neighbor_drift(md_since_bin)`

### marker boundary prior

typewell CSV に `Geology` があれば `Geology` 値の変化 TVT を marker とする。なければ typewell GR の大きな derivative / rolling z-score changepoint を marker fallback とする。horizontal row 側は GR derivative から `h_gr_regime_strength` を作り、strength が高い row でだけ candidate TVT が marker 近傍にあるほど likelihood を上げる。

### particle proxy likelihood

真の PF particle 列が保存されていないため、exp099 の candidate TVT columns を particle proxy として扱う。candidate ごとに:

- neighbor log likelihood: `-0.5 * ((candidate_tvt - neighbor_prior_tvt) / sigma_neighbor)^2`
- marker log likelihood: `h_gr_regime_strength * -0.5 * (distance_to_marker / sigma_marker)^2`

を計算し、softmax weighted TVT を作る。最終 candidate は baseline `likpf_mean` から weighted TVT 方向へ小さく clipped correction する。

## Variants

- `baseline_likpf_mean`
- raw candidates: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `pf_z` if present
- `neighbor_drift_prior`
- `marker_boundary_prior`
- `marker_plus_neighbor_prior`

主要 hyperparameter は `config.yaml` に置く。

## Outputs

- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe_variant_metrics.csv`
- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe_bucket_metrics.csv`
- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe_by_well.csv`
- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe_row_predictions.csv.gz`
- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe_summary.json`

## 採用判断

診断専用。global OOF が改善しても、worst-well regression、raw-test parity、hidden-like stress readout なしで inference port / submit しない。
