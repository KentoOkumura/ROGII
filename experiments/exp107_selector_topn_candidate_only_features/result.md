# exp107_selector_topn_candidate_only_features 結果

## 状態

Kaggle train v1 完了。提出はしない。

- Kernel: `kentookumura/exp107-selector-topn-candidate-only-features-train` v1
- Output: `experiments/exp107_selector_topn_candidate_only_features/kaggle/output/train_v1`
- rows: 3,783,989
- wells: 773
- models: 45 boosters
- runtime: 28,860.946 sec

## 実装内容

exp098 の target-free candidate ranking を再利用し、rank slot に入った候補だけから追加特徴量を作った。

- `top1_candidate_only`: 196 base + 6追加 = 202 features
- `top2_candidate_only`: 196 base + 15追加 = 211 features
- `top3_candidate_only`: 196 base + 21追加 = 217 features

候補集合全体の entropy / range、全 pairwise delta、source one-hot flag、`u_corr` / `u_resid` / `u_abs_resid` / `u_fit_degree` は使わない。

## OOF

| variant | model | features | RMSE |
| --- | --- | ---: | ---: |
| `top2_candidate_only` | `lgb2` | 211 | 9.437602823 |
| `top2_candidate_only` | `lgb1` | 211 | 9.437894828 |
| `top3_candidate_only` | `lgb2` | 217 | 9.458990 |
| `top2_candidate_only` | `lgb_mean` | 211 | 9.479092683 |
| `top3_candidate_only` | `lgb1` | 217 | 9.500311 |
| `top3_candidate_only` | `lgb_mean` | 217 | 9.527935589 |
| `top1_candidate_only` | `lgb1` | 202 | 9.551440 |
| `top1_candidate_only` | `lgb_mean` | 202 | 9.577677177 |

best は `top2_candidate_only` / `lgb2`。

## 比較

- vs exp073 raw anchor 9.526374749: -0.088771927
- vs exp077 policy 9.470514801: -0.032911978
- vs exp092 best lgb1 9.322479896: +0.115122927
- vs exp098 lgb1 9.358151052: +0.079451770
- vs exp098 lgb_mean 9.427447987: +0.010154835
- vs exp105 best 9.441103161: -0.003500339

exp105 compact はわずかに上回ったが、exp098 full rank-slot と exp092 には届かない。提出候補にはしない。

## 監査

rank source distribution は exp098 と同じ傾向。

- rank1: `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80%、`sc_ens` / `hyb` 0%
- rank2: `pf_ancc` 25.09%、`beam_mean` 23.19%、`likpf_mean` 51.72%、`sc_ens` / `hyb` 0%
- rank3: `pf_ancc` 41.26%、`beam_mean` 52.26%、`likpf_mean` 6.48%、`sc_ens` 8 rows、`hyb` 29 rows

best `top2/lgb2` の path continuity は崩れていない。

- abs step mean: 0.090228
- abs step p99: 0.738000
- abs step max: 10.718000
- step >= 10: 1
- step >= 25: 0

worst well は `86454a6f` RMSE 55.029583、次いで `fb03ae90` 43.787430、`389ae58f` 40.815479。

## 生成物

- metrics: `exp107_selector_topn_candidate_only_features_metrics.csv`
- by-well: `exp107_selector_topn_candidate_only_features_by_well.csv`
- bucket metrics: `exp107_selector_topn_candidate_only_features_bucket_metrics.csv`
- feature schema: `exp107_selector_topn_candidate_only_features_feature_schema.csv`
- model manifest: `exp107_selector_topn_candidate_only_features_lgb_models/manifest.json`
- predictions: `exp107_selector_topn_candidate_only_features_predictions.csv.gz`

SHA:

- summary SHA: `859fa539a92caa0b8199809ea19a99977ea1862b968da885cdfc56e4d28b664a`
- metrics SHA: `8e75e8b00fdae7d57e5874428fb249c5a2592eb311a5d07e38b58371d55dbf1f`
- feature schema SHA: `8cf8be55818ec3ea00cf19a82010c6ef185d49320f4d2402ca42ec1b59b88721`
- manifest SHA: `26d2e5a0c3b5029ec38a1f91ba966849b9a02f96513a78ec6ba8393dabf4b412`
- predictions decompressed SHA: `8f20066736360ab8457dfdfec5cd0a42f860a9ce069bf3236f0698952757f191`

## 結論

追加 rank-slot 列だけを top-n candidate-only に絞る方向は、exp098 full rank-slot より弱い。削った source flags、pairwise deltas、global disagreement / uncertainty のどれか、または full 64列の冗長な分岐候補自体が LightGBM に効いていた可能性がある。

exp107 は rejected。exp098 full rank-slot を比較基準として維持し、次は pruning より exp092 への小さな add-only merge、または candidate generation / likelihood 側を優先する。
