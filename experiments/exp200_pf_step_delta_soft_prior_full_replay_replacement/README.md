# exp200_pf_step_delta_soft_prior_full_replay_replacement

## 状態

- ルート: pf_beam
- 状態: completed_train_feature_cache_direct_pfbeam_rejected_no_submit
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-05
- 親: `pf_step_delta_soft_prior_full_replay_replacement` backlog / `exp072` / `exp186`

## 目的

raw train horizontal/typewell から exp072-style full replay train feature cache を作り直し、PF 系の粒子重みに per-step TVT delta soft prior を入れる。
既存 exp072 cache は生成入力として読まず、生成後の direct 比較対象としてだけ使う。

## 仮説

train evaluation rows の自然な per-step TVT delta は p95 0.05-0.06 ft、p99 0.07-0.08 ft 程度である。
PF 系の粒子 transition にこのレンジを超える急な step delta だけを soft penalty として入れれば、hard clip せずに不自然な jump を抑え、exp072 の `likpf_mean` を維持または改善できる可能性がある。

## 変更点

- `PF_ANCC`、`PF_Z`、128-seed likelihood-PF の particle likelihood に step-delta prior を追加する。
- prior: `delta_free010_cost0025_scale003`
  - `excess = max(0, abs(current_tvt - previous_particle_tvt) - 0.10)`
  - `prior = 0.025 * (excess / 0.03)^2`
  - `likelihood *= exp(-prior)`
- Beam search は exp072 baseline のまま変更しない。
- LightGBM 学習、inference port、submit は作らない。

## 検証方針

- Kaggle Notebook 実行を正とする。
- raw train horizontal/typewell 773 wells を入力にする。
- schema は exp072 互換の 196 features を期待する。
- 生成後に exp072 cache と `id` 完全一致を確認し、overall / distance bucket / by-well / step-delta rate を `direct_pfbeam_comparison.py` で比較する。

## 主な生成物

- `artifacts/exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_pixiux_likpf_step_delta_prior_public_replay_train_features.csv.gz`
- `artifacts/exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_summary.json`
- `artifacts/exp200_vs_exp072_overall_metrics.csv`
- `artifacts/exp200_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp200_vs_exp072_by_well_delta.csv`
- `artifacts/exp200_vs_exp072_step_delta_rates.csv`
- `artifacts/exp200_vs_exp072_summary.json`

## 実行入口

- 学習 notebook: `exp200_pf_step_delta_soft_prior_full_replay_replacement_train.ipynb`
- 推論 notebook: `exp200_pf_step_delta_soft_prior_full_replay_replacement_inference.ipynb`
- direct 比較: `python experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/direct_pfbeam_comparison.py`

## 注意

この実験は train feature cache generation までで、CV/LB は持たない。
`likpf_mean` が exp072 RMSE 11.594898 を維持または改善することを第一条件にし、許容悪化は最大 +0.02 RMSE とする。

## 所見

Kaggle train v1 は完了し、3,783,989 rows / 773 wells / 196 features の exp072-style full replay train cache を再生成できた。exp072 direct 比較では `id` mismatch 0。

主候補 `likpf_mean` は exp072 RMSE 11.594898 から exp200 RMSE 11.618341 へ +0.023444 悪化し、許容悪化 +0.02 を超えたため不採用。MAE と within10、short/mid distance bucket は改善したが、`1000_plus` と worst wells の悪化が残る。LightGBM、inference、submit は行わない。
