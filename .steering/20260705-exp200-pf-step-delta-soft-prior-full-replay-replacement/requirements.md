# 要件

## 依頼

`pf_step_delta_soft_prior_full_replay_replacement` backlog を実装する。

## 制約

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache` / `exp186_typewell_late_range_pfbeam_generation_soft_prior`
- raw train horizontal/typewell から exp072-style full replay train feature cache を再生成する。
- 既存 exp072 cache は生成入力として読まない。生成後の direct 比較対象に限定する。
- step-delta prior は `PF_ANCC`、`PF_Z`、128-seed likelihood-PF の particle likelihood のみに入れる。
- Beam search は初回実装では変更しない。
- 初期 active variant は 1 個に絞る。2 個目の候補は config に残すが、default run では inactive にする。
- LightGBM 学習、inference、submit は行わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam seed policy、gzip SHA、Kaggle kernel version を記録する。

## 受け入れ基準

- `experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/` に config、train/inference notebook source、feature cache wrapper、step-delta public replay implementation、direct comparison utility がある。
- `config.yaml` の `experiment.route` が `pf_beam`。
- selected prior は `delta_free010_cost0025_scale003`。
- 生成 cache の expected feature count は 196。
- `direct_pfbeam_comparison.py` が exp072 との `id` 完全一致、overall、distance bucket、by-well、candidate step delta rate を出せる。
- deterministic anchor としては扱わず、実行後は decompressed content SHA を主証拠として記録する。
