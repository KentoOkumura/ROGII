# 設計

## アプローチ

exp093 で確認した PF/Beam/likelihood-PF 候補の oracle headroom を、直接候補選択ではなく exp073 LightGBM の structured feature として使う。候補はまず self-GR を外した 5 候補に限定する。

- `pf_ancc`
- `beam_mean`
- `likpf_mean`
- `sc_ens`
- `hyb`

rank score は target-free な不確実性、候補間差分、last anchor からの距離だけで作る。rank1/rank2/rank3 の裸の絶対 TVT は入れず、`candidate - last_known_tvt`、rank 間差分、source id / source flag、score gap / entropy、U-space projection residual / correction、rank 間 U-space disagreement を特徴にする。

## 実験範囲

- 対象実験: `exp098_selector_rank_slot_features_on_exp073`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 比較対象: exp073 raw anchor、exp077 postprocess anchor、exp092 U-space structured feature fullrun
- 変更する変数: rank-slot structured feature groups
- 固定する変数: exp073/exp072 base feature surface、target `TVT - last_known_tvt`、5-fold GroupKFold by well、LightGBM family

## Variant

- `rank_slot_u_disagreement` のみを学習する。
- この pattern は rank1-3 の `candidate - last_known_tvt`、rank 間差分、source flag、score、score gap、entropy、U-space projection residual/correction、rank 間 U-space disagreement をすべて含む。
- `control_base` は exp073 固定比較値を使うため再学習しない。

## 再現性設計

- seed policy: `fixed_groupkfold_seed_no_new_pf_rng`
- stochastic 処理の有無: rank-slot feature generation 自体は deterministic。upstream PF/Beam cache と GPU LightGBM は stochastic component として扱う。
- PF/Beam / likelihood-PF: 新規生成せず、exp072 の deterministic cache を読む。
- 並列処理と乱数の関係: LightGBM は deterministic / force_col_wise / fixed threads を config に明記する。
- CPU/GPU runtime: primary は `gpu_repro_guard_dp_threads8`。必要なら CPU mode を追加で有効化できる。
- train cache / test feature regeneration の SHA 記録方針: train summary と model manifest に source cache、schema、feature columns を記録する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction SHA: saved booster と OOF prediction SHA を model manifest / metrics に記録する。
- submission SHA: inference は未選択なので記録しない。
- Kaggle package bootstrap 確認方針: `make prepare-kaggle-notebooks ... --strict` と `make validate-exp` で確認する。

## リスク

- リークリスク: rank slot 生成に true TVT を使うと即 leakage。実装では target-free score のみに限定する。
- CV/LB 不一致リスク: rank score は exp093 で `pf_ancc` を拾えていないため、selector bias を ML に注入する可能性がある。
- ランタイム/メモリリスク: exp073 full replay cache 全量と 1 variant x 3 LightGBM family のため Kaggle GPU train 前提。
- 再現性リスク: upstream PF/Beam cache が mounted source 依存。Kaggle kernel source と SHA を結果に残す。
- 採用判断リスク: 全体 CV 改善でも by-well regression が出やすい。worst-well、near rows、long-tail、path continuity を確認するまで inference port しない。
