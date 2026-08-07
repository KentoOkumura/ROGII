# 設計

## アプローチ

`exp072` の保存済み PF/Beam/likPF feature cache を読み、`likpf_mean` を default path とする。`pf_z` は主予測ではなく、Z trajectory と候補 path の形から target-free に信用できる区間だけで選ぶ。

gate は以下を使う。

- `abs(dzdmd)` と `-dZ/dMD` 由来の Z slope 強度。
- `pf_z` slope が `-dZ/dMD` に `likpf_mean` より近いかを表す alignment margin。
- `likpf_mean - pf_z`、`pf_ancc - pf_z`、`pf_ancc - beam_mean` の候補差分。
- `pf_z` curvature / roughness guard。
- `md_since`、tail rank、near-prefix guard。
- row / segment / well scope と switch-rate cap。

## 実験範囲

- 対象実験: `exp141_z_driven_pf_z_candidate_gate`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 参考親: `exp083_pf_beam_true_tvt_2d_well_eda`、`exp104_pf_z_seedbag_scale_cache`、`exp106_strict_exp072_pf_z_multiseed_scale_cache`
- 変更する変数: `pf_z` を選ぶ target-free gate 条件、switch-rate cap、segment minimum length、soft correction alpha。
- 固定する変数: input feature cache、`likpf_mean` default、`pf_z` candidate、評価 rows、metric。

## 再現性設計

- seed policy: `no_new_rng_posthoc_saved_cache_audit`
- stochastic 処理の有無: なし。保存済み exp072 cache を読むだけ。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。上流 cache の生成物としてのみ利用する。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。
- CPU/GPU runtime と deterministic flags: CPU-only posthoc audit。GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: input `.csv.gz` は raw file SHA と decompressed content SHA、schema SHA を `summary.json` に記録する。
- model manifest / prediction / submission SHA 記録方針: 新規モデル、submission は作らないため対象外。prediction sample は診断生成物として保存する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` 後に metadata と manifest を確認する。

## リスク

- リークリスク: gate 条件に true TVT / oracle を混ぜると不正になるため、評価列は scoring のみに隔離する。
- CV/LB 不一致リスク: train-side posthoc audit なので、良くても raw-test-compatible inference port と hidden-like stress check が必要。
- ランタイム/メモリリスク: exp072 full cache は大きい。必要列だけ `usecols` で読み、model training は行わない。
- 再現性リスク: 上流 exp072 cache の stochastic 性に依存するため、exp141 自体は deterministic anchor として扱わない。
