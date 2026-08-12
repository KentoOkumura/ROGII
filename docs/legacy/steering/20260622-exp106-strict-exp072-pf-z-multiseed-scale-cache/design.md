# 設計

## アプローチ

`exp104_pf_z_seedbag_scale_cache` の cache 読み込み、metrics、SHA 記録、notebook 構成を再利用し、PF 中核を exp072 `public_notebook_replay_audit.py` の `_pf_z` / `run_pf_z` と同じ strict 実装に差し替える。

最初に seed 1 本の `strict_pf_z_parity_seed` を `stable_seed("pf_z", well)` で再生成し、exp072 cache の `pf_z` と row-level diff を確認する。parity を満たす場合、well ごとに `stable_sha256(exp106, "strict_pf_z", well, seed_index)` 系の seedbag を走らせ、likelihood scale aggregation と seed 間不確実性を保存する。

## 実験範囲

- 対象実験: `exp106_strict_exp072_pf_z_multiseed_scale_cache`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較実験: `exp104_pf_z_seedbag_scale_cache`、`exp100_pf_z_unified_velocity_observation_prior`
- 変更する変数: exp072 strict `pf_z` の multi-seed 化、scale aggregation、cache 出力、parity diff 記録
- 固定する変数: exp072 feature cache row set、raw train horizontal/typewell 入力、TVT_input prefix-only fitting、CPU runtime、internet disabled

## 再現性設計

- seed policy: parity seed は exp072 と同じ `stable_seed("pf_z", well)`。multi-seed は `stable_sha256(experiment, "strict_pf_z", well, seed_index)`。
- stochastic 処理の有無: あり。particle initialization、process noise、resampling が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072 strict PF `pf_z` の seed bagging と likelihood scale aggregation を使う。
- 並列処理と乱数の関係: `num_workers=1` を前提に well 単位逐次。Numba 内では seed を明示し、thread scheduling に依存させない。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache raw/decompressed SHA、candidate wide raw/decompressed SHA、metrics SHA、summary SHA を保存する。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は作らない。
- Kaggle package bootstrap 確認方針: `validate-exp` と `prepare-kaggle-notebooks --strict` を通し、metadata と kernel source を確認する。

## リスク

- リークリスク: 評価区間 true TVT は scoring のみに使う。PF 生成は finite `TVT_input` prefix、MD、Z、GR、typewell GR だけを使う。
- CV/LB 不一致リスク: train-side diagnostic であり、提出候補化はしない。
- ランタイム/メモリリスク: 773 wells x 128 seeds x 600 particles は重い。smoke、64 seeds、128 seeds の段階実行を前提にする。
- 再現性リスク: seed 1 parity が exp072 cache と十分一致しない場合は multi-seed 評価へ進まない。
