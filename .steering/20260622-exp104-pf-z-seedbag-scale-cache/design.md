# 設計

## アプローチ

`exp103_pf_z_xy_likpf_ensemble_parity` の未完実装を親にして、実験名・候補名・生成物名を `exp104_pf_z_seedbag_scale_cache` に揃える。中核ロジックは exp100 best の `pf_z_xy_slope` を、likelihood-weighted 128 seedbag と scale ensemble に変換するもの。

## 実験範囲

- 対象実験: `exp104_pf_z_seedbag_scale_cache`
- Route: `pf_beam`
- 親実験: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: pf_z seedbag 化、seed likelihood scale aggregation、cache 出力、既存候補との同一 metrics 比較
- 固定する変数: exp072 raw-train-only feature cache、train pseudo-tail row set、true TVT の scoring-only 使用、CPU runtime、internet disabled

## 再現性設計

- seed policy: `stable_sha256_seed_from_experiment_pf_z_seedbag_well`
- stochastic 処理の有無: あり。particle initialization、process noise、resampling が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: pf_z seedbag と likelihood-weighted scale aggregation を使う。
- 並列処理と乱数の関係: `num_workers=1`。well ごとに stable seed を作るため、thread scheduling 依存を避ける。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache の raw SHA と decompressed content SHA、exp104 candidate wide/long の raw SHA と decompressed content SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は作らないため記録対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に `validate-exp` と notebook JSON check を通し、Kaggle train output 取得後に SESSION_NOTES / metrics / result を更新する。

## リスク

- リークリスク: raw train prefix の `TVT_input` だけで rate prior を fit し、評価区間 true TVT は scoring のみで使う。exp072 cache も raw-train-only feature cache を読む。
- CV/LB 不一致リスク: train pseudo-tail diagnostic であり、LB 候補として直接採用しない。
- ランタイム/メモリリスク: 773 wells、128 seeds、500 particles で CPU 長時間実行になりうる。必要なら Kaggle 上で max_wells smoke を先に使う。
- 再現性リスク: Numba 内で `np.random.seed(seed_base + seed_offset)` を使うが、well 単位逐次実行で global parallel RNG は使わない。deterministic submission anchor とは扱わない。
