# 設計

## アプローチ

exp072-compatible forward likelihood-PF を保持し、各 row の weighted pre-resampling particle state と resampling ancestor map を `lag+1` 行の ring buffer に保存する。row `t` を処理した時点で `t-lag` まで ancestor を逆引きし、row `t` の posterior weight で `t-lag` state の smoothed mean を計算する。最後の lag 行は forward mean を使用する。

## 実験範囲

- 対象実験: `exp235_fixed_lag_particle_smoother_pf`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較: exp072 `likpf_mean`、forward PF、lag 64/128/256
- 固定: transition、Gaussian likelihood、500 particles、128 seeds、resampling、seed aggregation
- 変更: ancestor/state retention と delayed smoothing のみ

## 再現性設計

- seed policy: `SHA256(experiment, well, lag variant, public_likpf, seed index)`
- stochastic 処理: particle propagation と systematic resampling
- 並列処理: well 単位 single worker。thread scheduling に RNG を依存させない
- CPU/GPU: CPU-only、GPU/internet disabled、Numba single worker
- memory: ring buffer は `(lag+1) x particles` の state / ancestor に限定
- SHA: input cache、row candidates は decompressed content SHA を保存
- bootstrap: Kaggle source / config / variant selection を notebook 上で確認

## リスク

- リークリスク: future GR は許可するが future TVT は禁止。tail fallback と row alignment を厳密検査する。
- CV/LB 不一致: train-side pseudo-tail のみであり、raw-test parity を別途確認する。
- runtime/メモリ: lag と particle history に比例するため variant 分割を許容する。
- モード消失: ancestor tracing は消えた正解 mode を復元できない。rejuvenation の代替とは扱わない。
- runtime: full-surface の naive ancestor trace は lag64 でも Kaggle 12時間枠を超えたため、stable SHA well assignment による4 shardへ分割する。各 shard の PF semantics は不変で、全shardのID / well / row_idxが厳密にdisjointかつ全体を被覆した場合だけmergeする。
