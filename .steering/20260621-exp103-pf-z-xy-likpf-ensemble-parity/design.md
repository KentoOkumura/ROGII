# 設計

## アプローチ

exp072 `lik_pf` と同じ状態定義 `pos = TVT + Z`、128 seed、scale `[3, 5, 8, 12]` を使う。seed ごとの path と log likelihood を保持し、`liks - max(liks)` から scale ensemble を作る。

exp100 の XY slope idea は単発 PF の TVT velocity prior ではなく、prefix だけで fitted rate prior として入れる。具体的には prefix の `d(TVT_input + Z)/dMD` を目的変数、`dZ/dMD` と `dXY/dMD` を説明変数にした線形 prior を作り、粒子の rate likelihood と seed log likelihood に反映する。

## 実験範囲

- 対象実験: `exp103_pf_z_xy_likpf_ensemble_parity`
- Route: `pf_beam`
- 親実験: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: `pf_z_xy_slope` の seed ensemble 化、likelihood scale ensemble 化、candidate metrics parity
- 固定する変数: exp072 保存済み `pf_z` / `likpf_*` baseline、train pseudo-tail 行、評価指標

## 再現性設計

- seed policy: `stable_seed(exp103, xy_likpf, seed_root, well)` を seed base とし、同一 well 内で 128 sequential seed を使う。
- stochastic 処理の有無: あり。PF 初期化、process noise、resampling が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 128 seed ensemble あり。Beam と ML 学習はなし。
- 並列処理と乱数の関係: `num_workers=1`。well 並列は使わず、thread scheduling 依存を避ける。
- CPU/GPU runtime と deterministic flags: CPU Kaggle Notebook、GPU なし。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache raw/decompressed SHA、schema SHA、raw train input SHA、candidate gzip raw/decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: モデル、prediction、submission は生成しない。
- Kaggle package bootstrap 確認方針: `make validate-exp` と `make prepare-kaggle-notebooks --strict` で notebook metadata と config を確認する。

## リスク

- リークリスク: XY rate prior が評価区間 true TVT を参照すると leak になる。実装では prefix `TVT_input` の有限行だけを使う。
- CV/LB 不一致リスク: train-side pseudo-tail 監査であり LB 直接推定ではない。submit 判断には inference port と code competition 実行が別途必要。
- ランタイム/メモリリスク: exp100 は 8 variant x 360 particles で約 66 分。exp103 は 1 family だが 128 seed x 500 particles なので長時間化する可能性が高い。
- 再現性リスク: Numba RNG と gzip raw SHA は環境差があるため、seed policy と decompressed SHA を主証拠にする。
