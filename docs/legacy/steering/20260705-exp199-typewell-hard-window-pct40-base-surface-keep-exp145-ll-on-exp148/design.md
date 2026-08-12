# 設計

## アプローチ

exp148 の LightGBM train flow をベースにし、base replay cache の入力だけを exp196 pct40 hard-window cache に差し替える。`build_u_projection_features` は exp196 base columns から再計算する。`build_learned_likelihood_features` は exp145 full-train `ml_features` をそのまま読み、exp196 base frame へ `id` / `well` で join する。

この実験は clean replacement ではなく、base surface 差し替え単体の低コスト診断である。`ll_candidate_tvt_*_minus_likpf_mean_tvt` は exp145 candidate TVT と exp196 `likpf_mean` の差分になるため、改善しても直接 submit しない。

## 実験範囲

- 対象実験: `exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: base 196 feature cache、projection_correction、u_disagreement
- 固定する変数: exp145 `ll_*` learned-likelihood cache、GroupKFold by well、LightGBM config family、target/residual 定義
- 初期スコープ外: current-test regeneration、saved-booster inference、submit、control retraining

## 再現性設計

- seed policy: fixed GroupKFold seed 42、LightGBM deterministic flags、new PF RNG なし。
- stochastic 処理の有無: train 実行では LightGBM GPU 学習が stochastic になり得る。feature merge 自体に RNG はない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成はしない。exp196 と exp145 の保存済み upstream cache を読む。
- 並列処理と乱数の関係: feature assembly は deterministic。LightGBM は `deterministic=true`、`force_col_wise=true`、`num_threads=8`、`gpu_use_dp=true` を使う。
- CPU/GPU runtime と deterministic flags: 初期 train は `gpu_repro_guard_dp_threads8`。CPU control は実装に残すが active にはしない。
- train cache / test feature regeneration の SHA 記録方針: train 完了時に exp196 gzip raw SHA / decompressed SHA、exp145 gzip raw SHA / decompressed SHA、schema SHA を `summary.json` と `metrics.json` に記録する。test regeneration は初期スコープ外。
- model manifest / prediction / submission SHA 記録方針: train 完了時に model manifest、OOF prediction SHA を記録する。submission SHA は submit しないため記録対象外。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict`、metadata の GPU / internet disabled / kernel sources、bootstrap 内 config を確認する。

## リスク

- リークリスク: exp196 と exp145 はどちらも target-free upstream cache だが、`ll_*` と base surface の provenance が混ざる。valid/test true TVT、oracle best、candidate true-error rank は使わない。
- CV/LB 不一致リスク: mixed provenance の OOF 改善は hidden-safe inference の根拠にならない。submit 前には clean regeneration が必要。
- ランタイム/メモリリスク: exp148 と同じ 3.78M rows x 約294 features + 15 boosters。GPU train は長時間になる。
- 再現性リスク: GPU LightGBM は bitwise anchor と見なさない。採用候補化する場合は SHA と rerun 差分を別途確認する。
