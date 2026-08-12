# 設計

## アプローチ

exp092 を親にして、既存の U-projection correction / disagreement features をそのまま生成する。そこへ exp098 の target-free rank-slot generator を移植し、モデルに渡す列は backlog で指定された代表列だけにする。

初回 variant は `u_projection_rank_slot_small_merge` の 1 本だけにする。これにより、exp092 anchor に対して rank-slot signal の非重複ゲインが残るかを、exp098 full union や pruning 再試行に広げる前に小さく検証する。

## 実験範囲

- 対象実験: `exp139_exp092_exp098_small_rank_slot_merge`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: exp092 feature surface に追加する small rank-slot feature subset
- 固定する変数: exp072 train cache、exp073/exp092 base 196 features、U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM lgb0/lgb1/lgb2 family、GPU deterministic mode

## Small Rank-Slot Columns

初期列は次に限定する。

- `rank1_candidate_minus_last_anchor`
- `rank1_score`
- `rank1_source_code`
- `rank2_candidate_minus_last_anchor`
- `rank2_score`
- `rank1_minus_rank2_score_gap`
- `rank_score_top1_margin`
- `rank1_u_slope`
- `rank1_u_curvature`
- `rank1_u_resid_mad`
- `rank2_u_slope`
- `rank2_u_curvature`
- `rank2_u_resid_mad`
- `rank_slot_u_std`
- `rank_slot_u_range`

## 再現性設計

- seed policy: GroupKFold seed 42。rank-slot feature generation は RNG なし。
- stochastic 処理の有無: 新規 stochastic feature generation はなし。上流 exp072 PF/Beam cache と GPU LightGBM training は stochastic component として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 本実験では再生成せず exp072 cache を読む。inference では public replay flow が raw test feature を再生成するため、feature parity と SHA を記録する。
- 並列処理と乱数の関係: rank-slot 生成は deterministic pandas/numpy 処理。LightGBM は deterministic flags、`gpu_use_dp=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8` を使う。
- CPU/GPU runtime と deterministic flags: 初回 active mode は `gpu_repro_guard_dp_threads8`。CPU mode は config に保持するが初回 active にはしない。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA、feature schema SHA、train/inference feature schema、prediction SHA、summary JSON を保存する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train manifest に model SHA と model count を保存する。inference / submit に進む場合は prediction SHA と submission SHA を `SESSION_NOTES.md` と `metrics.json` に追記する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、generated metadata、GPU/internet flags、kernel source、bootstrap manifest の `config.yaml` と補助 `.py` SHA を確認する。

## リスク

- リークリスク: rank-slot score が true TVT を使うと漏洩する。exp098 と同じ target-free score のみを使い、true TVT は LightGBM target と OOF metric に限定する。
- CV/LB 不一致リスク: exp092 は by-well regression warning がある。global OOF 改善だけでは submit しない。
- ランタイム/メモリリスク: exp092 の 240 features に rank-slot 15 columns と full rank-slot generation cost が追加される。exp098 full run よりは feature count が少ないが、3.78M rows の GPU LightGBM なので Kaggle 12h 制限を監視する。
- 再現性リスク: hidden test raw feature regeneration は normal notebook では観測できない。inference port 後は raw-test feature parity と submission-rerun behavior を別途確認する。
