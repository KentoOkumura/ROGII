# 設計

## アプローチ

exp139 では exp092 の U-projection correction / disagreement surface に exp098 rank-slot 代表 15 列を add-only したが、best `lgb1` は exp092 から +0.002428 悪化した。一方で `rank1_u_curvature`、`rank2_u_curvature`、`rank2_u_slope`、`rank1_u_slope` は重要度上位に入り、rank-slot signal 自体はあるが exp092 生成列と重複している可能性が高い。

exp147 では add-only をやめ、base 196 features は残したまま、exp092 生成列のうち rank-slot と意味が近い列だけを drop する。そこへ exp098 と同じ target-free rank-slot generator から作る rank1/rank2/rank3 の score、source、candidate-minus-anchor、U-shape、rank 間 gap、rank-slot spread を入れる。

## 実験範囲

- 対象実験: `exp147_exp092_exp098_rank_slot_replacement_only`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: rank-slot-overlap の exp092 generated columns を落とし、rank-slot replacement columns を入れる
- 固定する変数: exp072 train cache、exp073/exp092 base 196 features、U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM lgb0/lgb1/lgb2 family、GPU deterministic mode

## Replacement Columns

落とす列は、rank-slot の候補集合と重なる `pf_ancc`、`beam_mean`、`likpf_mean` 由来の projection correction と、その 3 者間 disagreement、全 source/correction spread に限定する。`pf_z` と `beam_med` 由来の projection correction は残す。

追加する列は rank1/rank2/rank3 の `candidate_minus_last_anchor`、`score`、`source_code`、`u_resid`、`u_slope`、`u_curvature`、rank 間 score / candidate gap、`rank_score_top1_margin`、`rank_slot_u_std/range` に限定する。

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
- CV/LB 不一致リスク: exp092 は by-well regression warning と near-row inconclusive がある。global OOF 改善だけでは submit しない。
- ランタイム/メモリリスク: features は exp092 とほぼ同規模だが、3.78M rows の GPU LightGBM なので Kaggle 12h 制限を監視する。
- 再現性リスク: hidden test raw feature regeneration は normal notebook では観測できない。inference port 後は raw-test feature parity と submission-rerun behavior を別途確認する。
