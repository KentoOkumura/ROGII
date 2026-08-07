# 設計

## アプローチ

exp098 の rank-slot generator は維持し、feature group だけを compact 版に差し替える。これにより rank1/rank2/rank3 の候補選択順、score、U-space shape の計算は exp098 と同一になり、差分は LightGBM に渡す列の削減に限定される。

残す列:

- `rank*_candidate_minus_last_anchor`
- `rank*_score`
- `rank*_source_code`
- `rank*_u_slope`
- `rank*_u_curvature`
- `rank*_u_resid_mad`
- `rank_score_entropy`
- `rank_score_top1_margin`
- `rank_slot_u_std`
- `rank_slot_u_range`

削る列:

- pairwise candidate delta / absdiff
- rank 間 `u_diff` / `u_absdiff`
- `u_corr` と `u_resid` の符号反転ペア
- `u_abs_resid`
- `u_fit_degree`
- source flag 群、特にほぼ選ばれない `sc_ens` / `hyb`

## 実験範囲

- 対象実験: `exp105_compact_rank_slot_features_on_exp098`
- Route: `ml_model`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 比較対象: exp098 `lgb1` / `lgb_mean`、exp092 `lgb1`、exp077 policy、exp073 raw anchor
- 変更する変数: rank-slot feature columns
- 固定する変数: exp098 rank scoring、candidate set、base 196 features、target、GroupKFold、LightGBM family

## Variant

- `compact_rank_slot_features` のみを学習する。
- expected feature count は 196 + 22 = 218。
- `control_base` は exp073 固定比較値を使うため再学習しない。
- exp098 full rank-slot は既存結果を比較基準にし、同じ exp105 内では再学習しない。

## 再現性設計

- seed policy: `fixed_groupkfold_seed_no_new_pf_rng`
- stochastic 処理の有無: rank-slot feature generation 自体は deterministic。upstream PF/Beam cache と GPU LightGBM は stochastic component として扱う。
- PF/Beam / likelihood-PF: 新規生成せず、exp072 の deterministic cache を読む。
- 並列処理と乱数の関係: LightGBM は deterministic / force_col_wise / fixed threads を config に明記する。
- CPU/GPU runtime: primary は `gpu_repro_guard_dp_threads8`。必要なら CPU mode を追加で有効化できる。
- train cache / feature schema / model manifest / prediction SHA は Kaggle train summary に記録する。
- deterministic submission anchor ではなく、train-side feature audit として扱う。

## リスク

- 重複列削除により分岐候補が減り、exp098 より悪化する可能性がある。
- source flags を落とすことで非線形な candidate identity signal を失う可能性がある。
- exp098 は exp092 より弱いため、改善しても即提出候補にはしない。
- OOF global 改善だけで判断せず、worst-well、distance bucket、path continuity、raw-test feature parity を確認する。
