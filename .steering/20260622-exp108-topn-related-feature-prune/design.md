# 設計

## アプローチ

exp098 の target-free rank ordering と rank-slot feature generation を再利用する。違いは、追加 rank-slot 64 列だけでなく、既存 196 base features も含めた feature surface 全体を静的 column set で pruning する点にある。

base features は次の group に分類する。

- `base_196_all`: exp072 cache の全 196 base features。
- `base_196_candidate_family`: `pf_*`、`beam_*`、`sc*`、`hyb*`、`sig_*`、`likpf*`、`tdpf*`。
- `base_196_non_candidate_context`: candidate family 以外の geometry / GR / typewell context。
- `base_196_topn_core_candidate_family`: `pf_ancc`、`pf_ancc_std`、`pf_ancc_delta`、`beam_mean_d`、`beam_std_d`、`beam_med_d`、`likpf_mean_d`。

rank-slot features は exp098 full groups を保った上で、prune 用に次の追加 group を作る。

- `rank_slot_top1_related`
- `rank_slot_top2_related`
- `rank_slot_top3_related`
- `rank_slot_topn_slot_features`
- `rank_slot_global_candidate_stats`
- `rank_slot_source_flags`
- `rank_slot_pairwise_disagreement`

prune 候補は次の 5 本。

- `exp098_full_260`: exp098 control。
- `top1_related_pruned_260`: non-candidate context + core candidate family + top1 related rank-slot features。
- `top2_related_pruned_260`: top2 related まで拡張。
- `top3_related_pruned_260`: top3 related まで拡張。
- `non_candidate_context_plus_topn_related`: base candidate family を全削除し、non-candidate context + top3 related rank-slot features のみ。

ただし GPU 節約のため、実際に学習する active variant は `top3_related_pruned_260` のみとする。top3 固定の根拠は exp098 の既存結果:

- rank3 は `pf_ancc` 41.26%、`beam_mean` 52.26%、`likpf_mean` 6.48% で、rank1/rank2 と異なる candidate family 情報を持つ。
- feature importance では `rank3_u_curvature`、`rank3_u_slope`、`rank3_u_resid_mad`、`rank3_candidate_minus_last_anchor` が有効。
- `sc_ens` / `hyb` は top3 でもほぼ選ばれないため、top4/top5 を使う根拠は薄い。

## 実験範囲

- 対象実験: `exp108_topn_related_feature_prune`
- Route: `ml_model`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: 学習に使う静的 feature column set。
- 固定する変数: exp073/exp072 196-feature surface、candidate score、GroupKFold by well、LightGBM config family、GPU reproducibility guard mode。

## 再現性設計

- seed policy: GroupKFold seed と LightGBM seed は exp098 と同じ固定設定。
- stochastic 処理の有無: feature grouping / pruning 自体に新規 RNG はない。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 cache の PF/Beam/likelihood-PF features を読むだけで、本実験では再生成しない。
- 並列処理と乱数の関係: pruning は deterministic pandas/numpy 処理。LightGBM は deterministic flags と fixed `n_jobs` / `num_threads` を使う。
- CPU/GPU runtime と deterministic flags: 既定は `gpu_repro_guard_dp_threads8`。CPU mode は config に残すが active ではない。
- train cache / test feature regeneration の SHA 記録方針: train summary に exp072 cache SHA、feature schema、prediction SHA を記録する。inference は未選択。
- model manifest / prediction / submission SHA 記録方針: Kaggle train 後に model manifest SHA と OOF prediction SHA を記録する。submission SHA は train-side audit なので不要。
- Kaggle package bootstrap 確認方針: `make prepare-kaggle-notebooks EXP=exp108_topn_related_feature_prune EXTRA_ARGS="--notebook train --run-on-push --strict"` 後に metadata と notebook import を確認する。

## リスク

- リークリスク: candidate score は target-free だが、downstream validation は by-well GroupKFold を維持する。
- CV/LB 不一致リスク: rank-slot signal は path continuity を崩す可能性があるため、OOF global だけでは submit しない。
- ランタイム/メモリリスク: active variant を 1 本に絞るため、top1/top2/top3 ablation は行わない。必要なら train 後の結果を見て別 exp で追加する。
- 再現性リスク: upstream exp072 cache と GPU LightGBM に依存する。deterministic anchor ではなく train-side audit として扱う。
