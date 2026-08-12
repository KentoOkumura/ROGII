# 設計

## アプローチ

exp098 の target-free candidate score と rank ordering を再利用し、rank slot に入った候補だけから特徴量を作る。候補値を予測として直接採用せず、exp073/exp072 base features に add-only で渡す。

feature group は次の 3 つに分ける。

- `top1_candidate_only`: `rank1_candidate_minus_last_anchor`、`rank1_score`、`rank1_source_code`、`rank1_u_slope`、`rank1_u_curvature`、`rank1_u_resid_mad`
- `top2_candidate_only`: top1/top2 の slot features と `top2_score_margin`、`top2_candidate_delta_spread`、`top2_u_spread`
- `top3_candidate_only`: top1/top2/top3 の slot features と `top3_score_margin`、`top3_candidate_delta_spread`、`top3_u_spread`

## 実験範囲

- 対象実験: `exp107_selector_topn_candidate_only_features`
- Route: `ml_model`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: rank-slot 追加特徴量の列集合。
- 固定する変数: exp073/exp072 196-feature surface、candidate score、GroupKFold by well、LightGBM config family、GPU reproducibility guard mode。

## 再現性設計

- seed policy: GroupKFold seed と LightGBM seed は exp098 と同じ固定設定。
- stochastic 処理の有無: feature generation 自体に新規 RNG はない。
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 cache の PF/Beam/likelihood-PF features を読むだけで、本実験では再生成しない。
- 並列処理と乱数の関係: top-n feature generation は deterministic pandas/numpy 処理。LightGBM は deterministic flags と fixed `n_jobs` / `num_threads` を使う。
- CPU/GPU runtime と deterministic flags: 既定は `gpu_repro_guard_dp_threads8`。CPU mode は config に残すが active ではない。
- train cache / test feature regeneration の SHA 記録方針: train summary に exp072 cache SHA、feature schema、prediction SHA を記録する。inference は未選択。
- model manifest / prediction / submission SHA 記録方針: Kaggle train 後に model manifest SHA と OOF prediction SHA を記録する。submission SHA は train-side audit なので不要。
- Kaggle package bootstrap 確認方針: `make prepare-kaggle-notebooks EXP=exp107_selector_topn_candidate_only_features EXTRA_ARGS="--notebook train --run-on-push --strict"` 後に metadata と notebook import を確認する。

## リスク

- リークリスク: candidate score は target-free だが、downstream validation は by-well GroupKFold を維持する。
- CV/LB 不一致リスク: row-wise rank signal は path continuity を崩す可能性があるため、OOF global だけでは submit しない。
- ランタイム/メモリリスク: top1/top2/top3 を同時に train するため exp098 単一 variant より LightGBM 実行時間が増える。
- 再現性リスク: upstream exp072 cache と GPU LightGBM に依存する。deterministic anchor ではなく train-side audit として扱う。
