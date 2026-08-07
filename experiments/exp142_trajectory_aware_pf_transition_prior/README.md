# exp142_trajectory_aware_pf_transition_prior

## 状態

- Route: `pf_beam`
- Status: `completed_train_side_rejected_no_submit`
- 親実験: `exp106_strict_exp072_pf_z_multiseed_scale_cache`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 提出候補: なし。trajectory-aware PF transition prior は不採用。

## 仮説

exp083 / exp141 の Z-driven 兆候から、TVT 急変の一部は地層面境界だけでなく坑跡 Z 成分の変化に同期している可能性がある。既存 `pf_z` は Z を velocity likelihood として見るが、transition の平均速度・process noise は局所 trajectory shape に十分追随していない。`dZ/dMD` と `d2Z/dMD2` を transition prior に入れることで、Z-driven 区間の候補多様性と oracle / gate headroom が増えるかを確認する。

## 検証方針

exp072 train pseudo-tail cache と raw train horizontal/typewell を使い、strict exp072 PF-Z parity、exp106 multiseed、trajectory-aware PF variants、既存 `likpf_mean` / `pf_z` / `pf_ancc` / beam を同じ rows で比較する。

主要指標は RMSE / MAE / within10、distance bucket、`abs_dzdmd` / `abs_d2zdmd2` bucket、worst-well、path roughness、effective sample size、resample count、collapse rate。直接 `likpf_mean` を超えることは初回の必須条件にせず、Z-driven 候補の追加価値と collapse しないかを優先して見る。

## 所見

Kaggle train v1 は完了。best trajectory-aware candidate `traj_pf_zmean_s006_noise025_scale_3` は RMSE 23.132450 / within10 0.614683 で、`likpf_mean` 11.594898 から +11.537553、`exp072_pf_z` 17.788171 からも +5.344279 悪化した。

near-prefix 0-250 ft では `likpf_mean` より良い bucket があるが、1000+ ft longtail で RMSE 25.710669 まで崩れた。transition prior への Z trajectory 直接注入は閉じ、Z は confidence / verifier / local diagnostics の材料に限定する。

保存済み生成物:

- `exp142_trajectory_aware_pf_transition_prior_candidate_metrics.csv`
- `exp142_trajectory_aware_pf_transition_prior_bucket_metrics.csv`
- `exp142_trajectory_aware_pf_transition_prior_by_well.csv`
- `exp142_trajectory_aware_pf_transition_prior_strict_pf_z_quality.csv`
- `exp142_trajectory_aware_pf_transition_prior_trajectory_pf_z_quality.csv`
- `exp142_trajectory_aware_pf_transition_prior_parity_diff.csv.gz`
- `exp142_trajectory_aware_pf_transition_prior_candidate_wide.csv.gz`
- `exp142_trajectory_aware_pf_transition_prior_summary.json`
