# 設計

## 1. 共通schedule source

SHA固定したexp209のscore-row posterior mean/stdをbase pathとする。visible prefixで
current-well robust affine初期値を作り、exp345と同じstateを使う。

```text
state = [intercept_b, log_scale_a]
transition = local-level random walk
slope bounds = [0.25, 4.0]
minimum prefix pairs = 40
minimum typewell GR std = 5
maximum prefix RMSE = 60
trim quantile = 0.90
robust iterations = 2
fallback = identity a=1,b=0
```

process noiseはouter-train visible-prefixだけからexp345式で一度推定する。
current-testでは同じ式を全train visible-prefixへ適用したdeployment tableを使う。
suffix TVT、error、fold-valid truthはschedule生成に使わない。

## 2. Variant A: causal EKF

各rowのfinite raw GRで一回更新したposterior stateをそのrowのscheduleとする。
missing raw GRはupdateをskipしてstateをpropagateする。covarianceはJoseph form。
scheduleを全rowfreezeしてからPFを一回実行する。

## 3. Variant B: bidirectional extended RTS

Variant Aと完全に同じforward EKF recordsを作り、score suffix全体を終端から
fixed-interval extended RTSで一回smoothする。identity transition、
`filtered_covariance * pinv(next_predicted_covariance)`、`rcond=1e-12`、
terminal smoothed=filtered、covariance symmetrization/floorはexp350から固定する。
future raw GRはtestでも観測可能だが、future TVTは使わない。

## 4. PFへの適用

両variantとも粒子TVTに対するemission centerだけを変更する。

```text
mu_t(particle) = a_t * GR_typewell(TVT_particle) + b_t
z_t = (GR_observed_t - mu_t(particle)) / sigma_GR_exp404
log L = -0.5 * min(z_t^2, 600)
```

sigmaはaで再scaleせずexp404 x1.0のまま固定する。PF state/dynamics、500 particles、
128 seeds、ESS resampling、roughening、missing-GR policy、temperature-5集約は固定。

raw GRはschedule updateとPF likelihoodの両方へ入るため、evidence double-useで
過信するriskがある。本実験ではtemperatureやsigmaで補正しない。

## 5. 段階・実行量・判定

Stage 0:

- 2 variants ×32 wells = 64 PF well-runs。
- 8,192 seed-well trajectories、4,096,000 particle starts。
- exp209 HMMとexp404 controlのrerunは0。
- causal schedule parity、RTS terminal/parity/covariance、fallback、finite coverage、
  boundary jump、truth-late、SHA、runtime/RSSをAND評価する。CVではない。

Stage 1:

- 全Stage 0 PASS・別承認時だけ2×773=1,546 PF well-runs、
  197,888 seed-well、98,944,000 particle starts。
- 各variantをexp404 controlへ独立判定し、同じOOFでwinnerを選ばない。
- 各variantに、pooled gain `>=0.05 ft`、4/5 folds、raw observed gain
  `>=0.05 ft`、raw missing/high missing/1000+/hidden-like 2面非悪化、
  by-well p95 `<=0`、worst `<=0.25 ft`、固定HMM/PF blend非悪化を要求する。

片方がPASSしても後続利用には新しいsteeringと別承認を要する。

## 6. 再現性・禁止事項・承認境界

- scheduleはdeterministic固定順、PFはexp404 stable per-well SHA256 seed。
- base path/process noise/schedule/forward covariance/smoothed schedule/predictionの
  logical/decompressed SHAをfreeze後にtruth、fold、roleをattachする。
- 初回runはanchorにしない。
- a/b、process noise、slope bound、trim、timing、causal/RTS blend、sigma、
  temperature、particle/seed、well/row gate、winner selection、same-OOF救済は禁止。
- 2026-07-30の追加依頼でcompact self-contained候補と専用testの実装を承認済み。
  その後の「実行してください」でcanonical train採用、Kaggle package/push、
  Stage 0実行を承認済み。Stage 1、inference、submissionは別承認とする。
