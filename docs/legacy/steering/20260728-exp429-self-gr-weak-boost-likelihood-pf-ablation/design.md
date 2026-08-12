# 設計

## 1. 結論

`exp429_self_gr_weak_boost_likelihood_pf_ablation`を、self-GRをPF内部の
particle observation likelihoodへ直接加える1変数・1 candidate ablationとして固定する。

`exp223`のHMMで最良だった固定weak boostをそのまま使い、exp072互換のtypewell GR
likelihood、transition、particle数、seed数、resamplingを一切変えない。
最終出力はfixed temperature-5をprimary、arithmetic seed meanをsecondaryとする。

## 2. 実験範囲

- 対象実験: `exp429_self_gr_weak_boost_likelihood_pf_ablation`
- Route: `pf_beam`
- 親実験: `exp417_scale5_seed_aggregation_promotion_audit`
- kernel control: `exp072_exp063_full_replay_feature_cache`
- self-GR式: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- PF実装参照: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- 変更する変数: particle observation log-likelihoodへ固定self-GR weak boostを加える。
- 固定する変数: raw input、known-prefix anchor、typewell GR emission、GR sigma x1.0、
  PF state/dynamics、500 particles、128 seeds、stable seed、ESS resampling、
  roughening、clamp、temperature-5、score rows、reporting folds。

## 3. self-GR surface

surfaceは`exp223`と同じ固定設定を使う。

- window radius: 12 rows
- descriptor offsets: `[-12, -8, -4, 0, 4, 8, 12]`
- top-k: 5
- prefix anchor stride: 3
- max / keep-last / min prefix anchors: `128 / 32 / 12`
- max window missing rate: 0.35
- Gaussian TVT sigma: 12 ft
- descriptor distance temperature: 1.5
- typewell agreement sigma: 18 ft
- quadratic clip: 60
- surface chunk: 256

anchor descriptorとanchor TVTはfinite `TVT_input` prefix rowだけから作る。
evaluation descriptorはunknown suffixの観測可能GRを使う。true TVTは使わない。

各evaluation rowについてtop-5 anchor TVTを中心とするGaussian mixture
log-likelihoodを、PFの許容TVT範囲
`[typewell_min - 100 ft, typewell_max + 100 ft]`、step 0.2 ft上に作る。
row内で平均0、標準偏差下限0.25としてstandardizeし、`centered_self_ll`を得る。
quality `q_t`も`exp223`と同じprefix coverage、missing、descriptor sharpness、
top1/top2 gap、typewell peak agreementの積で作る。無効rowは`q_t=0`とする。

## 4. particle likelihoodへの直接統合

唯一のscientific variantは
`likpf_scale5_selfgr_boost_only_a070_c100`とする。

各row `t`、particle `p`について、particleのTVT state
`x_tp = position_tp - Z_t`でself-GR gridを線形補間する。補間対象は
grid上で先に`clip(centered_self_ll, 0, 1)`したboostである。

```text
base_loglik_tp =
  -0.5 * min(((GR_t - TypeWellGR(x_tp)) / gr_sigma_w)^2, 600)

self_boost_tp =
  interp(clip(centered_self_ll_t, 0, 1), x_tp)

combined_loglik_tp =
  base_loglik_tp + 0.07 * q_t * self_boost_tp

particle_likelihood_tp = exp(combined_loglik_tp)
```

`gr_sigma_w`はexp072と同じknown-prefix residual stdのclip `[10, 60]`で、
multiplierは`1.0`、post clipはない。self-GRは非負boostだけで、proposal、
transition、position/rate、resampling条件を直接変更しない。ただしparticle weightが
変わるため、ESS、resampling、genealogy、seed trajectory、seed evidenceは結果として
変わり得る。これが本実験の対象である。

particle stateは上記padded range内へexp072と同じ方法でclampされるため、
self-GR補間の外挿は発生させない。missing GRのtypewell emission処理は
exp072と同じbidirectional linear interpolation後のtypewell mean fallbackを使う。
self-GR descriptorがmissing guardを満たさないrowは`q_t=0`でbase parityになる。

## 5. controlとreadout

### 5.1 primary

- control:
  exp404/417保存`likpf_scale_5_x1p0`、RMSE `10.914522073`
- candidate:
  `likpf_scale5_selfgr_boost_only_a070_c100`
- seed aggregation:
  full-suffix seed log-evidenceのfixed temperature `5.0`

temperature-5は本設計前に固定済みであり、実行後に変更しない。

### 5.2 fixed secondary

- control:
  exp072保存arithmetic mean、RMSE `11.594897884`
- candidate:
  同じself-GR PF seed bankのarithmetic mean

secondaryが良くてもprimaryへ差し替えない。scale 3/8/12、best seed、median、
mode、medoidは生成しない。

## 6. technical preflight

実装後、full前にtarget-freeでself-GR eligibilityを満たすwellをwell IDの
SHA256昇順から4本固定する。各wellで次の2 variantだけを同一seed labelで実行する。

- `alpha0_parity`: alpha 0。version 3では同じPF replayをabsolute float32で保存した
  exp404 `likpf_mean_x1p0`と最大差`<=1e-5 ft`。
- `alpha07_candidate`: 本candidate。finite coverage、非zero quality / boost、
  particle weight、ESS / resampling ledgerを確認する。

preflightはtechnical parityだけで、truth RMSEやpromotion判断に使わない。
preflight PASS後もfull Kaggle runには別承認を要する。

## 7. full評価

- rows / wells / folds: `3,783,989 / 773 / 5`
- score rows: `TVT_input` missing evaluation suffix
- group / reporting fold: `well_id` / exp226保存fold
- metric: pooled RMSE primary
- scopes:
  raw GR observed / missing、high missing、1000+、
  hidden-like spatial / typewell-purged
- tail:
  by-well delta p95、worst-well regression、improved/worsened well count
- fixed blend:
  exp209 exact HMMとの50:50 blendでcontrol/candidateを同じ式で比較
- PF diagnostics:
  self-GR valid/quality/boost、ESS、resampling、seed dispersion、
  seed evidence、position clip

candidate prediction、surface schema、logical/decompressed content SHA、
input manifestをfreezeした後だけtruth、error、fold、roleをlate joinする。

## 8. gate

### 8.1 technical gate

すべて必須とする。

- input SHA、3,783,989 rows、773 wells、5 folds一致
- preflight alpha0 arithmetic parity `<=1e-5 ft`
- alpha / clip / modeが`0.07 / 1.0 / boost_only`
- finite candidate coverage 1.0、fallback well 0
- self-GR anchorがfinite prefixだけ、truth-before-freeze read 0
- eligible well / rowとpositive boost applicationが1件以上
- seed、particle、PF dynamics、GR sigma x1.0、temperature-5が固定値と一致
- planned/actual execution count一致
- prediction、surface、schema、manifest SHA記録

### 8.2 scientific gate

すべて必須とする。

- primary scale5 RMSE gain `>=0.05 ft`
- primary fold改善 `>=4/5`
- arithmetic mean regression `<=0.0 ft`
- raw GR observed regression `<=0.0 ft`
- raw GR missing regression `<=0.0 ft`
- high-missing regression `<=0.0 ft`
- 1000+ regression `<=0.0 ft`
- hidden-like 2面の各regression `<=0.0 ft`
- by-well delta p95 `<=0.0 ft`
- worst-well regression `<=0.25 ft`
- fixed HMM/PF 50:50 blend regression `<=0.0 ft`

### Version 3 comparator訂正

version 2でalpha0は保存exp404 `likpf_mean_x1p0`と18,055行bit-exactだった。
保存exp072の`last_known_tvt + likpf_mean_d`はdelta保存からabsoluteへ戻す丸めにより
最大`0.000352 ft`差となったため、version 3はtechnical comparatorだけをexp404
absolute arithmetic列へ訂正する。上限`1e-5 ft`は緩和せず、scientific contract、
candidate generation、RNG、実行量、promotion gateは変更しない。

### Version 4 comparator dtype復元

version 3はexp404のfloat32列をCSVからfloat64へ読み戻した値と、exp429のメモリ上
float32予測を比較し、最大`0.000484375 ft`の10進serialization差を誤検出した。
exp404/exp429ともPF meanの凍結semantic dtypeはfloat32であるため、comparatorを
float32へ復元し、float32 bit pattern一致と従来`1e-5 ft`差をAND判定する。
posthocでは18,055/18,055行bit-exact、最大差`0.0 ft`。PF replay、prediction SHA、
surface SHA、activation counts、実行量、科学gateは変更しない。

PASSしてもraw-test port、inference、submissionは同じexp内の別設計・別承認とする。
FAIL時はalpha、clip、window、top-k、temperature、GR sigma、particle、seed、
transition、resampling、gate、blendで救済せずterminal closeする。

## 9. 実行量

technical preflight:

- 2 variants × 4 wells = 8 PF well-runs
- 1,024 seed-well trajectories
- 512,000 particle starts

full:

- scientific variants: 1
- candidate PF well-runs: 773
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- deterministic 4 CPU shards + 0-PF merge
- parent full control reruns: 0
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 10. 再現性設計

- seed policy:
  exp072互換`stable_seed("likpf", "train", well_id) + seed_index`
- stochastic処理:
  particle initialization、propagation、conditional systematic resampling、
  resampling roughening
- 並列処理:
  wellごとのstable seedを使い、global RNGをworker間で共有しない。
  shardはeval row countのdeterministic LPTで固定し、各wellは1 shardだけに置く。
- runtime:
  CPU-only、internet disabled、outer workers 2 / Numba thread 1を初期契約とする。
- SHA:
  raw/config/code/bootstrap、input schema/content、surface、prediction、
  well audit、merge inventory、metrics、manifestを記録する。
  gzipはdecompressed content SHAを主証拠とする。
- deterministic anchor:
  初回full runだけでは名乗らない。独立rerunでlogical prediction SHAが一致し、
  raw-test regenerationまで監査された場合だけ再分類できる。
- model/submission:
  model SHAとsubmission SHAは非該当。inference/submissionは無効。
- bootstrap:
  package作成時に埋め込みZIPのconfig、active variant、kernel source、
  CPU/internet、seed、shard indexを正規sourceと照合する。

## 11. リスク

- リークリスク:
  suffix GRは観測可能だが、self-GR anchor TVTへsuffix truthを混ぜるとリークする。
- 科学リスク:
  exp223のpositive signalがHMM smoothing固有で、PFではnonnegative boostが
  wrong basinを強化する可能性がある。
- safetyリスク:
  exp223とexp417はいずれもworst-well tailが大きく、pooled改善だけでは採用できない。
- ランタイム:
  self-GR surface生成とparticle interpolationがexp400より重くなり得る。
- 再現性:
  self-GRで早期weight差が生じるとresampling後のtrajectoryが大きく分岐するため、
  per-well stable seed、shard identity、prediction SHAが必須である。
