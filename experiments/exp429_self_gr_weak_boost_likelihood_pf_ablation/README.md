# exp429_self_gr_weak_boost_likelihood_pf_ablation

## 状態

- ルート: PF/Beam
- 状態: full 4 shard + merge完了・scientific FAIL・terminal close
- CV: `11.127406421`（control `10.914522073`、`+0.212884347 ft`悪化）
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-28
- 親実験: `exp417_scale5_seed_aggregation_promotion_audit`

## 仮説

同一horizontal wellのvisible-prefix self-GR motif surfaceを、exp072互換
likelihood-PFの各particle observation log-likelihoodへ固定weak boostとして
直接加えると、particle weight、ESS、resampling、trajectoryを通じて
fixed temperature-5 predictionを改善できる。

## 変更点

- exp223固定`boost_only / alpha=0.07 / clip=1.0`をparticle likelihoodへ加える。
- exp072 x1.0 typewell GR emission、transition、500 particles、128 seeds、
  resampling、roughening、clampを固定する。
- primaryはexp404/417の保存済みscale-5 x1.0 controlとの比較。
- arithmetic meanはexp072保存controlとのsecondary safety readout。
- `exp091`のcandidate比較と`exp128`のpost-hoc switchは直接PF結果に含めない。
- scientific variantは1、full parent control再実行は0。

## 検証方針

- Fold: exp226保存reporting fold 0--4
- Group: `well_id`
- Score rows: `TVT_input` missing evaluation suffix
- Leakage Check: self-GR anchorはfinite prefixだけ。candidate / surface / SHAを
  freezeしてからtruth、error、fold、hidden-like roleをlate joinする。
- Primary gate: scale5 gain `>=0.05 ft`、4/5 folds、固定scopeとwell-tailのAND。

## 実行入口

- 学習 notebook: `exp429_self_gr_weak_boost_likelihood_pf_ablation_train.ipynb`
- 推論 notebook: `exp429_self_gr_weak_boost_likelihood_pf_ablation_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp429_self_gr_weak_boost_likelihood_pf_ablation`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

compact self-contained train / fail-closed inferenceをJupytext percent形式から生成し、
正規train / inference notebookへ採用済み。Kaggle CPU preflight version 4は
technical PASSし、後続full 4 shardとstrict merge version 2まで完了した。
technical gateはPASSしたがscientific gateはFAILした。

## 設計上の実行量

- technical preflight: 2 variants × 4 wells = 8 PF well-runs
- full: 1 variant × 773 wells × 128 seeds × 500 particles
- full seed-well trajectories / particle starts: `98,944 / 49,472,000`
- full parent control / LightGBM / fold training / booster / model / HMM / Beam / GPU:
  すべて0

## 所見

### 良かった点

- HMMで支持された式をPF内部へ直接移植するため、未検証だった問いを正面から切り分ける。
- self-GR以外のPF変数を固定し、candidateやpost-hoc switchと混同しない。

### 悪かった点

- self-GR surface計算とparticle interpolationにより高コストである。
- HMM smoothingで有効だったboostがPFのwrong basinを強める可能性がある。

### リスク / 注意

- exp223とexp417はいずれもworst-well tailが大きいため、pooled改善だけではPASSしない。
- alpha、clip、window、top-k、temperature、PF parameterのsame-OOF救済は禁止する。

## 実行結果

- 専用contract test 13件、構文、Ruff F821、Jupytext round-trip、
  strict experiment validationはPASS済み。
- version 2は8 PF run、1,024 seed-well、512,000 particle startsを完走。
  alpha0は保存exp404 x1.0 arithmetic predictionと18,055行でbit-exactだったが、
  保存exp072 deltaのabsolute再構成との最大差`0.000352 ft`が`1e-5 ft`gateをFAIL。
- version 3はexp404保存float32値をfloat64として比較したため、
  同じfloat32 bit値のCSV往復差`0.000484375 ft`を誤検出した。
- 保存時の正規dtype `float32`へcomparatorを復元するだけのversion 4 debug retryを
  実行する。posthoc検証では18,055行すべてbit-exact、最大差`0.0 ft`。
  tolerance、PF/self-GR設定、実行量は変更しない。
- full 4 shardは`773 wells / 3,783,989 rows / 98,944 seed-well /
  49,472,000 particle starts`を完了した。
- merge technical gateはPASS。primary RMSEは`11.127406421`で、
  control `10.914522073`より`0.212884347 ft`悪化し、改善foldは`1/5`。
- arithmetic secondaryは`0.023416828 ft`改善したが、by-well p95
  `+0.770627049 ft`、worst well `+34.862601957 ft`、fixed HMM/PF blend
  `+0.070319209 ft`で固定guardをFAILした。
- `terminal_close_without_self_gr_or_pf_rescue_grid`。同一OOF rescue、
  inference、submissionへ進まない。

## 2026-07-28 Kaggle CPU preflight version 4結果

- Kaggle status: `COMPLETE`
- technical gate: `PASS`
- variants / wells / PF well-runs: `2 / 4 / 8`
- seed-well trajectories / particle starts: `1,024 / 512,000`
- alpha0 comparator: 保存exp404 `likpf_mean_x1p0`、semantic dtype `float32`
- bit-exact rows / comparator rows: `18,055 / 18,055`
- alpha0最大差: `0.0 ft`（固定上限`1e-5 ft`）
- candidate positive-quality rows / positive boost applications:
  `12,239 / 777,858,990`
- prediction / surface logical SHA:
  `997713bd08559411135bd48e9a19594fe4141885c08da3fd66b3070e96b009f3` /
  `6c4876f94fe94ec63da95b6b5f270cdc519bc2f50d3a9a992e6200cc46ac0c35`
- full 4 shard + mergeは後続承認により完了済み。inference、submissionは
  未実行で、scientific FAILにより無効のまま。

## 2026-07-29 full承認

- fixed full: 1 scientific variant / 773 wells / 98,944 seed-well /
  49,472,000 particle starts
- 実行方式: deterministic LPT 4 Kaggle CPU shards + zero-PF strict merge
- parent control、model、booster、HMM、Beam、GPU rerun: すべて0
- full/mergeは承認済み。inference、submissionは未承認。

## 2026-07-29 full結果

- 4 shard / merge: `COMPLETE / COMPLETE`
- technical / scientific gate: `PASS / FAIL`
- merged prediction logical SHA:
  `d7677deb40526274853178290d316efcc0b1bafe629d13c669f50ac062689ff0`
- artifact manifest SHA:
  `3620944f8ab0c6cf0b85c9fd7c11a9ed07897965324c774e43aa528dedbf694e`
- 実ファイル: `artifacts/kaggle_merge_v2/`
- 次: exp429内での再実行なし。既存の低優先
  `self_gr_quality_addonly_features_on_exp092`だけを、直接PF boostではない
  target-free risk特徴の独立候補として残す。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
