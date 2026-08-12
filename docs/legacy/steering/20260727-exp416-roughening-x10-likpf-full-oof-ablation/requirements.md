# 要件

## 仮説

roughening 10倍はexp410 target-late sentinelだけの偶然ではなく、全773 wellsでも
exp072 likelihood-PFの正解particle basin維持・再捕捉を改善する。

## 依頼

exp410でtarget-lateな12 wells / 16 episodesに対して最も大きな改善を示した
resampling roughening 10倍を、exp072 likelihood-PFの単一変更として全773 train
wellsで検証する実験を設計する。今回はbacklog、steering、実験scaffoldと設計確定までとし、
実装、Notebook採用、Kaggle package、実行、推論、提出は行わない。

## 制約

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 原因根拠: `exp410_likpf_particle_resampling_basin_audit`
- candidateはroughening position `0.10 -> 1.00 ft`、roughening rate
  `0.001 -> 0.010`の同時10倍だけとする。
- particles 500、seeds 128、stable well seed、GR scale、初期分布、momentum、
  process noise、ESS threshold 0.5、systematic resampling、Type Well grid、
  GR missing補間、算術seed平均をexp072から変更しない。
- 保存済みexp072 `likpf_mean`をcontrolに使い、control PFは再実行しない。
- candidate 1 variant ×773 wellsのみ。LightGBM / HMM / Beam / GPUは0。
- exp410のsentinel、cause、episode、truth、errorをPF生成やwell選択に使わない。
- truth、fold、hidden-like roleはcandidate predictionとSHAをfreezeした後だけ読む。
- 初回実行はKaggle CPU 4 shardとし、ローカルPF実行は行わない。
- 再現性は`docs/06_reproducibility.md`に従い、stable seed、RNG call順、
  shard-independent well seed、raw/decompressed/logical SHAを記録する。
- 実装と実行はそれぞれ別のユーザー承認を必要とする。

## 受け入れ基準

- 変更がroughening 2値の10倍だけであることを機械的に検証できる。
- active scientific variant 1、candidate PF well-runs 773、control rerun 0、
  LightGBM config / fold / booster / GPU / HMM / Beamがすべて0と記録されている。
- 3,783,989 rows / 773 wells / folds 0--4を欠落・重複・fallbackなしで生成する。
- 保存exp072 controlのRMSE parity差が`1e-5 ft`以内である。
- pooled RMSEを`0.05 ft`以上改善し、4/5 folds以上で改善する。
- raw-GR observedは`0.05 ft`以上改善し、raw-GR missing、1000+、
  hidden-like spatial / typewell-purgedを悪化させない。
- by-well RMSE差p95を悪化させず、worst-well regressionを`0.25 ft`以内にする。
- exp410 persistent-offset episode SSEを5%以上減らす。
- 上記scientific guardはAND条件で、FAIL時はroughening倍率や他parameterを救済探索せず
  branchを閉じる。
- deterministic anchorと呼ぶ場合はinput/code/config/prediction content SHA、
  Kaggle kernel version、rerun parityを記録する。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠とする。

## 実装後の生成物

- Jupytext compact self-contained train source / Notebook候補
- roughening-only diff、exact parent parity、stable seed、LPT shard、truth freeze、
  strict merge、AND gateのcontract tests
- 実行時には4 shard prediction / well audit / manifest / summary、merged prediction、
  fold/scope/by-well/episode metrics、scientific gate、SHA manifestを生成する

## 次のアクション

正規Notebook採用とKaggle package作成の明示承認を待つ。承認前にpushやPF実行を
行わない。
