# 要件

## 依頼

self-GRをlikelihood-PFのparticle likelihood / emissionへ直接組み込む実験を
バックログ化し、steeringと実験ディレクトリを作成して設計を確定する。
今回は設計とscaffoldだけを対象とし、PFコード、notebook、test、Kaggle package、
実行、推論、提出は実装しない。

## 問いの固定

- 検証対象は、同一horizontal wellのvisible-prefix self-GR surfaceを
  PFの各particleの観測log-likelihoodへ直接加える効果である。
- `exp091`のself-GR candidate横並び評価と、`exp128`の保存済みPF出力に対する
  post-hoc switch / blendはPF内部統合の結果として扱わない。
- `exp223`で支持された`boost_only / alpha=0.07 / clip=1.0`だけをPFへ移植する。
- self-GRをproposal、初期粒子、transition、postprocess、candidate、selector、
  blendとして使わない。

## 制約

- Route: `pf_beam`
- 親: `exp417_scale5_seed_aggregation_promotion_audit`
- PF kernel control: `exp072_exp063_full_replay_feature_cache`
- self-GR式参照: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- exact PF実装参照: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- primaryはfixed temperature-5 seed aggregation、controlはexp404/417の保存済み
  x1.0 scale-5 predictionとする。
- arithmetic seed meanはexp072保存値との固定secondary safety readoutとし、
  実行後にprimaryへ差し替えない。
- scientific variantは1つだけとし、alpha、clip、window、top-k、temperature、
  GR sigma、particle、seed、transition、resamplingのgridを行わない。
- unknown-suffix true TVT、error、fold、hidden-like roleはcandidate predictionと
  schema / logical content SHAをfreezeするまで読まない。
- self-GR anchorは同じwellのfinite `TVT_input` prefix rowだけから作る。
  evaluation suffixのGRは観測入力として使えるが、suffix true TVTは使わない。
- 保存済みcontrolをfull 773 wellsで再実行しない。alpha=0は固定4-wellの
  technical preflightだけに限定する。
- Fullは1 candidate × 773 wells × 128 seeds × 500 particlesとする。
- LightGBM config、trained fold、booster、model、HMM、Beam、GPUはすべて0。
- 実装、正規notebook採用、Kaggle package、preflight/full runは追加承認を要する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- backlog、steering 3文書、実験scaffold、`config.yaml`、`README.md`、
  `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`が
  `design_frozen_not_implemented`として整合する。
- particle weightへ加える式、self-GR surface、quality、state interpolation、
  missing / out-of-range処理が実装者の追加判断なしに読める。
- primary / secondary control、technical preflight、full実行量、truth-late freeze、
  technical/scientific gate、PASS/FAIL後の停止条件が固定されている。
- `exp091/128`をPF直接統合のnegative resultとして扱わないことが明記されている。
- canonical train / inference notebookはtemplate placeholderのままで、実装コードや
  testを追加していない。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 非目標

- `exp223` HMMの再実行またはHMM/PF prediction blend。
- `exp091/128/134/296`の救済grid。
- temperature-5自体の再promotionまたは`exp417`の再分類。
- scale 3/8/12、alpha 0.15、symmetric mode、negative self-GR penalty。
- self-GRによるparticle proposal、transition、initialization、rougheningの変更。
- current-test regeneration、inference、submission。

## 2026-07-28 実装承認追補

ユーザーの `exp429を実装してください` という追加依頼により、compact
self-contained train / fail-closed inference、正規Notebook採用、contract test、
target-free固定4-well asset、静的・round-trip・strict validationまでを承認範囲へ
追加した。Kaggle package、preflight/full run、raw-test inference、submissionは
引き続き別承認とする。

## 2026-07-28 version 3 technical comparator訂正承認

version 2は固定4 wells・8 PF runsを完走し、alpha0 predictionが保存exp404
`likpf_mean_x1p0`と18,055行でbit-exactだった。一方、保存exp072のfloat32 deltaを
absoluteへ再構成した値とのrow最大差`0.000352 ft`が`1e-5 ft`gateを超えた。
ユーザーの`version3を実行してください`により、alpha0 comparatorだけを保存exp404
`likpf_mean_x1p0`へ訂正して同じpreflightを再実行することを承認範囲へ追加する。
tolerance、self-GR、PF、seed、particle、well、実行量、科学gateは変更しない。
full 4 shard、inference、submissionは引き続き未承認とする。

## 2026-07-28 version 4 comparator dtype debug retry

version 3はexp404保存float32 artifactをfloat64として読み戻し、メモリ上の
exp429 float32予測と比較したため、同じfloat32 bit値のCSV serialization差を
technical差として誤検出した。保存artifactの正規dtype `float32`へ復元してから
bit-exact比較することだけをdebug修正し、同じpreflight量でversion 4を再試行する。
`1e-5 ft`上限、self-GR、PF、seed、particle、well、科学gateは変更しない。

## 2026-07-29 full 4 shard + merge承認

ユーザーの`full実行してください`により、preflight v4 technical PASSを前提に、
事前固定済み1 scientific variant / 773 wells / 98,944 seed-well /
49,472,000 particle startsを4 Kaggle CPU shardで実行し、保存4生成物のstrict mergeと
train-side科学gate判定までを承認範囲へ追加する。parent control、model、booster、
HMM、Beam、GPU rerunは0。inference、submission、same-OOF救済は含めない。

## 2026-07-29 full結果と停止判断

4 Kaggle CPU shardで固定1 variant / 773 wells / 98,944 seed-well /
49,472,000 particle startsを完走し、zero-PF strict merge version 2で
technical gateをPASSした。primary RMSEはcandidate `11.127406421`、
control `10.914522073`で`0.212884347 ft`悪化、改善foldは`1/5`。
by-well p95、worst well、fixed HMM/PF blendも固定上限をFAILしたため、
事前登録どおり`terminal_close_without_self_gr_or_pf_rescue_grid`とする。
inference、submission、same-OOF救済は実行しない。
