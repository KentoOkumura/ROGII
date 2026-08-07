# exp429_self_gr_weak_boost_likelihood_pf_ablation 結果

## 状態

Kaggle CPU full 4 shardとstrict merge version 2を完了した。
technical gateはPASS、事前登録scientific gateはFAILし、
`terminal_close_without_self_gr_or_pf_rescue_grid`で閉鎖した。
inferenceとsubmissionは未実行である。

## 仮説

exp223固定self-GR weak boostをlikelihood-PFのparticle observation likelihoodへ
直接加えると、PF内部のweight / ESS / resampling / trajectoryを通じて
fixed temperature-5 predictionを安全に改善できる。

## 設定

- 親: `exp417_scale5_seed_aggregation_promotion_audit`
- kernel control: `exp072_exp063_full_replay_feature_cache`
- self-GR式参照: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- candidate: `likpf_scale5_selfgr_boost_only_a070_c100`
- 変更: particle log-likelihoodへ固定self-GR nonnegative boostを追加
- 固定: exp072 x1.0 PF、500 particles、128 seeds、temperature 5
- 検証: all-well 3,783,989 rows / 773 wells / 5 reporting folds
- primary metric: fixed scale5 controlに対するpooled RMSE gain
- シード: 42

## 事前登録gate

- scale5 gain `>=0.05 ft`
- 4/5 folds改善
- arithmetic mean、raw observed/missing、high missing、1000+、hidden-like 2面、
  fixed HMM/PF blendを非悪化
- by-well delta p95 `<=0.0 ft`
- worst-well regression `<=0.25 ft`

## 再現性

- deterministic anchor: いいえ。preflight 1回だけで独立rerunなし
- seed policy: exp072互換stable SHA256 per-well + seed index
- preflight kernel: `kentookumura/exp429-self-gr-weak-boost-likpf-ablation-train`
- preflight kernel id_no / latest executed version: `128934717 / 4`
- full shard kernels: `...-shard0`--`...-shard3`、各version 1
- merge kernel: `kentookumura/exp429-self-gr-weak-boost-likpf-merge` version 2
- prediction logical SHA:
  `d7677deb40526274853178290d316efcc0b1bafe629d13c669f50ac062689ff0`
- surface logical SHA:
  `2bfc1a996c4f7ad01a48ef34ba333f56907e5f732c3f599d77cc4f27c58a2ba7`
- artifact manifest SHA:
  `3620944f8ab0c6cf0b85c9fd7c11a9ed07897965324c774e43aa528dedbf694e`
- model SHA / manifest SHA: 非該当
- submission SHA: 非該当
- rerun result: version 1はasset path不一致でPF実行前に停止。version 2でpathだけ修正

## 実装検証

- compact self-contained train / fail-closed inference: 実装済み
- 正規train / inference notebook: 採用済み
- target-free固定4-well asset SHA:
  `24358da10d2d853b25b4eeb68446c005e34364c78d7f0185af4ceb601effd876`
- full shardはSHA固定済みpreflight technical PASSなしでは開始不可
- fixed exp209 HMM + scale5 control RMSE: `10.084909679560383`
- 専用contract test: merge manifest dtype regression追加後`13 passed`
- 構文 / Ruff F821 / Jupytext round-trip / strict experiment validation: PASS
- Kaggle package / preflight: 実施済み
- full run: 4 shard + zero-PF strict mergeを完了、scientific gate FAILで閉鎖

## Preflight結果

- runtime to gate: `634.395 sec`
- variants / wells / PF well-runs: `2 / 4 / 8`
- seed-well trajectories / particle starts: `1,024 / 512,000`
- alpha0対保存exp072 absolute再構成のrow最大差:
  `0.00035199999911128543 ft`（上限`0.00001 ft`、FAIL）
- alpha0対保存exp404 x1.0 arithmetic predictionのrow最大差:
  `0.0 ft`（18,055行、4 wells、bit-exact）
- candidate positive-quality rows: `12,239`
- candidate positive boost applications: `777,858,990`
- summary file SHA:
  `715ce13b5f918184017678e1087b0ebf5c3607b262ed2f73caa8d1408adf0dd0`

実行量、finite出力、self-GR activationは契約どおりで、唯一のFAILは
alpha0を保存exp072の`last_known_tvt + likpf_mean_d`へ戻したabsolute値との
row最大差である。alpha0自体は同じPF replayを保存したexp404 x1.0 arithmetic
predictionと全18,055行でbit-exactだった。

## 解釈

fixed self-GR boostをPF内部へ入れると、primary temperature-5 predictionは
controlより`0.212884 ft`悪化した。fold 0だけは`0.320141 ft`改善したが、
fold 1--4は悪化し、特にfold 4は`0.768297 ft`悪化した。raw GR observed、
long-tail 1000+、hidden-like 2面、by-well tail、fixed HMM/PF blendでも悪化した。
同一well由来の局所boostが一部regimeでは有効でも、PFのweight/resamplingを通じて
誤ったbasinを持続的に強め、well-tailを増幅したと解釈する。

## 次

本実験では追加実行しない。alpha、clip、window、top-k、temperature、GR sigma、
particle、seed、transition、resampling、blend、selectorによるsame-OOF救済は
禁止契約どおり行わない。既存の低優先
`self_gr_quality_addonly_features_on_exp092`だけを、直接PF boostではなく
target-free quality/risk特徴として不均一効果を検証する独立候補に残す。
raw-test inferenceとsubmissionへ進まない。

## Preflight version 4結果

- Kaggle status / technical gate: `COMPLETE / PASS`
- notebook終了ログ時刻: `472.457 sec`
- variants / wells / PF well-runs: `2 / 4 / 8`
- seed-well trajectories / particle starts: `1,024 / 512,000`
- alpha0 comparator dtype: `float32`
- alpha0 bit-exact rows: `18,055 / 18,055`
- alpha0最大差: `0.0 ft`（上限`0.00001 ft`）
- candidate positive-quality rows / positive boost applications:
  `12,239 / 777,858,990`
- prediction logical SHA:
  `997713bd08559411135bd48e9a19594fe4141885c08da3fd66b3070e96b009f3`
- surface manifest logical SHA:
  `6c4876f94fe94ec63da95b6b5f270cdc519bc2f50d3a9a992e6200cc46ac0c35`
- summary / audit / prediction gzip file SHA:
  `2e9f066fd80813862d1e232ad66fd965020e63a3bddd976ef68303e79fe0d190` /
  `eb29bc5506e0b1ddb65cb1a4909fec52939fe2f7d3d35eb9471d41b9fd5fc65a` /
  `3f2b8acfa07b027e369fb79ab7e13741dd9b9cab514d163999165f345cf6af2d`

これはtechnical preflightの履歴である。後続full 4 shard + merge version 2で
scientific gateまで評価済み。

## Full 4 shard + merge version 2結果

- Kaggle status: 4 shardおよびmergeすべて`COMPLETE`
- coverage: `3,783,989 rows / 773 wells / 5 folds`
- scientific variants / PF well-runs: `1 / 773`
- seed-well trajectories / particle starts: `98,944 / 49,472,000`
- positive self-GR valid rows / boost applications:
  `2,096,654 / 131,102,661,385`
- technical gate: `PASS`
- primary candidate / control RMSE:
  `11.127406421 / 10.914522073`
- primary gain: `-0.212884347 ft`（必要`>=+0.05 ft`）
- improved folds: `1 / 5`（必要`>=4 / 5`）
- arithmetic secondary delta: `-0.023416828 ft`（PASS）
- by-well delta p95: `+0.770627049 ft`（上限`0.0 ft`、FAIL）
- worst-well regression: `+34.862601957 ft`（上限`0.25 ft`、FAIL）
- fixed HMM/LikPF 50:50 regression: `+0.070319209 ft`（上限`0.0 ft`、FAIL）
- scientific gate: `FAIL`
- decision: `terminal_close_without_self_gr_or_pf_rescue_grid`
- merge runtime / prediction freeze:
  `222.187 / 115.685 sec`
- 保存先:
  `artifacts/kaggle_merge_v2/`

merge version 1は773件のmanifest値が完全一致していたが、保存CSVの
`shard_index int64`と期待`int8`のdtypeだけでstrict equalityをFAILした。
version 2は読込dtypeを`int8`へ復元し、追加回帰testと取得済み4 manifestの
`773 x 5` strict equalityを通してから実行した。科学契約とshard予測は変更していない。
