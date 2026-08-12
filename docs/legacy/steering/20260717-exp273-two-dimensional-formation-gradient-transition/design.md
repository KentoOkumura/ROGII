# 設計

## アプローチ

各wellのknown prefixで `surface = TVT_input + Z` を作り、X/Yを中央値中心化したdesignに対して
deterministic Huber IRLSを行う。fit前にcentered XYのSVDからrank ratioとcondition numberを、
known-prefix stepのaxial headingからazimuth coverageを計算する。`min_points=64`、
`max_condition_number=20`、`min_rank_ratio=0.05`、`min_azimuth_coverage=0.02`を事前固定し、
1条件でも不通過ならgradientを使わない。

valid fitではIRLS最終weightからslope covarianceを作り、gradient中心とcovarianceの2固有軸上の
`+-1 sigma`点を合わせた5 prototypeを固定する。各prototypeではknown prefix末尾30行の
`delta surface - g_x delta X - g_y delta Y` を `delta MD` で割ったmedianをresidual initial rateとする。
exp209 HMM kernelのposition transitionを
`mu = g_x delta X + g_y delta Y + residual_rate_state delta MD - delta Z`
へ変更し、GR emissionが候補を支持するときだけgradientが効く弱いtransition hypothesisにする。
rate state/dynamicsと全HMM hyperparameterはexp209から固定する。

invalid fitではscalar `g=(0,0)` HMMをwellごとに1回だけ生成し、5 prototype列へ同じpathを複製する。
aggregateでは保存済みexp209 scalar pathとのfallback parity、candidate重複率、direct差、geometry層別、
hidden-like、worst-well、row/block/whole-well oracleを監査する。oracle prediction、candidate mean、
selector、raw-test pathは保存しない。

最大5 variants x 773 wellsを1 notebookに載せないため、wellを
`sha256("exp273::well_shard::<well>") % 2`で2 shardに分ける。`train_variant0/1`がcandidateを生成し、
正規`train` notebookが2 shard、exp209 control、exp115 hidden-like assignmentを統合する。

## 実験範囲

- 対象実験: `exp273_two_dimensional_formation_gradient_transition`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 実装参照: `exp268_multi_scale_initial_rate_candidates` のself-contained shard/aggregate構成。
- 変更する変数: plane-fit gradientと5 fixed ellipse prototype、position transitionのsurface move。
- 固定する変数: HMM TVT grid、41 residual-rate states、rate span/dynamics、momentum、GR emission、
  sigma mode、calibration、start position/rate prior、band、score rows。
- 比較candidate: 保存済みexp209 scalar control、5 gradient candidates。
- 禁止: outer-train/shared plane、他well target、formation label、candidate blend/mean、selector、
  emission/scale調整、direct correction、HMM+ML、inference、submission。

## 再現性設計

- seed policy: HMM/Huber/prototypeはno RNG。well shardだけをstable SHA256で決める。
- stochastic 処理の有無: なし。SVD/eighの固有vector符号はcanonical signへ正規化する。
- PF/Beam / likelihood-PF / seed bagging の有無: exact HMM 5 candidate。PF、likelihood-PF、seed baggingなし。
- 並列処理と乱数の関係: joblib thread 2 x Numba thread 2。乱数を使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false、2 shard。
- train cache / test feature regeneration の SHA 記録方針: exp209 control decompressed SHAをhard guardし、
  shard gzipのraw/decompressed SHA、schema SHA、aggregate candidate array content SHAを記録する。testは生成しない。
- model manifest / prediction / submission SHA 記録方針: model/submissionは対象外。train-side candidate
  array content SHAだけをprediction証拠として記録する。
- Kaggle package bootstrap 確認方針: `train_variant0/1`はkernel sourceなし、正規`train`はexp209、exp115、
  2 shard kernel sourceを持つ。prepare時にmetadataとbootstrap config/sourceを照合する。

## リスク

- リークリスク: candidate生成へ渡すhorizontal frameからunknown suffix true `TVT`をdropする。
  `TVT_input`がfiniteなknown prefixだけでplane、guard、prototype、residual rateを固定し、その後だけtrue TVTを
  diagnosticsへattachする。source inspection testで順序を固定する。
- CV/LB 不一致リスク: official evaluation-tail形状のtrain-side candidate auditであり、raw-test再生成や
  Public LBを主張しない。oracle headroomはdeployable scoreと分ける。
- ランタイム/メモリリスク: 最大5 x 773 HMM well-runs。2 shardに分け、invalid fitではscalar HMM 1回を
  5列へ複製し、posterior tensorは保存しない。Kaggle push前に実測見積もりを再確認する。
- 再現性リスク: no RNGだがNumba parallel reduction、LAPACK、gzip metadataでbyte差が起こり得る。
  deterministic anchorとせず、canonical eigenvector sign、decompressed SHA、metric toleranceを主証拠にする。

## 成果物

- `train_variant0/1`: target-free hash shardごとの5 candidateとplane/HMM診断。
- canonical `train`: 2 shardと保存済みcontrolのstrict aggregate audit。
- disabled `inference`: raw-test prediction、hard switch、submissionをfail-closedにするguard。
- config、contract test、README、SESSION_NOTES、result、metrics。

## 次のアクション

静的検証完了後はKaggle CPU未実行で停止する。別途push承認が得られた場合だけ、
shard 0/1を実行し、coverageとSHAを確認してからaggregateを1回実行する。
