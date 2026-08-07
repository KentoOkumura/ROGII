# 要件

## 目的と仮説

`KAGGLE_DIRECTION.md` の backlog `two_dimensional_formation_gradient_transition` を
`exp273_two_dimensional_formation_gradient_transition` として実装する。known prefix の
`S = TVT_input + Z` を `S(X,Y) = g_x X + g_y Y + c` でrobust fitし、
`delta TVT = g_x delta X + g_y delta Y - delta Z` を中心とする2D formation-gradientを
exp209 raw exact HMMの独立transition candidateとして監査する。

scalar `dS/dMD`だけでは捉えにくいazimuth変化を2D gradientが補い、turning wellを中心に
保存済みscalar controlと相補的なpathを作る、という仮説を検証する。

## 親と変更点

- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 実装参照: `exp268_multi_scale_initial_rate_candidates`のself-contained shard/aggregate構成。
- 変更点: known-prefix Huber plane、5 gradient prototypes、transitionの2D surface moveだけを追加する。
- 固定点: exp209のgrid、residual-rate grammar、GR emission、calibration、prior、保存済みscalar control。

## 制約

- Route: `pf_beam`。
- 親は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` とし、保存済み
  scalar-rate HMMをcontrolとして再利用する。control HMMは再生成しない。
- plane fitは各well自身のknown prefix全体だけを使う。outer-valid/evaluation-tail true TVT、
  他wellのtarget-derived spatial neighbor、formation labelは使わない。
- planeはdeterministic Huber IRLSでfitし、gradient中心とweighted covarianceの2主軸に沿う
  `center / axis1_minus / axis1_plus / axis2_minus / axis2_plus` の5 prototypeを実行前に固定する。
- gradient HMMはexp209のjoint TVT-position/residual-rate state、grid、rate grammar、GR emission、
  calibration、sigma、start prior、bandを維持する。position transitionだけを
  `g_x delta X + g_y delta Y + residual_rate delta MD - delta Z` にする。
- known prefixのgeometryがrank不足、condition-number超過、azimuth coverage不足、fit非有限なら、
  5 prototypeは同一scalar-rate fallback pathへfail closedする。
- scalar controlを必ずbankに残す。candidate平均、blend、selector、hard switch、direct TVT補正、
  HMM+LightGBM、exp218 residual、raw-test inference、submissionは実装しない。
- row / 128・256・512-row block / whole-well oracleは、candidate固定後の診断に限定する。
- 5候補の全well実行はKaggle timeout回避のためtarget-free hashで2 well shardに分割する。
- active HMM variants 5、target wells 773、最大3,865 HMM well-runs、LightGBM config 0、fold 0、
  booster 0、GPU 0、raw-test inference 0、submission 0。Kaggle pushは今回の依頼範囲外。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `config.yaml` に `experiment.route=pf_beam`、親、Huber plane、geometry guard、5 prototypes、
  residual-rate HMM式、2 shard、実行コストがある。
- compact self-contained Jupytext notebookとして、2 shard generator、aggregate train、disabled inferenceを
  実装し、同一実験helper importと`__file__`を使わない。
- shardごとに5候補、gradient/prototype、plane/geometry diagnostics、residual initial rate、
  HMM std/loglik、fallback理由を保存する。
- aggregateはexp209 scalar controlと5新規候補をidでstrict alignし、expected 3,783,989 rows / 773 wellsを
  fail-closedに検証する。
- overall、distance、straight/turning、XY rank/condition、azimuth coverage、hidden-like、by-well/worst-well、
  fallback/duplicate、unique-best、row/block/whole-well oracleを出力する。
- true TVT/error/oracleがplane fit・prototype・HMM生成・fallbackへ流れないcontract testがある。
- Jupytext round-trip、py_compile、Ruff F821、固有test、`make validate-exp`が通る。
- no RNGだがNumba parallel差を許容するtrain-side auditであり、deterministic submission anchorとは扱わない。
- gzip生成物はraw SHAとdecompressed content SHAを分け、decompressed content SHAを主証拠にする。

## 成果物

- canonical aggregate、2 shard、disabled inferenceのJupytext `.py` / `.ipynb`。
- `config.yaml`、contract test、実行記録と結果テンプレート。
- Kaggle実行時のcandidate、geometry、duplicate、oracle、SHA監査artifact一式。

## 次のアクション

実装と静的検証後はKaggle CPU未実行の状態で停止する。push前にユーザー承認を得て、
5 variants / 2 shards / 最大3,865 HMM well-runs / 0 boostersとpackage metadataを再確認する。
