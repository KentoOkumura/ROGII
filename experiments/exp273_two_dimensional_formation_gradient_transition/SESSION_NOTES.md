# exp273_two_dimensional_formation_gradient_transition セッションノート

## 目的

backlog `two_dimensional_formation_gradient_transition`を、exp209 raw exact HMMのscalar-rate controlに
対する0-booster candidate-bank auditとして実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: completed / direct gradient candidates rejected
- CV / LB: scalar 11.938287、best gradient 12.169871 / 対象外
- inference / submission: disabled

## ユーザー承認済み設計

- 各wellのknown prefix全体だけでdeterministic Huber planeをfitする。
- gradient中心 + covariance ellipseの4軸点 = 5独立HMM候補。
- condition-number / XY rank / azimuth coverage guard不通過時はscalarへfallbackする。
- 保存済みexp209 scalar HMMをcontrolとして読み、再生成しない。
- GR emission、TVT grid、scale、selectorは変更しない。
- 2 CPU well shards、最大773 x 5 = 3,865 HMM well-runs。
- LightGBM config / fold / booster: 0 / 0 / 0。
- 初回実装時点では静的検証までとし、2026-07-18の追加承認後にKaggle CPU実行へ進んだ。

## 実装内容

- `S=TVT_input+Z`をX/Y中央値中心化designへdeterministic Huber IRLSでfitする。
- centered XYのSVDでrank ratio / condition number、doubled heading resultantでaxial azimuth coverageを計算する。
- weighted plane covarianceの固有vector符号をlargest-absolute component positiveへcanonicalizeし、
  `center / axis1_minus / axis1_plus / axis2_minus / axis2_plus`を固定する。
- HMM position transitionを
  `gx*dX + gy*dY + residual_rate*dMD - dZ`とし、exp209のjoint residual-rate state/dynamicsを維持する。
- residual initial rateはknown prefix末尾30行の
  `median((dTVT_input+dZ-gx*dX-gy*dY)/dMD)`だけで決める。
- generatorへ渡すhorizontalからtrue `TVT`をdropし、5 path凍結後だけdiagnostic targetをattachする。
- invalid geometryでは`g=(0,0)` HMMを1回だけ実行し5列へ複製する。aggregateで保存済みcontrolとのparityをhard guardする。
- aggregateはoverall / distance / geometry / hidden-like / by-well / duplicate / unique-best / oracleを保存する。
- oracle prediction、candidate mean、selector、raw-test prediction、submissionは保存しない。
- `exp268`の検証済みself-contained shard/aggregate構成だけを実装参照し、科学的な親はexp209とする。

## 実行前コスト契約

- active logical HMM variants: 5
- operational well shards: 2
- target wells / maximum HMM well-runs: 773 / 3,865
- invalid geometry: 1 scalar run/wellへ縮約
- LightGBM config / fold / booster: 0 / 0 / 0
- parent/control retraining: なし
- GPU / raw-test inference / submission: なし / なし / なし
- `execution.kaggle_push_approved=true`（2026-07-18ユーザー承認）

## リークガード

- plane、geometry guard、prototype、residual rateは同じwellのknown `TVT_input` rowsだけを使う。
- outer-valid/evaluation-tail target、他well target、target-derived neighbor、formation labelは使わない。
- true TVTはcandidate path固定後のtrain-side diagnosticsだけで使う。
- prototype数、guard threshold、oracle block、candidate bankをtarget metric確認前にconfigで固定する。

## 再現性

- `docs/06_reproducibility.md`を確認済み。
- HMM、Huber IRLS、SVD/eigh、prototype生成はno RNG。
- well shardだけを`sha256("exp273::well_shard::<well>") % 2`で決める。
- eigenvector signをcanonicalizeする。
- exp209 inputのdecompressed SHAをhard guardし、shard raw/decompressed SHA、schema SHA、
  aggregate candidate-array content SHAを記録する。
- Numba parallel / LAPACKの微小差を許容するためdeterministic submission anchorとは扱わない。

## コマンドログ

### 2026-07-17 作成

    make new-steering EXP=exp273_two_dimensional_formation_gradient_transition
    make new-exp EXP=exp273_two_dimensional_formation_gradient_transition SOURCE=experiments/exp268_multi_scale_initial_rate_candidates

- steering: `.steering/20260717-exp273-two-dimensional-formation-gradient-transition/`
- experiment: `experiments/exp273_two_dimensional_formation_gradient_transition/`
- `kaggle-review-exp` skillと`docs/06_reproducibility.md`を確認した。

## 静的検証

2026-07-17に以下を完了した。実データのローカルnotebook実行は行わず、初回full実行はKaggle CPUを正とする。

- exp273 contract tests: `9 passed`
- repository tests: `119 passed`
- 4 Jupytext sourcesの`.ipynb`生成と`--to ipynb --test`: PASS
- 4 sourcesの`py_compile`: PASS
- 4 sourcesのRuff `F821`: PASS
- `make validate-exp EXP=exp273_two_dimensional_formation_gradient_transition`: PASS
- `make validate-template`: PASS
- `review_exp_docs.py exp273 --root .`で証拠カテゴリを確認し、steering / READMEの成果物と次アクションを補完した。

軌跡層別の`turning_azimuth_coverage=0.10`とfallback-control parityの
`atol=1e-5 ft`は暗黙定数にせず`config.yaml`へ固定した。

## 実行承認とKaggle実行

### 2026-07-18 Kaggle CPU shard v1実行承認

- ユーザーの「実行してください」をKaggle CPU shard 0/1のpush承認として記録した。
- 実行前契約を再確認: 5 logical variants、2 shards、最大3,865 HMM well-runs、
  LightGBM config / fold / boosterは0 / 0 / 0、parent/control再生成なし、GPUなし。
- `execution.kaggle_push_approved=true`へ変更し、未承認configではnotebookがfail-fastするguardを維持した。
- 承認反映後のcontract 9 tests、Jupytext、py_compile、Ruff F821、strict experiment/template validationはPASS。
- shard 0/1をprivate CPU notebookとしてpushする。aggregateは両shardのcoverageとSHA確認まで実行しない。

### 2026-07-18 shard package preflight

- shard 0: `kentookumura/exp273-2d-formation-gradient-shard0`
- shard 1: `kentookumura/exp273-2d-formation-gradient-shard1`
- title slug、kernel id、`run_on_push=true`、private、CPU、internet false、competition sourceを照合した。
- shard notebookのkernel sourceは0。raw train dataはcompetition sourceから読み、外部controlを使わない。
- package内config SHAとbootstrap manifestのconfig SHAは両shardとも
  `dc7c60a571eebc78479dfc9fa9f4dffab3420cd62e8ce966fe24605f5f0e2708`で一致した。
- `RUN_KIND_OVERRIDE`はshard 0 / shard 1へ固定され、package内`kaggle_push_approved=true`を確認した。

### 2026-07-18 Kaggle CPU shard v1 push

    kaggle kernels push -p experiments/exp273_two_dimensional_formation_gradient_transition/kaggle/train_variant0
    kaggle kernels push -p experiments/exp273_two_dimensional_formation_gradient_transition/kaggle/train_variant1

- shard 0: `kentookumura/exp273-2d-formation-gradient-shard0`, version 1, id_no `127705719`
- shard 1: `kentookumura/exp273-2d-formation-gradient-shard1`, version 1, id_no `127705716`
- 両pushは成功し、`kaggle kernels pull -m`で同一kernel id、private、CPU、internet false、
  competition source、kernel source 0を再確認した。
- push直後の`kaggle kernels status`は両方`KernelWorkerStatus.RUNNING`。
- 通常logsは実行中に空になり得るため、空だけで失敗や再pushと判断しない。同じkernel idのversion 1を維持する。

### 2026-07-18 shard v1完了監査

- shard 0: COMPLETE、396 wells、1,910,995 rows、elapsed `9,359.141`秒、gradient valid / fallback `62 / 334`。
- shard 1: COMPLETE、377 wells、1,872,994 rows、elapsed `8,510.024`秒、gradient valid / fallback `49 / 328`。
- 合計773 wells / 3,783,989 rows、well overlap 0、全well status `ok`、stable SHA shard assignment一致。
- gradient valid / fallbackは合計`111 / 662 wells`。coverageが低いこと自体は事前guardの結果であり、
  thresholdを変更せずaggregateでdirect差とheadroomを監査する。
- shard 0 prediction content SHA: `0ee38a949f42ec14e313ef65e6148ed4dc4a13ef4bb472aada84adf55ba35def`
- shard 1 prediction content SHA: `c2525286a818d3f557013ebc5ed252b38db3cb34727db744e650a363b318af09`
- shard 0 rows raw / decompressed SHA:
  `acb943b7b91e723643dddb2259d492c8af9a81fb35bbd3187319633f473eb2c1` /
  `347b87554261cb904bc7f98d6d1eb64ed5aaa46f15720011b94797d814aeac97`
- shard 1 rows raw / decompressed SHA:
  `ae78cfc15e8bb0a1191f0300261723a3b379c0f9d4fb68e0b17b81ddae7d9e48` /
  `98939d080e4b5bfa3ab93631601d496946b0e9ddcc8aac4bb10a430eeda7d407`
- schema SHAは両方`779024b3e784c6db345bdb27a6e3e2c6dfd8b60614fba2daf490327842e70548`で一致。
- by-well / input-manifest SHAもsummary記載値とdownload実ファイルが一致した。

### 2026-07-18 aggregate package preflight

- aggregate: `kentookumura/exp273-2d-formation-gradient-aggregate`
- private CPU、internet false、run-on-push true、competition sourceを確認した。
- kernel sourcesはexp209 control、exp115 hidden-like、exp273 shard 0/1の4件だけ。
- aggregate前の再現性ガードとして、確認済みshard別rows / wells / raw SHA / decompressed SHAを
  `config.yaml`へ固定し、`load_shards`でfail-fastするよう追加した。shard生成ロジックは変更していない。
- 変更後のcontract 9 tests、Jupytext round-trip、py_compile、Ruff F821、strict validationはPASS。
- package内config SHAとbootstrap manifest SHAは
  `3064941b722b88f63abcc44b629c9a7f631aaaf56e95c0aa93b6f006d4c4e66e`で一致した。
- `RUN_KIND_OVERRIDE=aggregate`、5 variants / 0 configs / 0 folds / 0 boosters、
  parent/control再生成なし、inference/submissionなしを再確認した。

### 2026-07-18 Kaggle aggregate version 1 push

    kaggle kernels push -p experiments/exp273_two_dimensional_formation_gradient_transition/kaggle/train

- kernel: `kentookumura/exp273-2d-formation-gradient-aggregate`
- version 1、id_no `127731254`、push成功。
- `kaggle kernels pull -m`でprivate CPU、internet false、competition source、4 kernel sourcesを確認した。
- push直後のstatusは`KernelWorkerStatus.RUNNING`。

### 2026-07-18 Kaggle aggregate version 1完了監査

- status `COMPLETE`、runtime `161.445`秒、3,783,989 rows / 773 wells。
- input coverage、well overlap 0、shard別rows / wells / raw/decompressed SHA、保存済みexp209 control parityはPASS。
- aggregate summaryに記録された10 CSVのSHAをdownload実ファイルで再計算し、全件一致した。
- scalar control RMSEは`11.938287`。gradient direct 5候補は全て悪化し、best axis1-minusも
  `12.169871`、delta `+0.231584 ft`だった。
- best gradient deltaは1000+で`+0.263798 ft`、hidden-like spatial / typewell-purgedで
  `+0.318575 / +0.322224 ft`、geometry-valid rowsで`+1.687249 ft`、turning rowsで
  `+0.981277 ft`。worst-well `dd7d638e`は`+36.118726 ft`悪化した。
- scalarを含むoracle deltaはrow / block128 / block256 / block512 / whole-wellで
  `-0.188841 / -0.188599 / -0.188204 / -0.187164 / -0.178637 ft`。
- aggregate prediction content SHAは
  `87e59647018cf69f187d293e462afa737334c1d90da6e56ed385a85ecae0b79d`。
- direct gradient仮説は棄却する。candidate平均、selector、raw-test inference、submissionは実行しない。

## 次のアクション

低-中優先の0-booster `formation_gradient_prefix_stability_risk_readout_on_exp273`だけを候補に残す。
full / last-256 / last-512 known-prefix planeのgradient角度・大きさ・fit残差が、direct候補の悪化を
target-freeに5 foldsで再現して説明できるかを確認する。HMM再実行、valid targetによるthreshold fit、
hard gate、inference、submissionは行わず、5 foldsで再現しなければbranchを閉じる。
