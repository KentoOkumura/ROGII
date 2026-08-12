# exp430_huber_seed_evidence_reaggregation セッションノート

## 目的

exp404 互換の固定 128 seed PF 軌跡を一度だけ再生し、Gaussian と
Huber `delta=1.345` の seed evidence 集約をcommon-trajectory条件で比較する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_gate_failed_terminal_close`
- CV: Huber `12.992939553`、matched Gaussian `12.999103257`
- LB: なし
- Kaggle package / push / run: fixed4 preflight v2、full 4 shard、merge v1完了
- merge: technical PASS / scientific FAIL
- inference / submission: 未承認、無効

## 2026-07-29 full 3+1実行承認

- ユーザーの「exp430 fullを3+1で実行してください」により、technical
  preflight PASS済みの固定full 4 shardのpackage、push、Kaggle CPU実行を
  承認済みとして記録した。
- 実行量はscientific variant `1`、PF well-runs `773`、
  seed-well trajectories `98,944`、particle starts `49,472,000`、
  CPU shards `4`。
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU、
  parent independent full rerunはすべて`0`。mergeもPF run `0`。
- 先行して承認済みのexp429 strict merge用にCPU枠を1つ残すため、
  exp430 shard 0--2を先行投入し、exp429 merge完了後にshard 3を投入する。
- 本承認にexp430 merge、inference、submissionは含めない。4 shard完了後の
  strict mergeと科学gateは、4 summary SHAを固定して別途承認を得る。

## 2026-07-29 full shard 0--2 push

- shard 0--3のprivate CPU / internet off / run-on-push packageを生成した。
  全packageでbootstrap内configと配置configのbyte一致、preflight v2 kernel
  input、stage / shard index、metadataを照合した。
- 共通scientific contract SHA:
  `a7bf622ea2357804896a62ba0adb2143b4aa39164ae2751c1146e284d9fe6e10`
- 共通execution code SHA:
  `0dff10b802d091777a4590a59e450337e7b2542ca6472874e92e6077774cd42f`
- shard 0--3 package config SHA:
  `93ad6fcb91899acc60a7ec0f8b5e38e975fd6b9753e2bdbc3df9d57bd6ee8bab` /
  `66644d8aca2a689201cee185a9bf727c2506d927d1a50423f51971073c14280d` /
  `f33c54556b6782d12c16e692b9a5d3a6987ebdd0ae4aa5cd4d7db2f578608a53` /
  `a05df57acfc8589981b11b82f1b6994b1d4a44ad00bd61647ba4175424841129`
- shard 0 / 1 / 2をversion 1としてpushした。Kaggle id_noは
  `129042140 / 129042150 / 129042163`。push後pullでprivate、CPU、
  GPU/internet off、固定inputを確認し、3本とも`RUNNING`。
- shard 3 packageは作成・検証済みだが、exp429 strict merge用の予約枠を
  維持するため未push。

## 2026-07-29 full shard 3 push

- exp429 strict merge version 2の`COMPLETE`を確認した後、予約していたCPU枠へ
  shard 3をversion 1としてpushした。
- Kaggle id_noは`129046380`。push後pullでprivate、CPU、GPU/internet off、
  preflight v2を含む固定inputを確認し、状態は`RUNNING`。
- これでfull 4 shardはすべてKaggle上で実行開始済み。exp430 merge、
  inference、submissionは引き続き未承認で、4 shard完了後も自動実行しない。

## 2026-07-29 full shard 0--2完了

- shard 0 / 1 / 2はversion 1 `COMPLETE`。summary JSONだけを取得し、
  stage、実行量、preflight SHA、scientific contract SHA、truth-lateを監査した。
- rows / wells:
  `946,128 / 193`、`946,017 / 193`、`946,112 / 193`
- runtime seconds:
  `2,428.812 / 2,655.740 / 2,690.160`
- PF well-runs / seed-well trajectories / particle startsは各
  `193 / 24,704 / 12,352,000`。
- summary SHA:
  `aac60fd98cc98d242e15cb1e288241ce948a99ea64eef0362a391feb34288a74` /
  `d8176eafaa04ddc86f569778aa7dbd5697bb01e71d9deff185e09975a4a604d9` /
  `60a5ea2043d35c6f9241357c4791b6c92839051a9dbe8ab540cf8dfd3fe4bff0`
- 3本ともscientific contract SHA
  `a7bf622ea2357804896a62ba0adb2143b4aa39164ae2751c1146e284d9fe6e10`
  とpreflight summary SHA
  `3a34add3b77abc08add3fb8c37c4fadb8a4f249577c13ca0116552e417c6eddc`
  が一致した。
- suffix TVT / fold / hidden-like role / errorのfreeze前アクセスは3本とも
  すべて`0`。shard 3は引き続き`RUNNING`。

## 2026-07-29 full 4 shard完了・merge入力固定

- shard 3 version 1、Kaggle id_no `129046380`の`COMPLETE`を確認し、
  summary JSONだけを取得して監査した。
- shard 3は`945,732` rows、`194` wells、`194` PF well-runs、
  `24,832` seed-well trajectories、`12,416,000` particle starts、
  runtime `2,592.977 s`。
- shard 3 summary raw SHA:
  `66d78b737952f922c7de71c02b4bccafab901293dab1453d5706c7d877ad074c`
- shard 3 trajectory / prediction / evidence logical SHA:
  `9e4d1033fe6aebd17ba2da604b6f180a92d107d1fb5ea8fdb12115b8813c43fe` /
  `f87ce98a1b98f72306235285be04489608a2edbea9d70b938ea8097646f25bce` /
  `095999bef12639858b5e6afd12fd7c391dbe65d55e123f531d629279d7097b6c`
- 4 shard合計は`3,783,989` rows、`773` wells、`773` PF well-runs、
  `98,944` seed-well trajectories、`49,472,000` particle startsで、
  固定実行量契約と完全一致した。
- 4本ともstage/status、scientific contract SHA、preflight summary SHA、
  scientific variant `1`、shared-bank readout `2`、parent full rerun `0`、
  LightGBM/fold/booster/model/HMM/Beam/GPU `0`を確認した。
- suffix TVT / fold / hidden-like role / errorのfreeze前アクセスは4本とも
  すべて`0`。weight sum最大絶対誤差も全て`1e-12`以下。
- shard index順のsummary SHAを`data.full_shards.expected_summary_sha256`へ、
  4 Kaggle Notebook rootを`data.full_shards.candidates`へ固定した。
- `execution.selected_stage=null`、`runtime.kaggle.run_on_push=false`、
  `execution.merge_approved=false`を維持する。strict mergeはPF run `0`の
  truth-late CPU集約だが、別承認までpackage/push/runしない。

## 2026-07-29 strict merge実行承認

- ユーザーの「exp430 mergeを実行してください」により、4 shardの固定summary
  SHAを入力とするstrict truth-late mergeのpackage、push、Kaggle CPU実行を
  承認済みとして記録した。
- scientific variantはHuber `delta=1.345`の`1`候補。matched Gaussian、
  arithmetic mean、保存exp404は同じ凍結予測から比較するreadoutであり、
  新しいtrajectoryやcontrolを生成しない。
- mergeのPF well-run / seed-well trajectory / particle startは
  `0 / 0 / 0`。LightGBM config / trained fold / booster / model /
  HMM / Beam / GPU、親full control再実行もすべて`0`。
- 既存4 shardの集約対象は`3,783,989` rows、`773` wells、reporting fold `5`。
  truth attachment後にoverall、fold、fixed scopes、paired by-well tailの
  固定AND gateを一度だけ判定する。
- Kaggle kernelは
  `kentookumura/exp430-huber-seed-evidence-reaggregation-merge`、
  private CPU、internet off、run-on-pushとする。
- 本承認にinference、raw-test PF、submissionは含めない。科学gateがPASSしても
  自動推論・自動提出は行わない。

## 2026-07-29 strict merge version 1 push

- merge_v1 package config / execution code / notebook SHA:
  `07a67bea0b04362da5738a8e605b92858fb653c614ceab2dbe5d5fcc8304899c` /
  `0dff10b802d091777a4590a59e450337e7b2542ca6472874e92e6077774cd42f` /
  `b0be49f9a1b3760ad6e8a5edb1579e1302630bc42374c37cb1cb8eef2c514759`
- 配置configとbootstrap内configのbyte一致、`selected_stage=merge`、
  `merge_approved=true`、4 candidate roots、4 expected summary SHAを照合した。
- metadataはcanonical kernel id/title、private、CPU、internet off、
  run-on-push、exp072/exp226/exp115/preflight/full shard 0--3の8 kernel inputs、
  保存exp404 dataset inputを照合した。
- version `1`をpush。Kaggle id_noは`129051025`、確認時は`RUNNING`。
- push後の正規config/packageは`selected_stage=null`、
  `runtime.kaggle.run_on_push=false`へ戻し、同versionの誤再実行を防ぐ。

## 2026-07-29 strict merge version 1結果

- Kaggle kernel version `1`、id_no `129051025`は`COMPLETE`。
  merge runtimeは`397.41868472099304 s`。
- 4 shard summary SHA、input SHA、rows `3,783,989`、wells `773`、
  5 folds、実行量、finite coverage `1.0`、shared trajectory identity、
  weight sum、parent / arithmetic parity、truth-lateのtechnical 11 checksは
  すべてPASSした。
- merge前のtruth / fold / hidden-like role / error accessはすべて`0`。
  truth attachment後もprediction logical SHA
  `39cb4f03561e75eefd5047fdaaeede4361cd95697b3ab297dcf5716eb4021213`
  を再照合した。
- Huber RMSEは`12.992939553297264`、matched trajectory-residual Gaussianは
  `12.999103256822222`。改善は`0.006163703524958208 ft`で固定
  `0.10 ft` gateに届かなかった。
- matched Gaussian比はnonworse fold `4 / 5`だけPASS。shallow
  `+0.005483 ft`、raw GR missing `+0.003271 ft`、high missingness
  `+0.006872 ft`、roughness low `+0.008403 ft`、hidden-like spatial /
  typewell-purged `+0.032498 / +0.032884 ft`が悪化し、all-scope gateをFAILした。
- paired-well squared-error delta p95は`+0.4642216562912155`、
  worst paired-well RMSE deltaはwell `c3957531`の
  `+2.6586746567715913 ft`でtail gateをFAILした。
- 保存exp404 temperature-5 RMSE `10.914522073423171`に対して
  `2.0784174798740924 ft`悪化し、nonworse fold `0 / 5`、全fixed scopeで悪化。
  arithmetic mean `11.59489788373621`にも`1.398041669561053 ft`悪化した。
- technical gate `true`、scientific gate `false`、最終decisionは
  `huber_seed_evidence_reaggregation_rejected_close_without_rescue`。
- summary / promotion gate / artifact manifest raw SHA:
  `1e2bbc0b52f0de8b12a9c49d9177883e92cb95c97f01cd68b646d9f00e9e2870` /
  `ce2993d20b599fa7203aa28e86f13ddce64a418d334f86678460011e6cff25ad` /
  `ae07934cc75a34034ee8c53b5a89f0c85b46db4d7824e484de291a16efab460e`
- evidence / global trajectory manifest logical SHA:
  `10199e3be8a80d1a6b7b6e4f6b81e068244f3c5f2ed3338c7b970e63b2c0daee` /
  `ff6712863caf189f4dbaeaf730d57f6c0210fbe9fcd957e2f10f4e202fc05bde`
- 小さいsummary、gate、metrics、scope、by-well、manifest、identityだけを
  `/tmp/exp430-merge-v1-output`へ取得してSHAとworst wellを実ファイル監査した。
  Kaggle output archive全体とtrajectory bankは取得していない。
- 失敗原因はHuber delta単体より、trajectory-residual evidence familyが
  保存exp404 parent marginal evidenceと異なるseed順位・集中度を作る目的関数差の
  可能性が高い。Huberはmatched Gaussianをわずかに改善したが、parent marginalとの
  大差とtailを解消しなかった。
- 事前登録どおりdelta / temperature / clip / scale / particle / seed /
  filtering尤度、well/row gate、affine / AR1 / self-GR / reinjectionの
  same-OOF救済は行わない。inference、raw-test PF、submissionなしで閉鎖する。

## 2026-07-29 technical preflight 実行承認

- ユーザーの「実行してください」を、直前に提示した fixed 4-well Kaggle
  CPU technical preflight の package / push / run 承認として記録した。
- 実行対象は PF scientific variant `1`、well `4`、PF well-run `4`、
  seed-well trajectory `512`、particle start `256,000`。
- LightGBM config / trained fold / booster / model / GPU は
  `0 / 0 / 0 / 0 / 0`。
- 親full controlの独立再学習・再実行は`0`。同じtrajectory bankから
  Gaussian matched controlとHuber candidateを再集約する。
- full 4 shard、merge、inference、submissionは承認範囲外であり、
  preflight PASS後も自動では実行しない。

## 2026-07-28 設計記録

- PF 親を exp404 x1.0、比較根拠を exp417、Huber 式を exp389 に固定した。
- PF scientific variant は1。Gaussian/Huber の2 readoutは同じtrajectory bankを使う。
- full実行量を773 well-runs、98,944 seed-well trajectories、
  49,472,000 particle startsに固定した。
- LightGBM config / fold / booster / model / HMM / Beam / GPUは
  `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- 親実験の独立full rerunは0。一度のreplayがmatched controlも供給する。

## 2026-07-28 実装

- ユーザーの「exp430を実装してください」を実装承認として記録した。
  Kaggle package、push、preflight/full/merge実行、inference、submissionは
  承認範囲に含めない。
- compact self-contained train / inferenceをJupytext percent形式で作成し、
  正規train / inference notebookへ採用した。
- trainは次の3段階を実装した。
  1. fixed SHA-first 4 well technical preflight
  2. deterministic LPT assignmentによるfull shard 0..3
  3. 4 shard SHA固定後のtruth-late mergeとpromotion gate
- per-seed trajectoryはfloat64 `.npy` memmapへ保存し、row index、seed score、
  PF audit、raw/logical SHAを確定してからGaussian/Huber evidenceを計算する。
- 親exp404のmarginal Gaussian scoreもaudit readoutとして保存し、
  保存exp404 T=5との技術parityを確認できるようにした。科学候補はHuber 1本だけ。
- arithmetic replayの技術comparatorは保存exp404 `likpf_mean_x1p0`列に固定した。
  exp429で判明した保存exp072 deltaからabsoluteへ戻す際の最大約`0.000352 ft`の
  表現差は診断として記録するが、PF parity gateには使わない。
- same bankを1 worker / 4 workersで再採点してprediction/evidence logical SHAを
  比較するpreflightを実装した。PF seedはwell IDとseed indexのstable SHAで固定する。
- suffix TVT、fold、hidden-like roleはcombined prediction/evidence freeze後だけ読む。
- deep/shallowは`md_since=1000 ft`、high missingnessはwell missing率`0.30`、
  roughnessはfrozen arithmetic meanのwell別二階差分RMS中央値splitに固定した。
- mergeは4 shard rootと4 summary SHA、full shardはpreflight summary SHAが
  未固定ならfail closedする。
- inference notebookはtrain gate PASSと別承認がない限り例外停止し、
  raw-test PF、submissionを生成しない。

## 実行量契約

- technical preflight variants / wells / PF well-runs: `1 / 4 / 4`
- preflight seed-well trajectories / particle starts: `512 / 256,000`
- full scientific variants / PF well-runs: `1 / 773`
- full seed-well trajectories / particle starts: `98,944 / 49,472,000`
- full shards / merge PF runs: `4 / 0`
- parent independent full rerun: `0`
- LightGBM config / trained fold / booster / model / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 再現性メモ

- seed:
  `stable_seed("likpf", "train", well_id) + seed_index_0_to_127`
- stochastic components:
  particle初期化、rate/position noise、systematic resampling、roughening
- parallel policy:
  well別local seed、deterministic LPT 4 shards、shard順非依存のseed label
- freeze:
  float64 trajectory bank → trajectory logical SHA → evidence/readout SHA →
  combined prediction SHA → truth/fold/hidden-like role
- gzipはdecompressed content SHAを主証拠とする。
- cross-rerun parity未確認のためdeterministic anchorとは呼ばない。

## notebook比較

- 親exp404 compact train: `2,174`行
- exp430 compact train: `3,015`行
- exp430は親のruntime/input/PF/freeze/late truth/metrics/生成物の章を保持し、
  trajectory memmap、evidence再採点、preflight、4 shard、mergeの章を追加した。
- 同一実験helper importなし。notebook-safeに`__file__`を使わない。
- train notebook: 25 cells
- inference notebook: 9 cells

## 検証ログ

```bash
.venv/bin/python -m py_compile \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_train.py \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_train.py \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_inference.py \
  experiments/exp430_huber_seed_evidence_reaggregation/tests/test_exp430_huber_seed_evidence_reaggregation.py \
  --select F821,F401,F841
.venv/bin/pytest -q experiments/exp430_huber_seed_evidence_reaggregation/tests/test_exp430_huber_seed_evidence_reaggregation.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp430_huber_seed_evidence_reaggregation/exp430_huber_seed_evidence_reaggregation_compact_selfcontained_inference.py
```

- syntax: PASS
- Ruff `F821,F401,F841`: PASS
- exp430専用contract tests: comparator regression追加後 `12 passed`
- Jupytext round-trip: PASS
- 親exp404 PF kernelとの小配列RNG / trajectory / diagnostic: bit-exact PASS
- `make validate-exp EXP=exp430_huber_seed_evidence_reaggregation`: strict PASS
- `make validate-template`: PASS
- `make update-summary`: 431 experimentsで更新完了
- Kaggle notebook実行: version 2 technical preflight PASS

## 2026-07-29 preflight package検証

- kernel id / title:
  `kentookumura/exp430-huber-seed-evidence-reaggregation-train` /
  `exp430 huber seed evidence reaggregation train`
- private / CPU / internet: `true / true / false`
- `run_on_push=true`、`execution.selected_stage=preflight`
- fixed well assetはbootstrapへ格納し、SHA
  `f06e14df380b3213be01cdc8a2613d0e929313858171529e8e397eaa82c5aea0`
  と一致した。
- config bootstrap SHA:
  `edcacd4a0cfb1b93ebae5fec214c9d0f98ae2b570ad7ca34349b07a1349cf31b`
- package作成前のcontract test: `11 passed`

## 2026-07-29 Kaggle preflight v1

- kernel version `1`をcanonical slugへpushした。
- Kaggle kernel id_no: `128974735`
- Kaggle側metadataでprivate、CPU、internet無効、4つのinput sourceを照合した。
- push直後の状態: `RUNNING`
- 実行時間`174.301117 s`でtechnical FAIL。fullは開始していない。
- 実行量、float64 bank、shared bank SHA、1/4 worker prediction/evidence
  logical SHA、truth-late、weight sumはPASSし、12 checks中10 checksがPASSした。
- FAILはparent marginalとarithmetic meanの2 parityのみ。両者とも最大差は
  `0.00048437499935971573 ft`だった。
- v1 summary raw SHA:
  `947b0f47f7e50cef34a58255cc9a5ab73568f1c4e20781d7d63dc28b8a3eb038`
- trajectory bank raw / logical SHA:
  `b095585a1f41e6cf201673df55f51c387414a71a65fec85216bcd6d8f5d83985` /
  `73829c9275de381bf63d3a6214199fb5fa7fd8778d9ffe34d57a3701750b94e2`
- prediction raw / logical SHA:
  `ab9f32f9d6fe71981956a95a2225d61176766a4429eb3b447adc598cc1e1cedb` /
  `4860ad5a2cfc3eb58738a396a7983c41d5a9f16cf19f023405185c1e4d58be39`
- evidence logical SHA:
  `ed50a81380ef73308f4fa435490120236028056eeb79703e3fd557cb183cef8e`

### v1 failure診断とv2修正

- exp404は4 prediction列を`float32`へcastしてからCSV保存する。
- v1比較器はexp430実行中のbinary float32値
  （例`11183.7646484375`）と、exp404 CSVからfloat64へ再読込した10進値
  （例`11183.765`）を直接比較していた。観測差はfloat32の半ULPと一致する。
- exp404 frozen datasetを取得して18,055行を実値照合したところ、
  exp430 v1 CSVとexp404 CSVはparent marginal、arithmetic meanとも
  全行で最大差`0.0 ft`、float32 array parityも完全一致した。
- v2では両辺を親の保存dtypeであるfloat32へ正規化してから比較する。
  tolerance `1e-5 ft`、PF、seed、trajectory、evidence、readout、実行量は変更しない。
- 修正後の専用contract test: `12 passed`。
- v2 package config / code / notebook SHA:
  `70af9e9b366106c6cb9575cd8edb51277293e96534929cb26506f5ea40675349` /
  `0dff10b802d091777a4590a59e450337e7b2542ca6472874e92e6077774cd42f` /
  `2c2294a7a9316b65fc1e6acf77f7e53651f2899b9157d7dcb2ec6f1aef5bf5df`
- v2 bootstrap内のcomparator修正とfixed-well asset SHAを再照合した。

## 2026-07-29 Kaggle preflight v2 PASS

- canonical kernel version `2`、id_no `128974735`、CPU / internet無効で完了した。
- runtime: `311.1591536998749 s`
- 12 / 12 technical checks: PASS
- parent marginal / arithmetic mean parity: `0.0 / 0.0 ft`
- weight sum最大絶対誤差: `3.3306690738754696e-16`
- 1 worker / 4 workersのprediction / evidence logical SHA: 一致
- truth / fold / hidden role / errorのfreeze前アクセス: すべて`0`
- preflight summary raw SHA:
  `3a34add3b77abc08add3fb8c37c4fadb8a4f249577c13ca0116552e417c6eddc`
- trajectory bank raw / logical SHA:
  `b095585a1f41e6cf201673df55f51c387414a71a65fec85216bcd6d8f5d83985` /
  `73829c9275de381bf63d3a6214199fb5fa7fd8778d9ffe34d57a3701750b94e2`
- prediction raw / logical SHA:
  `ab9f32f9d6fe71981956a95a2225d61176766a4429eb3b447adc598cc1e1cedb` /
  `4860ad5a2cfc3eb58738a396a7983c41d5a9f16cf19f023405185c1e4d58be39`
- evidence raw / logical SHA:
  `ac7655553adcb64b4dc8d6d9add1baf86c8b326d2ac35cb4be9e9f659c72734d` /
  `ed50a81380ef73308f4fa435490120236028056eeb79703e3fd557cb183cef8e`
- v1/v2でtrajectory、prediction、evidence、parent seed score、row identity、
  trajectory indexのraw SHAがすべて一致した。fixed4範囲をdeterministic anchorとする。
- preflightはpromotion evidenceではない。full shard、merge、inference、
  submissionは未承認で、`selected_stage=null`、`run_on_push=false`へ戻した。
- ローカルKaggle packageもpost-run configで再生成し、誤再実行防止を照合した。
- 最終検証: contract tests `12 passed`、Ruff PASS、Jupytext round-trip PASS、
  strict experiment validation PASS、template validation PASS。

## 次のアクション

1. exp430はterminal closeとし、inference / submissionへ進めない。
2. delta / temperature等のsame-OOF rescueを行わない。
3. 独立した原因説明が必要な場合だけ、保存済みweight ESS・best-seed disagreement・
   parent marginal weight差を使う0-PF attribution readoutを低優先で別承認する。
