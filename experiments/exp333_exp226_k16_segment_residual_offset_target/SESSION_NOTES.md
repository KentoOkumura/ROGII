# exp333 セッションノート

## 2026-07-23 downstream exp361

- exp361で保存済みStage 1 OOFをexp293 fixed12へadd-oneし、technical/novelty guardをPASSした。
- H512 / whole-well oracle改善は`+0.133103876 / +0.102132339 ft`、
  H512 strict unique-best`11.5064%`、5/5 folds改善。
- 元のdirect-promotion FAILは変更しないが、candidate pathとしてのcurrent-test inferenceは
  別承認で実装する価値が支持された。
- inferenceは同じexp333内で扱い、新規expは切らない。現時点では未承認・未実装。
- 単独採用、平均blend、selector変更、submissionは支持されていない。

## 目的

exp226 OOF residualをK16 segment単位の平均offset targetへ集約し、row-wise residual学習より安定して予測できるかを検証する設計を確定する。

## 現在の状態

- 2026-07-21: backlog、steering、experiment scaffoldを作成し、設計を固定。
- Route: `ensemble`。exp226 baseとML residual offsetの両方を使う。
- 状態: Stage 0 Kaggle CPU v1 PASS / Stage 1 preflight v1 PASS / full Stage 1 Kaggle CPU v1 COMPLETE・固定gate FAIL・branch close。
- CV / LB: `9.076676660936826 / -`。inference・submissionは未実施。
- Notebook: compact self-contained trainを正規trainへ採用済み。inferenceはfail-closed。

## 固定事項

- exp226数値互換K16、12,368 segments。
- targetは`mean(TVT-exp226_nested_pred)`、sample weightはrow数。
- offset-only constant broadcast。slope/clip/shrink/taper/interpolationなし。
- featureはprojection/U-disagreement/GRWRのsegment finite meanと固定7構造列だけ。
- exp145 learned-likelihood、selector score、truth/error/oracle、well IDを禁止。
- Stage 1はsaved exp226 outer fold × inner 4-foldのstrict nested base target。
- modelはexp228 `lgb1` 1 config、5 CPU boostersだけ。

## 実行量

- Stage 0: 1 headroom readout、model/config/fold train/booster `0/0/0/0`。
- Stage 1最大: 1 variant × 1 config × 5 outer folds = 5 CPU boosters。
- nested exp226: 25 donor-field/kappa fits、3,865 prediction well-runs。
- parent/control再学習: 0。GPU: 0。
- Stage 0 run、Stage 1実装、Stage 1 preflight、Stage 1 full runの承認は消費済み。固定gate FAILのためbranchを閉じ、inference、submissionは未承認・未実施。

## 2026-07-21 Stage 1実装

- ユーザー依頼「Stage1に進んでください」をStage 1実装承認として扱った。Kaggle package/push/run、推論、提出の承認には拡張していない。
- 実装量は1 variant × 1 LightGBM config × 5 outer folds = 5 CPU boosters。strict nested exp226はouter 5 + inner 20 = 25 donor-field/kappa fits、3,865 prediction well-runs。parent/control再学習0、GPU 0。
- saved exp226 outer foldを正とし、outer-train wellは`sha256("exp333|outer={f}|well={well_id}")`のsort + round-robinで4 inner foldsへ固定する。
- outer-validはfull outer-train、outer-train targetはinner-trainだけから生成する。outer-valid predictionとsaved exp226 OOFの最大差`1e-8 ft`をhard guardにした。
- exp072 cacheは`target`列をloadせず、prefix anchorもraw `MD/Z/TVT_input`だけで復元する。exp228 source dependencyから`projection_correction`、`u_disagreement`、`gr_wavelet_rotation_confidence`生成器だけを使い、learned-likelihood/selector/truth/error/oracle列をfeatureへ入れない。
- row featureはK16ごとにfinite float64 meanへ集約し、全non-finiteはNaNを保持する。追加構造列は固定7列だけ。targetはlate join後のmean residual、weightはsegment row数、broadcastはconstantで、clip/shrink/taper/interpolation/slopeなし。
- exp228 `lgb1`固定params、early stopping 250、CPU deterministic/force-col-wise/8 threadsを1 configだけ実装した。
- pooled/fold/near/1000+/hidden-like 2面/segment boundary ±8/by-well p95/worst-well/segment targetの全固定gateを実装し、同一OOF救済や追加configは持たない。
- input/fold/segment/feature schema/content/nested prediction/segment target/model/OOFのSHA manifestを保存する。inference/submission artifactは生成しない。
- 重いexp226 predictorとGRWR generatorはrepository方針で補助実装に残してよい範囲として、SHA対象のbootstrap dependency sourceへ固定した。Notebook側には入力、split、集約、学習、評価、保存先を展開した。
- 32-well preflightはfull-source 25 fitsを実測し、selected prediction時間を3,865 runsへ、target-free feature時間を3,783,989 rowsへ外挿し、5 boostersとartifact I/Oに固定1,800秒reserveを加えたprojected full runtimeを8.5時間gateへ掛ける。preflight自身は0 booster。
- Stage 1は別名`*_stage1_compact_selfcontained_train.py/.ipynb`へ作成し、既存Stage 0正規trainと同名compact `.ipynb`は上書きしていない。

### Stage 1実装検証

```text
.venv/bin/python -m py_compile <stage1_train.py> <inference.py> <dedicated_test.py>
.venv/bin/ruff check <stage1_train.py> <inference.py> <dedicated_test.py> --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp333_exp226_k16_segment_residual_offset_target/tests/test_exp333_exp226_k16_segment_residual_offset_target.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <stage1_train.py> --output <stage1_train.ipynb>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <stage1_train.py>
```

- py_compile / Ruff: PASS。
- 専用pytest: `14 passed`。
- Jupytext変換 / `--test`: PASS。
- hidden-like assignment SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`、config固定値と一致。
- strict experiment / project validation: PASS。bootstrap support collectionでexp226 source/config、exp228 source、hidden-like assignmentの4依存を解決できることを確認。
- repository regressionは既知のexp296 testを除外して`494 passed, 2 skipped`。exp296の既知2失敗はStage 0記録のとおりで、exp333変更範囲外のため変更していない。
- 実装完了時点は`selected_stage=stage_1_implementation_complete`、`kaggle_push_approved=false`、`stage_1_preflight_approved=false`、`stage_1_run_approved=false`。この時点ではStage 1実測、booster生成、Kaggle runは0だった。

## 2026-07-21 Stage 0実装

- ユーザー依頼「exp333を実装してください」を、凍結設計に記載された最初の承認境界であるStage 0実装の承認として扱った。
- 保存済みexp226 OOFのphysical/decompressed SHAを検証し、truth/error列を読まずに`well_id,row_idx,suffix_offset,tvt_pred,fold`だけをロードする。
- row identity、well単位fold、suffix offset、exp226互換K16 assignment、fold mapを検証・SHA化してtarget-free contractをfreezeする。
- freeze後だけ`tvt_true`をrow keyでlate joinし、segmentごとのfloat64 mean residualをconstant broadcastする0-model oracle readoutを実装した。
- oracle offset、segment target、oracle predictionはdeployable生成物へ保存しない。truth/segment target/oracle readoutはcontent SHAだけをsummaryに残す。
- pooled gain`>=1.00 ft`かつfold gain`>=0.50 ft`を5/5要求する固定gateを実装し、FAIL時は`FAIL_CLOSE_BRANCH`を返す。
- Stage 1 / LightGBM / nested exp226コードは追加していない。inferenceは明示的なfail-closed stubにした。
- 既存の正規Notebook scaffoldは上書きせず、`*_compact_selfcontained_train.py/.ipynb`と`*_compact_selfcontained_inference.py/.ipynb`を別名で作成した。

### 実装検証

```text
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py> <dedicated_test.py>
.venv/bin/ruff check <compact_train.py> <compact_inference.py> <dedicated_test.py> --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp333_exp226_k16_segment_residual_offset_target/tests/test_exp333_exp226_k16_segment_residual_offset_target.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
```

- py_compile: PASS。
- Ruff: PASS。
- 専用pytest: `10 passed`。
- Jupytext変換: train / inferenceともPASS。
- Jupytext `--to ipynb --test`: train / inferenceともPASS。
- strict experiment validation / template validation: PASS。
- saved exp226 OOF decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`、config固定値と一致。
- parent exp226にcompact self-contained trainはない。通常train sourceは111行/6章、exp333 compact trainは940行/9章で、入力freeze、K16 assignment、late truth、oracle metric、SHA、orchestrationをNotebook上に展開した。
- `__file__`参照: 0。Kaggle package/push/run: 0。Stage 0実測: 0。
- repository testは既知のexp296を除外して`480 passed, 2 skipped`。全体では未変更のexp296に`2 failed`が残り、statusが旧test期待`kaggle_cpu_*`と不一致、完了後`run_variant=false`と旧approval test期待が不一致の既存状態である。exp333専用testとは独立のためexp296は変更していない。

## 再現性

- outer foldはsaved exp226 identity、inner foldはstable SHA256、LightGBM random state 0。
- CPU deterministic/force-col-wise/8 threadsに固定し、global RNGを使わない。
- raw/input、saved OOF、fold map、K16 assignment、row/segment feature schema/content、nested base、segment target、model manifest、OOF row predictionのSHAを保存する。
- gzipはdecompressed content SHAを主証拠とする。
- outer-valid nested exp226とsaved exp226 OOFの最大差`1e-8 ft`をhard guardにする。
- deterministic anchorは、実装・run・rerunをしていないため主張しない。

## コマンドログ

### 2026-07-21 Stage 1 full Kaggle CPU train実行承認

- ユーザー依頼「次にすすんでください」を、preflight PASS後に記録済みの次段階であるfull Stage 1 Kaggle CPU trainの承認として扱う。
- 実行量: 1 active variant × 1 LightGBM config × 5 outer folds = 5 CPU boosters。
- strict nested exp226: outer 5 + inner 20 = 25 donor-field/kappa fits、3,865 prediction well-runs。
- parent/control再学習0。比較には保存済みexp226/exp228/exp263を使う。GPU 0、internet off。
- preflight専用kernelとモデル生成物を分離するため、canonical full-train kernelを`kentookumura/exp333-k16-segment-residual-stage1-train`、titleを`exp333 k16 segment residual stage1 train`とする。
- full Stage 1の全固定gateを評価する。自動救済、追加config/seed、inference、submissionへは進まない。
- pre-push専用testで、fail-closed inference stubが`stage_1_run_approved=false`まで要求し、train承認状態のimportを拒否する契約不整合を検出した。inference/submissionの無効・未承認guardは維持し、train承認とは独立させる最小修正をcompact inference source/Notebookへ反映した。

### 2026-07-21 full Stage 1 Kaggle CPU train v1

```bash
make prepare-kaggle-notebooks \
  EXP=exp333_exp226_k16_segment_residual_offset_target \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp333-k16-segment-residual-stage1-train --title 'exp333 k16 segment residual stage1 train' --run-on-push --strict"
kaggle kernels push \
  -p experiments/exp333_exp226_k16_segment_residual_offset_target/kaggle/train
kaggle kernels pull kentookumura/exp333-k16-segment-residual-stage1-train \
  -p /tmp/kaggle-pull/exp333-k16-segment-residual-stage1-train -m
kaggle kernels logs kentookumura/exp333-k16-segment-residual-stage1-train
kaggle kernels output kentookumura/exp333-k16-segment-residual-stage1-train \
  -p /tmp/kaggle-output/exp333-stage1-train-v1 \
  --file-pattern "(^metrics\\.json$|stage1_(contract\\.json|summary\\.json|fold_metrics\\.csv|bucket_metrics\\.csv|hidden_like_metrics\\.csv|boundary_metrics\\.csv|by_well_metrics\\.csv|feature_importance\\.csv|model_manifest\\.json|sha_manifest\\.csv)$)"
```

- Kernel: `kentookumura/exp333-k16-segment-residual-stage1-train`
- version / id_no / status: `1 / 128116592 / KernelWorkerStatus.COMPLETE`。
- runtime: CPU、GPU false、internet false。summary完了`1,781.997 sec`、final log`1,790.189 sec`（約29.84分）。preflight外挿`6,434.437 sec`より短く完走。
- 実行量: 1 variant / 1 config / 5 outer folds / 5 LightGBM boosters、strict nested exp226 25 fits / 3,865 prediction well-runs、parent/control再学習0。
- coverage: `3,783,989 rows / 773 wells / 12,368 segments`。model featureは136列。
- outer-valid exp226 parity最大差`1.8189894035458565e-12 ft`、target/error pre-freeze load 0、feature/input contractはPASS。
- pooled: exp226 `9.427109597` → Stage 1 `9.076676661`、改善`0.350432936 ft`。ただし固定pooled上限`8.894085501`を`0.182591160 ft`超過しFAIL。exp228 `8.944085501`より`0.132591160 ft`悪く、exp263 `8.238331715`より`0.838344946 ft`悪い。
- fold Stage 1 RMSE: `8.901897 / 8.514004 / 9.946437 / 9.100572 / 8.883281`。exp226比改善は`0.554733 / 0.617955 / 0.338390 / 0.086735 / 0.163200 ft`で5/5 folds PASS。
- segment-target weighted RMSEも5/5 foldsでzero-offset priorを改善。
- scope delta vs exp226: near 0--250 `+0.057439 ft` FAIL、250--1000 `-0.238416 ft`、1000+ `-0.380695 ft` PASS、hidden spatial `-0.367928 ft` PASS、hidden typewell-purged `-0.355678 ft` PASS、boundary ±8 `-0.329524 ft` PASS。
- by-well p95 delta `-0.352018 ft`はPASS。一方worst well `7987f2f2`は`+8.099023 ft`で固定上限`+0.25 ft`をFAIL。
- gateはpooled、near 0--250、worst-wellの3件がFAIL。decision=`FAIL_CLOSE_BRANCH`、scientific PASS=false、inference candidate threshold=false。
- feature freeze SHA: `b2c7bff40f9fc994bd60471c03d9085ba48137c30b358402bfbb1cadecc4a078`。
- model manifest SHA: `3e4c99b2451c7731331bca7fabba44353ce7ac7e80e7c821f4799983d8f297a9`。5 model SHAはmanifestとSHA manifestで一致。
- OOF prediction SHA: `dbb3f41642a2d6a9da704d276ed6398b706059078bcfcaca95e17e5c7af00784`。segment target SHA: `65a73d741778509edc732949b4a1ba2b94152796a111b836a6af3898c6b00027`。
- CV/fold/scope/by-well/model/SHA確認に必要な小容量ファイルだけを`--file-pattern`で取得した。247 MB nested prediction、98 MB OOFなどの大容量archiveは取得していない。
- pandasのDataFrame fragmentation `PerformanceWarning` 844件とnbconvert/mistune `SyntaxWarning` 2件があったが、Traceback/MemoryError/RuntimeWarningは0で、全生成物保存後にCOMPLETEした。branch closedのためperformanceだけの再pushはしない。
- full-run承認は消費済みとして`selected_stage=stage_1_train_completed_fail_closed`、push/run承認falseへ戻した。追加config、same-OOF救済、inference、submissionは実行しない。

### 2026-07-21 Stage 1 parity/runtime preflight実行承認

- ユーザー依頼「実行してください」を、記録済みの次段階である32-well Kaggle CPU parity/runtime preflightの承認として扱う。
- 実行対象: 1 preflight audit、LightGBM active variant 0、model config 0、trained fold 0、booster 0、GPU 0。
- strict nested exp226は全773 wellsをsource候補としてouter 5 × inner 4 + outer 5 = 25 donor-field/kappa fitsを実測する。prediction targetはsaved outer foldで均衡化した32 wells、最大160 well-runs。
- target-free feature generatorも同じ32 wellsで実測し、full row数へ外挿する。projected full runtimeはnested外挿 + feature row-scale外挿 + 固定1,800秒の5-booster/artifact I/O reserveで評価する。
- parent/controlの再学習は0。nested fitはStage 1 base生成の実測であり、保存済みexp226 controlの再生成や比較用booster学習ではない。
- Stage 1別名Notebook候補を正規train Notebookへ採用してpreflightだけ実行する。full Stage 1 train、inference、submissionは未承認で、自動移行しない。
- planned kernel: `kentookumura/exp333-k16-segment-residual-stage1-preflight`。
- title: `exp333 k16 segment residual stage1 preflight`。id末尾とtitle由来slugを一致させる。

### 2026-07-21 Stage 1 parity/runtime preflight v1

```bash
make prepare-kaggle-notebooks \
  EXP=exp333_exp226_k16_segment_residual_offset_target \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp333-k16-segment-residual-stage1-preflight --title 'exp333 k16 segment residual stage1 preflight' --run-on-push --strict"
kaggle kernels push \
  -p experiments/exp333_exp226_k16_segment_residual_offset_target/kaggle/train
kaggle kernels pull kentookumura/exp333-k16-segment-residual-stage1-preflight \
  -p /tmp/kaggle-pull/exp333-k16-segment-residual-stage1-preflight -m
kaggle kernels logs kentookumura/exp333-k16-segment-residual-stage1-preflight
kaggle kernels status kentookumura/exp333-k16-segment-residual-stage1-preflight
```

- Kernel: `kentookumura/exp333-k16-segment-residual-stage1-preflight`
- version / id_no / status: `1 / 128114252 / KernelWorkerStatus.COMPLETE`。
- runtime: CPU、GPU false、internet false。kernel sourceはexp072 feature cacheとexp226 train output。
- bootstrapは22 support filesを格納し、configはlocalとbyte-identical。exp226 source/config、exp228 source、hidden-like assignmentを含むことをpush前に確認。
- 実行量: `1 preflight / 0 LightGBM variant / 0 config / 0 trained fold / 0 model / 0 booster`。parent/control再学習0。
- selected: `32 wells / 166,533 target-free feature rows`。strict nested exp226はfull 773-well source poolで`25 fits / 160 prediction well-runs`。
- elapsed: 全体`491.884531 sec`、nested`338.206046 sec`、feature`153.678483 sec`。nested内訳はfit`300.803808 sec`、prediction`34.845080 sec`。
- outer-valid parent parity最大差: `1.8189894035458565e-12 ft`。固定上限`1e-8 ft`をPASS。
- full Stage 1外挿: nested`1,142.530269 sec` + feature`3,491.906651 sec` + 5-booster/artifact reserve`1,800 sec` = `6,434.436920 sec = 1.787344 h`。
- runtime gate: `6,434.437 <= 30,600 sec`でPASS。marginは`24,165.563 sec = 6.713 h`。
- feature列の逐次挿入に対するpandas `PerformanceWarning`とnbconvert/mistuneの`SyntaxWarning`が出たが、parity、coverage、runtime summary、kernel completionには影響しなかった。
- logsだけでparity/runtime判定と実行量確認が完結し、Stage 1 artifactやCV実ファイルは生成していないためKaggle output archiveは取得していない。
- preflight完了時点では承認を消費済みとして`selected_stage=stage_1_preflight_completed`、`kaggle_push_approved=false`、`stage_1_preflight_approved=false`へ戻した。この時点ではfull Stage 1 run、inference、submissionは未実行・未承認だった。

### 2026-07-21 Stage 0 Kaggle CPU実行承認

- ユーザー依頼「実行してください」を、Stage 0のKaggle CPU package/push/run承認として扱った。Stage 1、inference、submissionの承認には拡張しない。
- 実行対象: 1 headroom readout、active variant 0、model/config 0、trained fold 0、booster 0、GPU 0。
- 保存済みexp226 OOFをkernel sourceとして読むだけで、parent/control再学習は0。
- canonical train notebookへcompact self-contained Stage 0を採用する。canonical inferenceは未採用・未実行。
- planned kernel: `kentookumura/exp333-k16-segment-residual-stage0-train`。
- title: `exp333 k16 segment residual stage0 train`。id末尾とtitle由来slugを一致させる。
- credential check: Kaggle CLI OAuth / legacy credential利用可能。API tokenは未設定だがCLI 2.2.3のOAuth認証を使用する。

### 2026-07-21 Stage 0 Kaggle CPU v1

```bash
make prepare-kaggle-notebooks \
  EXP=exp333_exp226_k16_segment_residual_offset_target \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp333-k16-segment-residual-stage0-train --title 'exp333 k16 segment residual stage0 train' --run-on-push --strict"
kaggle kernels push \
  -p experiments/exp333_exp226_k16_segment_residual_offset_target/kaggle/train
kaggle kernels pull kentookumura/exp333-k16-segment-residual-stage0-train \
  -p /tmp/kaggle-pull/exp333-k16-segment-residual-stage0-train -m
kaggle kernels logs kentookumura/exp333-k16-segment-residual-stage0-train
kaggle kernels status kentookumura/exp333-k16-segment-residual-stage0-train
```

- Kernel: `kentookumura/exp333-k16-segment-residual-stage0-train`
- version / id_no / status: `1 / 128109500 / KernelWorkerStatus.COMPLETE`
- runtime: CPU、GPU false、internet false、kernel sourceは`kentookumura/exp226-k16-kappa-repro-train`だけ。
- bootstrap configはlocal configとbyte-identical。Stage 0 `1 readout / 0 variant / 0 config / 0 trained fold / 0 booster`、Stage 1 disabledをpush前に確認。
- readout完了`104.648883 sec`、final log`114.706912 sec`。
- coverage: `3,783,989 rows / 773 wells / 12,368 segments / 5 folds`。
- exp226 RMSE: `9.427109596582222`。expectedとの差`8.88e-15 ft`。
- K16 oracle mean-offset RMSE: `1.1306025263265356`。
- pooled gain: `8.296507070255686 ft`、固定閾値`1.00 ft`をPASS。
- fold gain: `8.359820609 / 8.002651449 / 9.211416557 / 8.005800122 / 7.879884974 ft`。全foldで固定閾値`0.50 ft`をPASS。
- decision: `PASS_STAGE0`。technical/scientificともPASS、Stage 1は別承認後に実装可能。
- input decompressed SHA: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- truth SHA: `3b50d64f7d6bb4cee0cea3dc637494c41605d6e26a4bb9172fecbfde36e8c82e`。
- segment assignment SHA: `6b833c0bcdbe2b82b2e16df23f5fd0dae1412a890a90cf53743610fbb01e07e3`。
- segment target SHA: `15f47cbb46b655c50b57718d6549d79d5e15101aa7ffde74022c069f1c1f6dfb`。
- oracle readout SHA: `616157b44fc769b111071771f959b37e3ab3f958b7657ee25b20252cf4753d38`。
- oracle offset、segment target、oracle predictionはdeployable生成物として保存していない。CV確認はlogsで完結したためKaggle output archiveは取得していない。
- Stage 0 run承認は消費済みとしてconfigをfail-closedへ戻した。再push、Stage 1実装、inference、submissionは行っていない。

```bash
task new-steering EXP=exp333_exp226_k16_segment_residual_offset_target
```

- `task`が環境に存在せず未実行。

```bash
make new-steering EXP=exp333_exp226_k16_segment_residual_offset_target
make new-exp EXP=exp333_exp226_k16_segment_residual_offset_target
```

- steering/scaffold作成済み。
- scaffold作成、Stage 0実装・validation、Kaggle CPU v1完了。

## 次

exp333 branchは閉じる。低優先の次候補として、保存済みexp333 OOFを予測変更に使わず、near/worst悪化がsegment offsetの大きさ・符号・well寄与の偏りに集中するかだけを0-boosterで監査する。根拠がなければfamilyを終了し、同じOOFでclip/shrink/gateを探索しない。

## 2026-07-23 current-test candidate inference

### 承認と範囲

- exp361 v2がexp333をfixed exp293 bankへのadd-one候補として支持した後、ユーザーの
  「次に進んでください」をcurrent-test候補artifact生成の別承認として扱う。
- 同じ`exp333_exp226_k16_segment_residual_offset_target`内で実装し、新しいexpは作らない。
- 元のdirect-promotion `FAIL_CLOSE_BRANCH`、near 0--250 / worst-well failure、
  final inference禁止、submission禁止は履歴として維持する。今回のoverrideは
  candidate artifact生成だけである。

### Kaggle push前の固定実行量

- current-test candidate variant: 1
- 保存済みLightGBM model inference: outer fold 0..4の5 pass
- 学習するLightGBM config: 0
- 学習するfold: 0
- 新規booster: 0
- parent/control再学習: 0
- GPU: 0、CPU 8 threads、internet off
- raw-test replay: exp072 v2 stable SHA256 per-well seed、PF seeds 128、
  particles 500
- selector / blend / fixed12 average / clip / shrink / taper / slope: 0
- `submission.csv`生成: 0、competition submit: 0

### 実装

- steering:
  `.steering/20260723-exp333-exp226-k16-segment-residual-offset-target-current-test-candidate-inference/`
- Jupytext source:
  `exp333_exp226_k16_segment_residual_offset_target_current_test_compact_selfcontained_inference.py`
- raw current testからexp072と同じ196列を再生成し、exp228のtarget-free
  U projection / disagreement / GRWR生成器で129 row featureを構築する。
- exp226 inference v1（14,151行、SHA
  `b71e15f7dc7e66f7be70db4a81d9ec72e1001ff2ba13907c3aba24938e906047`）
  をbaseにする。
- trainと同じK16、finite float64 segment mean、7 structural列を用いて
  3 wells × 16 = 48 segment、136 model featureを作る。
- Stage 1 train v1のmodel manifest SHA
  `3e4c99b2451c7731331bca7fabba44353ce7ac7e80e7c821f4799983d8f297a9`
  と5 model SHA、feature順、saved OOF SHA manifest、saved train summary値を
  同じNotebookでfail-closed照合する。
- 5 modelのsegment offsetをfloat64等重み平均し、exp226へ無加工で加える。
- artifactはcandidate row、segment prediction、feature schema、projection/GRWR
  summary、model/boundary/input/SHA audit、summaryのみとし、submissionは作らない。

### ローカル検証

- 専用pytest: `16 passed`。
- Python AST/import: PASS。
- Ruff F821: PASS。
- Jupytext `--to ipynb` / `--to ipynb --test`: PASS。
- 別名candidate Notebookを正規
  `exp333_exp226_k16_segment_residual_offset_target_inference.ipynb`へ採用した。
- Kaggle Stage 1 train出力から5 model、manifest、feature schema、summary、
  SHA manifestだけを取得し、固定SHAをconfigへ記録した。学習・再実行は0。

### Kaggle CPU実行

```bash
make prepare-kaggle-notebooks \
  EXP=exp333_exp226_k16_segment_residual_offset_target \
  EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp333-k16-segment-residual-candidate-inference --title 'exp333 k16 segment residual candidate inference' --run-on-push --strict"
make push-kaggle-infer EXP=exp333_exp226_k16_segment_residual_offset_target
kaggle kernels status kentookumura/exp333-k16-segment-residual-candidate-inference
kaggle kernels logs kentookumura/exp333-k16-segment-residual-candidate-inference
kaggle kernels output kentookumura/exp333-k16-segment-residual-candidate-inference \
  -p /tmp/kaggle-output/exp333-candidate-inference-v2 \
  --file-pattern '(^metrics\.json$|.*current_test_(candidate\.csv\.gz|segment_predictions\.csv|feature_schema\.csv|model_audit\.csv|boundary_audit\.csv|input_manifest\.csv|summary\.json|sha_manifest\.csv)$)'
```

- Kernel: `kentookumura/exp333-k16-segment-residual-candidate-inference`
- canonical version / id_no / status: `2 / 128368525 / COMPLETE`
- version 1はsaved train/model/exp226 SHA parityをPASS後、raw replayが205 featureを
  返す一方でexp072 cache schemaが196列であることを検知し、予測前にERROR停止した。
  追加9列は`likpf_scale_{3,5,8,12}`、`likpf_mean`と各deltaで、exp072が正規に
  除外するdiagnosticだった。
- version 2はexp072の`feature_columns_for_variant(...,
  "pixiux_likpf_public_replay")`で同じ196列へsubsetする最小修正のみ。
  feature/target/model/gateは変更していない。
- v2 runtime: feature generation`51.446 sec`、5 saved-model inference`3.032 sec`、
  summary total`65.258 sec`。
- coverage: `14,151 rows / 3 wells / 48 K16 segments`、129 row features /
  136 model features / 5 saved models。
- saved train decision=`FAIL_CLOSE_BRANCH`、model manifest SHA、5 model SHA、
  feature schema SHA、saved OOF file/decompressed SHAはすべてparity PASS。
- current row feature / segment feature / segment prediction / candidate prediction
  content SHA:
  `9475721131bfd93a036d0a636d473a8cf6cc8d7d46eaf203b3879ccba6272a79` /
  `c6cbe6deedc58dda2b7df4ec072489f881b5893263c02983ad2be2da54ff786c` /
  `fc54e72e1865b6669d5a1af2998af26abc0c97473d284ec7ea86c286c8213923` /
  `316e3b770925cc3e0013bfbc46a86629619b8f214a5913a482c83607c3865d02`。
- candidate file / decompressed SHA:
  `d7a6bae97b7ea81aaa41f7b7850a1d56286ccb57db354ed62893c28d74ef49c1` /
  `7571c6281bd2ab484e7bf536a876b8072407b272a0ef0ec5112ca06897a717cd`。
- offset min/max/mean/std:
  `-4.249479 / +2.592369 / +0.289689 / 1.717736 ft`。
- downloaded artifactをsample/exp226 v1と独立照合し、ID順、一意性、finite、
  base+offset、5-fold mean、suffix/K16境界、file/decompressed SHAをPASSした。
- Kaggle全output一覧にも`submission.csv`は存在しない。model/booster学習、
  parent/control再学習、selector/blend/fixed12 average、competition submitは0。
- 出力は
  `kaggle/output/inference_v2/artifacts/exp333_exp226_k16_segment_residual_offset_target_current_test_candidate.csv.gz`
  へ保存した（`experiments/*/kaggle/`はgitignore対象）。
- one-run authorizationは消費済みとして
  `candidate_inference_approved=false` /
  `candidate_inference_authorization_consumed=true`へ戻した。

### 次

candidate artifact生成は完了。fixed bankへの組み込みは、13候補selector再学習と
target-free safety gateのどちらを単一変更として設計するかをユーザー判断に委ねる。
単独採用、平均blend、weight探索、submissionへは自動移行しない。
