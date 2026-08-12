# exp494 セッションノート

## 目的

exp413の保存済みLightGBMを固定アンカーに、同じStage D final370面の
CatBoost / XGBoostと固定物理候補をbounded stackingし、
最終提出候補としての平均性能、scope、well-tail、hidden runtimeを監査する。

## 現在の状態

- Route: `ensemble`
- 状態: reference submission COMPLETE / Public LB 7.228、不採用
- 親: `exp413_scale5_likpf_full_replacement_on_exp335`
- 親CV / Public LB: `7.884802794404715` / `7.201`
- 物理候補: exp413 scale5-overlay版`exp226_w500_50_50`
- 物理候補OOF: `8.070218793924594`、対応Public LBなし
- exp263同名候補`8.238331` / Public LB `7.800`は履歴contextのみ
- CV / Public LB: `7.827450885176479 / 7.228`
- 実装: Stage 0--5の別名Jupytext train候補あり
- 正規train Notebook: 採用承認済み
- Kaggle package / train run: version 1 Stage 0 fail-closed、version 2 kernel death
- inference / submission: COMPLETE / ref `55134873`

## 2026-07-30 設計

- ユーザーが7段階の最終アンサンブルパイプラインについて、
  backlog、実験ディレクトリ、steeringの作成と設計確定を承認した。
- `kaggle-review-exp`に従い、steeringを先に作ってからexp494 scaffoldを作成した。
- `kaggle-strategy`に従い、Late phaseの最終提出候補として既存候補との
  優先順位とnegative evidenceを確認した。
- exp413 Stage DにはOOF / modelsはあるがfinal370 matrix自体は常設されていない。
  そのためStage 0で同じ固定入力からfold別float32 matrixをmaterializeし、
  schema / content SHAを凍結してからStage 1へ進む設計にした。
- 物理候補はOOF `8.238331`、5/5 folds改善の
  `exp226_w500_50_50`だけに固定した。Public LB 7.678の候補を含む
  same-OOF / Public-LB candidate選択は行わない。
- stackingはLGB >= 0.60、Cat <= 0.25、XGB <= 0.20、
  Physics <= 0.20の非負・和1・interceptなしに固定した。
- 10-model制約を維持するためstrict nested base refitは行わない。
  readoutはOOF-level cross-fitとして記録し、strict nestedとは呼ばない。

## GPUコストガード

- active variants: 2 (`catboost_pixiux_cb0`, `xgboost_cdeotte_v3`)
- model/config数: CatBoost 1 + XGBoost 1 = 2
- outer folds: 5
- 合計新規GPU models: 10
- CatBoost最大iterations: 5 x 8,000
- XGBoost最大trees: 5 x 450 = 2,250
- exp413 LightGBM control再学習: 0
- 40 selector / 20 signed selector再学習: 0
- 新規PF/HMM/Beam: 0
- stacking / confidence gate学習booster: 0

2026-07-31のpush前再確認でもこの実行量に変更はなく、
ユーザーの`実行してください`をKaggle train実行の明示承認として記録した。

## 再現性メモ

- seed: CatBoost 7、XGBoost 42、foldはexp413 manifest固定
- stochastic components: CatBoost GPU、XGBoost GPU、
  hidden inferenceで継承するexp413 per-well PF replay
- deterministic anchor: false。独立rerun一致前は指定しない
- feature: final370 logical schema SHAとfold別float32 matrix content SHAを保存予定
- model: 5 Cat / 5 XGBの個別SHAとmanifestを保存予定
- prediction: family OOF / constant stack / gate / hidden prediction SHAを保存予定
- submission: submit-check後のroot `submission.csv` SHAを保存予定
- gzip: decompressed content SHAを主証拠とする
- bootstrap: embedded / loose config byte parityとcontract SHAをpush前に確認予定

## 2026-07-30 train-side実装

- ユーザーの`exp494を実装してください`を、凍結済み設計に対する
  train-side Stage 0--5実装承認として記録した。
- 既存の正規Notebookは明示承認なしに上書きせず、次を作成した。
  - `exp494_exp413_cat_xgb_physics_bounded_stack_compact_selfcontained_train.py`
  - `exp494_exp413_cat_xgb_physics_bounded_stack_compact_selfcontained_train.ipynb`
- Stage 0はexp413 Stage 0/C/S/Dの保存SHA、3,783,989 rows / 773 wells /
  outer 5 folds、clean273+nested74+signed23=final370、row key、
  fold別float32 matrix content SHA、物理候補formula/coverageを、
  family model importと学習より前に検証する。
- full matrixをStage 0で5 fold分常設せず、row chunkでcontent SHAを先に計算し、
  Stage 1でfold単位に一度だけfull materializeして同じSHAを再検算する。
- Stage 1はCatBoost `cb0` 1 config x 5とXGBoost Cdeotte v3
  1 config x 5だけを学習する。exp413 15 LGB、40 selector、20 signed selector、
  PF/HMM/Beamはload-onlyで再学習0。
- Stage 2--3はLGB/Cat/XGB/Physicsのpooled/fold/scope/hidden-like/by-well、
  prediction correlation、residual correlation、error covariance、
  disagreement quantilesを保存する。
- Stage 4はSLSQPの非負・和1・interceptなし固定boundを5-fold
  OOF-level cross-fitし、fold weight中央値を同じbounded simplexへ射影する。
- Stage 5はconstant stackの全AND gate PASS時だけ、target-free disagreement
  q50/q90をmeta-train 4 foldsで求め、constantから最大0.25 ftの補正を評価する。
- constant stack FAIL時はexp413 LGBを維持し、gate、candidate、parameter、
  bound、weight、threshold rescueを実行しない。
- train runは`authorization.kaggle_train_run_approved`とStage 0--5の全run flagが
  trueになるまで、大きな入力を読む前に停止する。
- hidden inferenceはtrain gate PASSと別承認が必要なため未実装。

### Notebook構成比較

- 親exp413 compact self-contained train: 9章 / 766行
- exp494 compact self-contained train候補: 9章 / 2,258行
- exp494は親の入力・lineage確認に加え、Stage 0 matrix preflight、
  Cat/XGB学習、4-family監査、bounded stack、conditional gate、
  再現性出力をNotebook上で追える構成にした。

### Static validation

- `task validate-exp EXP=exp494_exp413_cat_xgb_physics_bounded_stack`
  - `task`未導入のため実行不可（`task: command not found`）。
- `make validate-exp EXP=exp494_exp413_cat_xgb_physics_bounded_stack`
  - passed
- Jupytext `--to ipynb` / `--test`
  - passed
- `py_compile`
  - passed
- `ruff --select F821`
  - passed
- `pytest -q experiments/exp494_exp413_cat_xgb_physics_bounded_stack/tests/test_exp494_exp413_cat_xgb_physics_bounded_stack.py`
  - 10 passed
- `make test`
  - exp494のtest実行前のcollectionで、既存のexp297 / exp301 / exp333 /
    exp336 / exp349の設定契約エラー5件により停止した。
  - exp494専用test 10件は上記のとおり独立してPASSしており、
    この全体suite停止のために既存実験は変更していない。
- ローカルnotebook実行、Kaggle package、push、train runは未実施。

## 2026-07-31 train実行承認

- ユーザーの`実行してください`を、正規train Notebook採用、Kaggle package /
  push / Stage 0--5 train runの承認として記録した。
- 実行対象を再確認した。
  - active variants: 2
  - model configs: CatBoost 1 + XGBoost 1 = 2
  - outer folds: 5
  - 合計新規GPU models: 10
  - exp413 LightGBM control再学習: 0
  - selector / signed selector再学習: 0
  - 新規PF/HMM/Beam: 0
- Stage 6 hidden inference、inference run、submissionは今回の承認外として
  falseのまま維持する。
- Kaggle credential checkerはOAuth credentialとlegacy credentialを確認した。
  API tokenは未設定だが、Kaggle CLIのOAuth認証経路を使用する。
- compact self-contained候補を正規train Notebookへ採用した。
- canonical kernelを次へ固定してprivate T4 packageを生成した。
  - kernel: `kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-train`
  - title: `exp494 exp413 cat xgb physics bounded stack train`
  - GPU: `NvidiaTeslaT4`
  - internet: false
  - run on push: true
- competition 1件、dataset source 1件、kernel source 8件、
  bootstrap dependency 21件をmetadata / embedded bundleへ含めた。
- push直前のGPU quotaは`16.82 h / 45.00 h`残存だった。
- package検証値:
  - notebook bytes: 824,673
  - notebook SHA256:
    `ee852400c76b4ba3a0696588d06976aef13f10a45b91ab5d1ad3e16a004c5d51`
  - embedded / loose config SHA256:
    `8f7dd902dccfb07afc90c9f303fd5853459f8e2e26a089ed54cd5c5841f5259a`
  - config byte parity: PASS
- Kaggle private T4 version 1をpushし、run-on-pushで実行開始した。
  - kernel:
    `kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-train`
  - id_no: `129213293`
  - started: `2026-07-30 22:14:29 UTC`（2026-07-31 07:14:29 JST）
- push後のmetadata pullでprivate / GPU / internet off /
  `machine_shape=NvidiaTeslaT4` / input source 9件を確認した。

## 2026-07-31 train version 1 ERROR

- status: `KernelWorkerStatus.ERROR`
- runtime: `873.980775812 sec`
- failure stage: Stage 0 physical candidate contract
- traceback:
  `ValueError: physical OOF RMSE mismatch: 8.070218793924594`
- frozen expected RMSE: `8.238331`
- CatBoost / XGBoost trained models: `0 / 0`
- CV、family audit、stack、confidence gate: 未到達
- 分類: environment / path / GPU障害ではなくcandidate semantic source mismatch。
- 原因:
  - v1は`ReplacementCandidateCache`を使用した。
  - 同cacheはexp413のscale5 `likpf_mean` overlayを適用し、
    `exp226_w500_50_50`を同じIDのまま再構成する。
  - 凍結`8.238331`はoverlay前のexp263 original candidateを指す。
  - したがって同名IDでも予測内容が異なり、fail-closed guardが正しく停止した。
- 科学結果は0であり、10-model GPU学習コストは消費していない。

## 2026-07-31 exp413 scale5-overlay契約確定

- ユーザー確認により、親がexp413でありhidden inferenceも同じexp413 replayを
  使うため、物理familyはscale5-overlay版の一択と確定した。
- candidate IDは`exp226_w500_50_50`のまま、semanticsを次へ一意化した。
  - source: exp413 scale5 replacement candidate bank
  - semantic slot: `likpf_mean`
  - value source: `likpf_scale_5_x1p0`
  - formula:
    `0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`
  - expected OOF RMSE: `8.070218793924594`
- overlay前のexp263同名候補OOF `8.238331` / Public LB `7.800` /
  5/5 fold改善は、scale5版へ転用しない履歴contextとして分離した。
- version 2でも実行量は2 variants / 2 configs / 5 folds /
  10 GPU models、control・selector・PF/HMM/Beam再学習0のまま。
- 専用test 10件、Jupytext round-trip、`py_compile`、F821、
  strict experiment / template validationを再通過した。
- version 2 package:
  - notebook bytes: 842,314
  - notebook SHA256:
    `157d39ae6b1c75ca186efd0e7f9c4a17b991d5432bbe97644cddf82b2519aa1f`
  - embedded / loose config SHA256:
    `9f05425e0c6c2e24b87d5dabab703e8bb1fc28dda92ab394b9dfc0a3568302d3`
  - config byte parity: PASS
  - private / T4 / internet off / run-on-push: PASS
- push前GPU quotaは`16.58 h / 45.00 h`残存だった。
- 同じcanonical kernelへversion 2をpushして実行開始した。
  - id_no: `129213293`
  - started: `2026-07-30 23:05:53 UTC`（2026-07-31 08:05:53 JST）
  - metadata pullで`NvidiaTeslaT4`の反映を再確認した。

## 2026-07-31 train version 2 ERROR

- status: `KernelWorkerStatus.ERROR`
- kernel death: `1395.195735877 sec`、最終log: `1406.978336262 sec`
- failure: `nbclient.exceptions.DeadKernelError: Kernel died`
- Stage 0契約表示後、fold完了stdoutは0件、Kaggle filesも0件だった。
- 完了・再利用可能なboosterは0。旧実装はCatBoostとXGBoostをfold単位で
  両方完了した時だけcountを表示するため、CatBoost outer0が内部で完了していた
  可能性までは除外できないが、再利用できるmodel出力はない。
- 分類: host RAM peak超過と判断した。以下はversion 2時点の初期仮説であり、
  version 3の進捗logにより直接停止点はStage 0後処理へ絞られた。
  - final370のfold0 train / valid生float32行列を保持したまま、
    `feature_matrix_sha256()`が`array.tobytes()`で約4--5 GiBを複製していた。
  - さらに生行列とcompact/signed DataFrameを保持したままCatBoost内部Poolを
    構築していた。
  - exp274の同規模CatBoost成功実装はPool構築後に生行列を解放してからfitしており、
    version 2の明示Python例外なしのkernel deathとも整合する。
- 科学結果: CV、family audit、stack、confidence gateは未到達。

## 2026-07-31 version 3 memory fix

- candidate、feature、fold、target、CatBoost/XGBoost parameter、stack/gateは変更しない。
- SHA計算を`memoryview`によるzero-copy更新へ変更した。SHA byte列の内容は同一。
- CatBoostはexp274と同じく`Pool`を構築し、生行列とfold DataFrameを解放してから
  fitする。
- CatBoost終了後に同じfold surfaceを再読込し、Stage 0のmatrix SHAと再照合して
  XGBoostだけを実行する。これによりCatBoostとXGBoostの巨大行列を直列保持する。
- Stage 0 fold preflight、CatBoost Pool ready、XGBoost matrix readyを
  fold単位でstdoutへ表示し、次回の停止点と完了model数を追跡可能にした。
- version 3実行量:
  - active variants: 2
  - model configs: CatBoost 1 + XGBoost 1 = 2
  - outer folds: 5
  - 新規GPU models: 10
  - exp413 control / selector / signed selector再学習: 0
  - 新規PF/HMM/Beam: 0
- version 2の再利用可能出力は0のため、version 3は10 modelsを実行する。
  control再学習は含まれず、元のtrain実行承認範囲を維持する。
- 専用testは11件PASS。Jupytext round-trip、`py_compile`、F821もPASSした。
- 親compact 9章766行に対し、更新後exp494は9章2394行。
- push前GPU quotaは`16.19 h / 45.00 h`残存。
- version 3 package:
  - notebook bytes: `850,726`
  - notebook SHA256:
    `f799dc16d4da97efac3d28898cda9a38011141387ebe68fa93041b3c20d5a4d9`
  - embedded / loose config SHA256:
    `d66ed543b46acbfbeda296dc954f39df4ea5cb1185d2da0a1769e19cb61ca17d`
  - config byte parity: PASS
  - private / T4 / internet off / run-on-push: PASS
  - package notebook内のzero-copy SHA、CatBoost Pool後release、
    XGBoost fold再読込、control再学習0: PASS
- 同じcanonical private T4 kernelへversion 3をpushした。
  - kernel version: 3
  - id_no: `129213293`
  - started確認: `2026-07-31 03:42:38 UTC`
  - status: `KernelWorkerStatus.RUNNING`
  - push後pull: private / GPU / internet off /
    `machine_shape=NvidiaTeslaT4` / input source 9件を確認
  - pullしたremote notebookでもzero-copy SHA、CatBoost Pool、
    XGBoost fold再読込、進捗logを確認

## 2026-07-31 train version 3 ERROR

- status: `KernelWorkerStatus.ERROR`
- kernel death: `1317.984938071 sec`、最終log: `1328.763579883 sec`
- failure: `nbclient.exceptions.DeadKernelError: Kernel died`
- Stage 0 clean273: 3,783,989 rows / 773 wells / 273 base featuresまで完了。
- Stage 0 matrix preflight: outer fold 0--4の5/5を完了。
- 最終preflight log: `1235.228607064 sec`。
- `family_train/fold_start`は0件で、CatBoost / XGBoost完了modelは`0 / 0`。
- reusable output / CV / family audit / stack / gate: なし。
- 分類: Stage 0後処理のhost RAM peak。
  - fold DataFrameを5回生成・解放した後のallocator resident memoryを
    `gc.collect()`だけではOSへ十分返せていなかった。
  - 続く3,783,989行の`physical_candidate_oof.parquet`生成は、
    6列の親copy、候補ID、予測列、Parquet変換を全行同時保持していた。
  - preflight 5/5完了からkernel deathまで約82.8秒で、
    `family_train`未到達というlog evidenceに一致する。

## 2026-07-31 version 4 memory fix

- 科学contract、candidate、feature、fold、target、model parameter、
  stack/gate、10-model数は変更しない。
- `release_process_memory()`で`gc.collect()`後にLinux `malloc_trim(0)`を呼び、
  fold preflight / model間の解放済みarenaをOSへ返す。
- `physical_candidate_oof.parquet`は全行DataFrameを作らず、
  250,000 rowsずつPyArrow `ParquetWriter`へ書く。
- Stage 0 post-preflight、物理OOF書込完了、Stage 0完了、family fold start、
  surface load、train matrix完了にRSS / high-water mark付きlogを追加した。
- 次段の潜在peakも予防した。
  - `assemble_matrix`は全base列の300万行copy後に32列を選ぶ順序をやめ、
    32列を先に選んでからfold行を抽出する。
  - finite検証を全matrix boolコピーではなく32,768-row chunkで行う。
  - CatBoost Pool構築前にcompact/signed fold DataFrameを解放する。
- version 3は学習0本で、version 4も2 variants / 2 configs / 5 folds /
  10 GPU models、control・selector・PF/HMM/Beam再学習0。
- 専用test 12件、Jupytext round-trip、`py_compile`、F821をPASS。
- 親compact 9章766行に対し、更新後exp494は9章2543行。
- push前GPU quotaは`15.82 h / 45.00 h`残存。
- version 4 package:
  - notebook bytes: `858,418`
  - notebook SHA256:
    `df8f4f533e17aca9e051cb7e08b47fac66d85f9e2a35ef21f44301a9c3b2b285`
  - embedded / loose config SHA256:
    `65b2a4819e670790618c09fca28ca17db7239df27e4a2c91083e745b3fb151e1`
  - config byte parity: PASS
  - private / T4 / internet off / run-on-push: PASS
  - package notebook内のallocator trim、chunk Parquet、列先行matrix、
    chunk finite、RSS logs、control再学習0: PASS
- 同じcanonical private T4 kernelへversion 4をpushした。
  - kernel version: 4
  - id_no: `129213293`
  - started確認: `2026-07-31 04:18:51 UTC`
  - status: `KernelWorkerStatus.RUNNING`
  - push後pull: private / GPU / internet off /
    `machine_shape=NvidiaTeslaT4` / input source 9件を確認
  - remote notebookでもallocator trim、chunk Parquet、列先行matrix、
    chunk finite、Stage 0後処理logを確認

## 2026-07-31 train version 4 ERROR

- status: `KernelWorkerStatus.ERROR`
- kernel death / 最終log: `2372.486 / 2383.616 sec`
- Stage 0 clean273: 3,783,989 rows / 773 wells / 273 base featuresまで完了。
- Stage 0 matrix preflight: outer fold 0--4の5/5を完了。
- Stage 0後処理:
  - fold manifest ready: RSS `15.375 GiB`
  - chunk物理OOF書込完了: RSS `14.719 GiB`
  - Stage 0 complete: RSS `15.277 GiB`
- family train outer fold 0:
  - fold start: RSS `15.357 GiB`
  - compact/signed surface load後: RSS `18.341 GiB`
  - train final370 matrix生成後: RSS `22.715 GiB`
  - CatBoost train / valid Pool生成完了
  - 実行中high-water mark: `27.526 GiB`
- CatBoost / XGBoost完了model: `0 / 0`
- Kaggle files / reusable output / CV / family audit / stack / gate: なし。
- failure: `nbclient.exceptions.DeadKernelError: Kernel died`
- 分類: CatBoost fold 0 fit開始時のhost RAM peak。Python例外ではなくkernel全体が
  停止しており、Pool生成までは完了した。固定数値370特徴に対するCatBoost内部
  量子化が、常駐clean273 DataFrame / Poolと重なってRAM上限を越えたと推定する。
  GPU OOMなら通常はCatBoost例外となるため、直接logとRSS推移はhost OOMと整合する。
- version 4でStage 0全行Parquet peakは解消したため、version 3の停止原因診断は
  有効だった。ただしfamily学習開始後の別peakが顕在化した。

## 2026-07-31 version 5 disk-backed memory fix

- candidate、final370内容・順序、fold、target、CatBoost / XGBoost parameter、
  stack / gate、10-modelコスト契約は変更しない。
- clean273の273特徴だけをStage 0完了後にfloat32 NPY memmapへ列chunkで書き、
  `id / well / target / last_known_tvt / md_since`だけをmemoryへ残して
  273列DataFrameを学習前に解放する。
- family学習ではfold行列をmemmapから32列ずつ再構成する。組み立て後はmappingを
  閉じ、既存Stage 0のfold別matrix content SHAと完全一致を再確認してからfitする。
- CatBoost Poolはtrain Pool生成直後にtrain raw matrixを解放し、その後valid Poolを
  生成してvalid raw matrixも解放する直列順序へ変更した。
- runtime cacheはfamily学習終了時または通常Python例外時に削除し、科学成果物や
  artifact SHAへ含めない。
- 小規模testでDataFrame経路とmemmap経路の行列値・matrix SHA完全一致を確認した。
- 専用test 13件、Jupytext round-trip、`py_compile`、F821、
  strict experiment / template validationをPASSした。
- version 4は学習0本・再利用model 0のため、version 5も2 variants / 2 configs /
  5 folds / 10 GPU models、control・selector・PF/HMM/Beam再学習0である。
- push前GPU quotaは`15.15 h / 45.00 h`残存。
- version 5 package:
  - notebook bytes: `867,859`
  - notebook SHA256:
    `486f408f7a4148acb0d124b9c0c1eadb9a28d2c6bfb6f0c3d65b52a2340b2818`
  - embedded / loose config SHA256:
    `471a9d23fb9ae8dddb0811bf896583faf303f4b7eb1d5bb4141fad30752aa0a0`
  - config byte parity: PASS
  - private / T4 / internet off / run-on-push: PASS
  - package内memmap source / CatBoost raw matrix直列解放: PASS
- 同じcanonical private T4 kernelへversion 5をpushした。
  - kernel version: `5`
  - id_no: `129213293`
  - status: `KernelWorkerStatus.RUNNING`
  - remote metadata: private / GPU / internet off /
    `machine_shape=NvidiaTeslaT4`
  - remote notebookはlocal packageと21 cellのsourceが完全一致し、
    source SHAは
    `3b13e0eacbcf45a548da427b6abc73ef610291bd80b8c3724a44dc79fc530b4a`
  - remote embedded config SHA、memmap fix、CatBoost raw matrix直列解放を確認

## 2026-07-31 train version 5 COMPLETE / terminal close

- kernel / version / id_no:
  `kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-train` /
  `5` / `129213293`
- status: `KernelWorkerStatus.COMPLETE`
- train summary elapsed: `5187.904674 sec`
- rows / wells / final features: `3,783,989 / 773 / 370`
- 実行量:
  - CatBoost: `5 / 5 models`
  - XGBoost: `5 / 5 models`
  - parent LightGBM再学習: `0`
  - selector / signed selector再学習: `0`
  - 新規PF/HMM/Beam: `0`
- memory fix実測:
  - Stage 0 complete RSS: `15.018 GiB`
  - 4,132,116,116-byte clean273 memmap生成・DataFrame解放後:
    RSS `10.722 GiB`
  - CatBoost Pool ready RSS: fold 0で`17.242 GiB`、fold 1--4で
    `17.722--17.833 GiB`
  - process high-water mark: `29.767 GiB`
- family pooled RMSE:
  - saved exp413 LGB: `7.884802794`
  - CatBoost: `8.108026060`（parent比`+0.223223266 ft`）
  - XGBoost: `8.052470087`（`+0.167667292 ft`）
  - fixed physics: `8.070218794`（`+0.185416000 ft`）
- OOF-level cross-fit bounded stack:
  - candidate RMSE: `7.827450885`
  - exp413比gain: `0.057351909 ft`
  - nonworse folds: `5 / 5`
  - fold delta: `-0.028693 / -0.124335 / -0.078905 /
    -0.012167 / -0.038771 ft`
  - deployment weights:
    LGB `0.681703` / Cat `0.103730` / XGB `0.014568` /
    Physics `0.200000`
- scope / hidden-like deltaは全て改善:
  - near 0--250: `-0.018691 ft`
  - mid 250--1000: `-0.011009 ft`
  - far 1000+: `-0.064420 ft`
  - hidden-like spatial: `-0.072030 ft`
  - hidden-like typewell-purged: `-0.068393 ft`
- scientific AND gate:
  - technical / minimum gain / 5-fold / fixed scope / hidden-like: PASS
  - by-well p95: `+0.634421 ft`でFAIL（上限`0.0 ft`）
  - worst well: `+3.843641 ft`でFAIL（上限`+0.25 ft`）
  - improved / worsened wells: `426 / 347`
  - overall: `FAIL`
- 固定fail actionに従い、選択予測は`exp413_lgb`、selected RMSEは
  `7.884802794`。conditional gateは評価せず、weight / candidate /
  parameter / bound / threshold rescue、inference、submissionを行わない。
- selective output確認:
  - 10 model file、metrics、fold/scope/hidden、weights、OOF、
    feature/model/reproducibility manifestの存在を確認
  - model manifest SHA:
    `20617536a2735613a9b311fd2db4b83f386d0043d9f497fd77d0c331c3daab2c`
  - OOF SHA:
    `9cc8eb305d55a2d587d4519b0ce51e23cf263e595af0c95793d0f2d1d3f96a38`
  - weight SHA:
    `8630a41c74c7ba2dd8adca61cb444f328e24df369f2a173a3d8a222237d2ee9f`
  - reproducibility manifest SHA:
    `f086f09dede5ab48de6021888a51908a208c3a0fbbabc9506f067397e59d9cac`
- reproducibility manifestの既知の軽微な欠陥:
  - `exp494_train_metrics.json`をmanifest生成後に最終rewriteしたため、
    manifest内の同file SHA `11c33a...830d`は古い。
  - 最終metrics実体SHAは
    `97c80de5a69d1e1ba03812c1c2b1de9b97eb24bbe76c52cb22d69614d1115dc7`。
  - 最終metrics内のreproducibility manifest SHAは実体
    `f086f0...9cac`と一致し、model / OOF / weight / fold / scopeの
    selective readback SHAも一致する。科学判定は再分類しない。
- 保存した小規模証拠:
  `kaggle/output/train_v5_selected/`。model本体や全OOF archiveは
  terminal FAILのためローカルへ丸ごと取得していない。

## 次のアクション

1. exp494はscientific terminal FAIL、exp413をML / submitted anchorとして維持する。
2. 2026-07-31のユーザー明示overrideにより、version 5 constant stackだけを
   hidden-safe参考提出する。conditional gate、routing、trajectory後処理、
   same-OOF rescueは行わない。
3. compact self-contained inferenceを検証・正規Notebook化し、Kaggle T4で実行する。
4. output取得、submit-check PASS後にcompetitionへ提出し、採点完了まで監視する。

## 2026-07-31 reference submission override / inference implementation

- ユーザー指示: 「とりあえず今の実装で提出まで進みたい」
- 解釈: train guard FAILは撤回せず、version 5で凍結したconstant bounded stackの
  hidden inference・submissionだけを明示overrideする。
- 使用weight: LGB `0.681702678534061` / CatBoost
  `0.10372958993775055` / XGBoost `0.01456773152818835` / Physics `0.2`。
- 追加処理: なし。conditional gate、well-level routing、trajectory後処理、
  weight/candidate/threshold再推定、Public-LB補正は禁止。
- 保存model: selector 40、signed selector 20、LGB 15、CatBoost 5、XGBoost 5。
  inference中のbooster trainingは0。
- 親exp413 current-test inference version 4をhidden-safe再生成基盤として、
  Jupytext percent形式の別名compact inference sourceへ実装を開始した。
- 親compact inferenceは7章 / 1,563行、exp494候補は同じ7章 / 1,799行。
  追加分はexp494 authorization、10 family model SHA、凍結weight、
  constant-stack / global fallback / submission監査であり、親のraw-test再生成章を維持した。
- `*_compact_selfcontained_inference.py`から候補Notebookを生成し、ユーザーの
  提出指示に基づき正規`*_inference.ipynb`へ同じcell sourceを採用した。
- validation:
  - `py_compile`: PASS
  - `ruff --select F821`: PASS
  - Jupytext `--to ipynb --test`: PASS
  - exp494専用test: `14 passed`
  - strict `make validate-exp`: PASS
  - `make validate-template`: PASS
- Kaggle inference package:
  - kernel: `kentookumura/exp494-exp413-cat-xgb-physics-bounded-stack-inference`
  - title: `exp494 exp413 cat xgb physics bounded stack inference`
  - private / T4 / internet off / run-on-push: PASS
  - kernel sources: exp413 hidden-safe 11 sources + exp494 train version 5 source
  - packaged notebook bytes / SHA256: `867,263` /
    `58359eda470eb77ffbe6ba4c2563a57407bf767f94ac8686eeef22444b148109`
  - source / packaged loose / embedded config SHA256:
    `adc3128c2e718a1ce2ff51d18a1614131e78034d80b5bba61b2c64390a49a490`
  - bootstrap members: `54`、parent exp413 / exp335 / candidate contract / compact
    inference sourceを含むことを確認した。
- 初回canonical候補`exp494-exp413-cat-xgb-physics-bounded-stack-inference`
  （53文字）は`SaveKernel 400`。pullは403、mine listにも存在せず、Kaggle側に
  notebookが作成されていないことを確認した。train slugは49文字で成功済みのため、
  title由来slugとの上限不一致を原因候補とし、意味を保った49文字未満の
  `exp494-exp413-cat-xgb-physics-stack-inference`へ短縮して再packageする。
- 短縮slug package:
  - id/title slug parity: PASS
  - private / T4 / internet off / run-on-push: PASS
  - source / loose config SHA256:
    `b9e607d36cb87239fd9a4fc68b0849b132f526709826d71982e0466cfa87708f`
  - packaged notebook SHA256:
    `c6b1b244b1f4981811de829166c4bf7bca976d0f3642360f3d8adcb661822b2b`
  - exp494専用test `14 passed`、strict validation PASS。
- inference v1:
  - kernel / version / id_no:
    `kentookumura/exp494-exp413-cat-xgb-physics-stack-inference` /
    `1` / `129268848`
  - status: `KernelWorkerStatus.COMPLETE`
  - artifact生成完了時刻: `393.921 sec`
  - public output rows / wells: `14,151 / 3`
  - training calls / fallback rows: `0 / 0`
  - root `submission.csv`: `464,082 bytes`
  - SHA256:
    `29fc30575fb0bc528f6550e7f3e2158c764641e3988ffb0e5174119d643c510e`
  - sample ID/order exact、duplicate 0、NaN/Inf 0、finite 100%。
  - prediction min / max / mean / std:
    `11593.370324 / 12239.283867 / 11905.117342 / 277.873281`。
  - exp413 public output比: 全14,151行変更、mean abs `0.565796 ft`、
    p95 abs `1.462802 ft`、max abs `1.955841 ft`。
  - skill checker: FAIL 0 / WARN 0、repo `make submit-check`: PASS。
- competition submission:
  - ref / submitted at: `55134873` / `2026-07-31 10:38:28.727000 UTC`
  - message: `exp494 constant stack reference override; tail guard FAIL retained`
  - final status / Public LB: `COMPLETE / 7.228`
  - monitor runtime: `268 min`
  - exp413 Public LB `7.201`比: `+0.027`悪化
  - scientific selectionは`exp413_lgb`、train tail guardは`FAIL`のまま維持する。
  - exp494は不採用。overall / ML submitted anchorはexp413を維持する。
  - route別の数値参照では、従来exp082 `7.601`を上回るためexp494を
    ensemble-route Public-LB referenceとするが、robust promotionとは扱わない。
