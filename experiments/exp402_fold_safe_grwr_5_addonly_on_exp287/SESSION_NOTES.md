# exp402_fold_safe_grwr_5_addonly_on_exp287 セッションノート

## 目的

exp264 availability auditでformation依存として除外されたGRWR 5列を、
exp287のfold-safe formation roleから再計算し、exp287 421列へadd-onlyする
実験。設計確定後の追加指示により、Stage 0の別名Jupytext候補、専用test、
0-booster preflightを実装する。正規Notebook採用とKaggle実行は行わない。

## 現在の状態

- Route: `ml_model`
- 状態: Stage 0 implementation complete・未実行
- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- CV / LB: なし
- steering:
  `.steering/20260726-exp402-fold-safe-grwr-5-addonly-on-exp287/`
- 正規notebook / `settings.py`: 未編集placeholder
- 別名Jupytext train/inference候補: 実装済み
- 専用test: 8件PASS
- Kaggle package / run: なし

## 2026-07-26 設計確定

### 根拠

- exp264監査では、無効GRWR 6列のうち5列が候補spreadへ旧formation依存
  `tvt_dense*`を含めたこと、1列が旧exp111 entropyへ依存したことを分離している。
- exp287は`tvt_dense_d / tvt_densew_d / tvt_dense50_d`をouter fold内で再生成し、
  3列はいずれもformation 74のgain上位だった。
- したがってexp402ではformation依存5列だけを独立仮説にし、exp396 entropy依存
  interactionは含めない。

### 固定特徴

1. `grwr_candidate_tvt_std`
2. `grwr_candidate_tvt_range`
3. `grwr_dwt_energy_ratio_w065_x_candidate_std`
4. `grwr_fft_rotation_ratio_x_candidate_range`
5. `grwr_dwt_minus_raw_ncc_gap_x_candidate_range`

候補TVTは`pf_ancc`のabsolute値と、`last_known_tvt`へ
`beam_mean_d / likpf_mean_d / sc_ens_d / hyb_d /
tvt_dense_d / tvt_densew_d / tvt_dense50_d`を加えた計8候補とする。
float32 stack、`ddof=0`の標準偏差、max-min range、固定3 interactionを使う。

### 固定入力

- exp287 OOF SHA:
  `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA:
  `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 feature schema SHA:
  `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`
- exp287 formation fold manifest SHA:
  `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- corrected exp264 OOF SHA:
  `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- exp218 availability audit SHA:
  `6f93a502c9b58301e49da6effbf47b36d4635d4045157681749a762f08c89c67`
- rows / wells / folds: `3,783,989 / 773 / 5`

### 0-booster preflight

- model / booster / prediction / submission: `0 / 0 / 0 / 0`
- exp287のouter-train self-excluded roleとouter-valid outer-train-only roleを使う。
- current-testはraw inputから全train wellsをreferenceに再生成し、静的なpublic-test
  feature artifactを入力にしない。
- 5列のschema、finite、重複なし、outer-role別logical-content SHA、
  target formation read 0、旧GRWR値/score系不使用をAND gateにする。

### 条件付き学習量

実装と実行は別承認が必要。

- scientific variant: 1
- LightGBM configs: 3
- outer folds: 5
- GPU boosters: `1 × 3 × 5 = 15`
- exp287 / exp264 control再学習: 0
- final feature count: `421 + 5 = 426`

promotion gateはexp396と同じexp287/exp264比較面を固定し、pooled
`<= -0.02 ft`、4/5 folds、全scope`<= +0.02 ft`、by-well p95`<=0`、
worst-well vs exp264`<=+0.25 ft`、exp264比+1/+3/+5 ft悪化well数
`<=135/39/14`の全ANDとする。

## 2026-07-26 Stage 0実装

ユーザーの「exp402を実装してください」をimplementation-onlyの承認として扱い、
次を追加した。

- `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_train.py`
- 同名の別名候補`.ipynb`
- `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_inference.py`
- 同名のfail-closed候補`.ipynb`
- `experiments/exp402_fold_safe_grwr_5_addonly_on_exp287/tests/test_exp402_fold_safe_grwr_5_addonly_on_exp287.py`

train候補は11 numbered chaptersで、親exp287の保存artifact/SHA、exp264 OOF、
availability allowlist、exp072 candidate contextをfail-closed検証する。
旧exp218 generator全体は呼ばず、固定source/config SHAを確認したうえで、
必要なDWT detail energy ratio、FFT rotation energy ratio、likpf default-candidate
DWT-minus-raw NCC gapの3成分だけを同じ式でraw GRから再生成する。

outer foldごとにexp287のmatching train/valid formation partitionを読み、
train self-exclusionとvalid outer-train-only referenceをmanifestで検証する。
固定8候補からfloat32 / `ddof=0`の5列を作り、10 partitionそれぞれの
file SHA、schema SHA、id-sorted float32 logical-content SHAを保存する。

current-testは静的なtest feature artifactを使わない。SHA固定したexp072 sourceで
raw test 3 wellsを再生成し、全773 train wellsをformation referenceにして
target formation read 0を検証する。固定実行量は次のとおり。

- PF ANCC: 3 well-runs
- PF Z: 3 well-runs
- Beam: 21 paths
- likelihood-PF: 3 well-runs / 384 seed-well trajectories /
  192,000 particle starts（128 seeds × 500 particles × 3 wells）
- model / booster / final prediction / submission: `0 / 0 / 0 / 0`

exp072のPF乱数はsource内のfeature family / split / well由来stable SHA256 seedを
使用する。実行前なのでcurrent-test content SHAのdeterministic anchorはまだ
主張しない。Stage 1のLightGBM学習とinferenceは未実装のままfail closedにした。

親exp287 trainは9 markdown cell / 362行、今回の候補は11 numbered chaptersの
compact self-contained構成である。行数は機能追加により大きいが、親helperの
全文貼り付けではなく、Stage 0に必要なpath/SHA guard、GR 3成分、role cache、
current-test orchestrationだけを内包した。

## 2026-07-26 Stage 0実行承認

- ユーザーの「実行してください」により、正規train Notebook採用、
  Kaggle private CPU package / push / Stage 0 runを承認済みとして記録した。
- 実行量:
  - scientific model variant / model config / trained fold / booster:
    `0 / 0 / 0 / 0`
  - current-test PF ANCC / PF Z / Beam path:
    `3 / 3 / 21`
  - likelihood-PF:
    `3 well-runs / 384 seed-well trajectories / 192,000 particle starts`
  - parent control retraining / final prediction / submission: `0 / 0 / 0`
- Stage 1の1 variant / 3 configs / 5 folds / 15 GPU boosters /
  control再学習0、inference、submissionは承認対象外で、自動実行しない。
- runtime: CPU、internet off、GPU off、private、run-on-push true。
- canonical kernel:
  `kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
- canonical title:
  `exp402 foldsafe grwr5 on exp287 train`
- Kaggle inputs:
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`
  - `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train`
- Kaggle OAuth credentialを確認した。API tokenは未設定だが、CLI実行に使う
  OAuthとlegacy credentialは有効。
- 別名候補を正規train Notebookへ採用した。採用後SHAは
  `9dbe5ef95037abfe65332a171edce4ad451025f3548b03f9cffc1752bf214dec`。
  正規inference NotebookはSHA
  `80dada427766981550774db25c8c7ccb7c776b5d2fb096eca5d3eb2619aa5017`
  のplaceholderを維持する。
- package監査で、SHA固定済みexp072 sourceの読み込みを動的module loadから
  repository namespaceの静的importへ変更した。科学式、seed、実行量は不変。
- 採用後もJupytext round-trip、pycompile、Ruff、専用test `8 passed`、
  strict `validate-exp`をPASSした。
- strict package生成をPASSし、bootstrap内config / exp072 source /
  exp218 source+config / availability audit / `src`の存在とSHAを確認した。
- version 1をpushした。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - id_no: `128627922`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - Kaggle pullでprivate、internet/GPU off、入力3件を再確認した。
- 現在version 1を監視中。同じkernel idのまま完了を待ち、空logsだけを
  理由に再pushしない。
- 約45分間、同じversion 1を監視し、最終確認時点のstatusは
  `KernelWorkerStatus.RUNNING`。実行中のCLI logsは空だった。
- ユーザーの「監視は止めていいです。完了したら連絡します。」により、
  Codex側のpollingを停止した。Kaggle kernelはキャンセルせず実行継続。
  完了連絡後に同じkernel id/versionのlogsと必要なSHA生成物だけを回収する。

## コマンドログ

Stage 0承認前は学習・推論・Kaggleコマンドを実行していない。

```bash
make new-steering EXP=exp402_fold_safe_grwr_5_addonly_on_exp287
make new-exp EXP=exp402_fold_safe_grwr_5_addonly_on_exp287 SOURCE=templates/experiment
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <candidate.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <candidate.py>
.venv/bin/python -m py_compile <train.py> <inference.py> <test.py>
.venv/bin/ruff check <train.py> <inference.py> <test.py> --select F821,F401,E9
.venv/bin/pytest -q experiments/exp402_fold_safe_grwr_5_addonly_on_exp287/tests/test_exp402_fold_safe_grwr_5_addonly_on_exp287.py
make validate-exp EXP=exp402_fold_safe_grwr_5_addonly_on_exp287
```

検証結果:

- pycompile: PASS
- Ruff `F821,F401,E9`: PASS
- Jupytext round-trip train/inference: PASS
- 専用test: `8 passed`
- strict `validate-exp`: PASS
- exp218 synthetic formula parity: selected 3 source componentsが`atol=1e-6`で一致
- 正規train/inference notebook SHA:
  `dc7040ad...e0a` / `80dada42...017`で変化なし

## 再現性メモ

- feature seed policy: RNGなし、exp287のfold/row identityを継承
- downstream seed: exp287の42を継承
- stochastic components: 将来承認後のGPU LightGBMだけ
- train PF/Beam/HMM: 保存済みtarget-free候補値のload-only、再実行0
- current-test PF/Beam: fixed exp072 sourceとper-well stable seedでStage 0時に再生成
- feature evidence: outer-roleごとのid-sorted float32 logical-content SHA
- gzip evidence: decompressed content SHA
- GPU: DP/deterministic/force_col_wise/threads 8を固定するがbitwise anchorとは呼ばない
- model / prediction / submission SHA: 現時点では非該当
- Kaggle kernel id / version: 未作成

## 禁止事項

- 旧exp218 GRWR 5列の値
- `grwr_ll_entropy_x_dwt_energy_ratio_w065`
- exp396 score 27列またはexp111 score
- feature/candidate/formula/threshold/grid探索
- sample weight、error-segment weight、hard gate、direct TVT correction
- exp287 / exp264 control再学習
- 同一OOF救済、gate緩和、promotion前の推論・提出

## 2026-07-26 Stage 0 version 1終了確認

- 23:03 JST時点のKaggle statusは
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。
- `lastRunTime`は`2026-07-26T00:06:39.493000`（09:06:39 JST相当）。
  status確認まで約13時間56分であるためruntime上限によるcancelが最有力だが、
  Kaggle APIは手動cancelとruntime-limit cancelを区別しない。
- 保持ログは15.354秒の
  `Implementation candidate only: ...`を最後に途切れ、tracebackはない。
  これは`run_experiment(CONFIG)`によるStage 0本処理へ入る直前の出力であり、
  以後の詳細な停止地点はログから特定できない。
- `kaggle kernels files`は0件。preflight manifest、10 role partitions、
  current-test partition、feature content SHAは完成物として取得できない。
- technical gateはPASS/FAILではなく未完了。model / booster /
  final prediction / submissionは`0 / 0 / 0 / 0`のまま。
- 同じversion 1の再pushは行っていない。retryはconfigどおり別の
  ユーザー承認を必要とする。

## 次のアクション

再実行する場合は、role partition生成とcurrent-test PF/Beam replayを分割するか、
完成済み中間cacheを明示的に受け渡す設計へ変更し、各runの実行量とSHA境界を
再監査してから承認を得る。Stage 0 PASS後も15 GPU booster学習、推論、提出は
自動では行わない。

## 2026-07-26 分割retry実装・実行

- ユーザーの「設計変更と再実行を進めてください」により、科学仕様を変えず
  Stage 0を次の3 private CPU runへ分割することと再実行を承認済み。
  1. 0A `train_roles`: train source components 3列と10 outer-role partitions
  2. 0B `current_test`: current-test 3 wellsのraw PF/Beam replay
  3. 0C `aggregate`: A/Bのsource/config/file/content SHAとtechnical gate統合
- model / booster / final prediction / submissionは各runとも`0 / 0 / 0 / 0`。
  0Bの固定量はPF ANCC 3、PF Z 3、Beam 21 paths、likelihood-PF
  3 well-runs / 384 seed-well trajectories / 192,000 particle starts。
- wrapper 3件、phase manifest、同一execution identity、
  upstream immutable file SHA再検証を実装した。
- Jupytext round-trip、pycompile、Ruff、strict `validate-exp`をPASS。
  専用testはaggregate synthetic file SHA検証を追加して`10 passed`。
- 0A/0B/0C packageを同じ状態から一括生成した。
  - config SHA:
    `98dd377e2b7bbfe3daf4340aaefbca5be97b393d267ce939319fb114bf406176`
  - implementation source SHA:
    `665f41ad628c3fdb04ec5fa595c9e38378ed5bbf6f475a0c646cdda29b6cfb20`
  - private、CPU、internet off、run-on-push true
- Stage 0A version 1をpushした。
  - kernel: `kentookumura/exp402-foldsafe-grwr5-train-roles`
  - id_no: `128687498`
  - status: `KernelWorkerStatus.RUNNING`
  - Kaggle pullでprivate、CPU、internet off、exp072/exp287/exp264の
    input 3件を確認した。
- Stage 0Bの初回pushは
  `Maximum batch CPU session count of 5 reached.`で受理されなかった。
  exp410の4 shardとexp402 0AがRUNNINGで5枠を使用中。別実験をcancelせず、
  自然完了で1枠空き次第、同じ凍結済み0B packageだけを再pushする。
- 0C packageは0A/0Bと同じconfig/sourceで凍結済み。A/B完了後までpushしない。

## 次のアクション

CPU枠が空き次第0Bをpushする。0A/0Bのlogsと必要なSHA生成物を確認し、
両方PASSした場合だけ0Cをpushする。Stage 1の15 GPU boosters、inference、
submissionは自動実行しない。

## 2026-07-27 train-roles cancel確認とouter-fold shard化

- 18:35 JSTの確認でStage 0A
  `kentookumura/exp402-foldsafe-grwr5-train-roles` version 1は
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。保持ログは13.859秒の
  `Stage 0A train-role generation started`で終了し、tracebackとoutputは0件。
- 実装監査で、生成前にexp287の約1 GiB級formation role parquet 10件を
  すべて物理SHA走査し、その後さらに10件をreadして論理content SHAを計算する
  二重I/O境界を確認した。科学式の失敗ではなく、technical gateは引き続き未完了。
- ユーザーの承認済み「設計変更と再実行」および「残りを実行」の範囲で、
  科学仕様を変えず次の8 private CPU runへ再分割した。
  1. `train_source`: 親OOF/exp264/exp072整合、GR source 3列生成、
     formation role read 0
  2. `train_fold0` ... `train_fold4`: 各runが該当outer-foldの
     train/valid 2 partitionだけを物理SHA・論理content SHA検証して生成
  3. `current_test`: 固定3 well raw replay
  4. `aggregate`: 7 upstream outputと10 role ledgerをSHA統合
- 実行量は全runでmodel / booster / final prediction / submission
  `0 / 0 / 0 / 0`。Stage 1計画は1 variant / 3 configs / 5 folds /
  15 GPU boosters / control 0のままで未承認。
- 合成aggregate testを含む専用test `10 passed`、pycompile、Ruff
  `F821,F401,E9`、Jupytext round-trip、strict `validate-exp`をPASS。
- 8 packageを同じ状態から生成した。
  - config SHA:
    `82bcce7c7d6e0694ffc67a2898068213e68d1b52cb560f89f63f8788f701bce0`
  - implementation source SHA:
    `7098ebe2063faeaee2d0d9b65d910648777da7726f682679ce1f17a8548c4ac4`
  - private、CPU、internet off、run-on-push true

## 2026-07-27 fold 4実行

- ユーザーの「fold4を実行してください」により、凍結済みStage 0
  `train_fold4` packageの実行を承認済み。
- 元slug `kentookumura/exp402-foldsafe-grwr5-train-fold4` は、前回の
  CPU quota失敗後にstatus 404である一方、再pushが
  `Notebook not found`となる予約済み・参照不能状態だった。
- 科学コード、config、入力、実行量は変更せず、既存current-testと同じ
  operational workaroundとしてslug/titleだけを`-v2`へ変更した。
  後段aggregateの`kernel-metadata.json`もfold 4 sourceだけ`-v2`へ更新した。
  aggregateのconfig上のcanonical identityとsentinel-based artifact探索は維持。
- 21:46 JSTにKaggleへpushし、次を確認した。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-train-fold4-v2`
  - version: `1`
  - id_no: `128789362`
  - status: `KernelWorkerStatus.RUNNING`
  - private、CPU、internet off、run-on-push true
  - config SHA:
    `82bcce7c7d6e0694ffc67a2898068213e68d1b52cb560f89f63f8788f701bce0`
  - fold wrapper SHA:
    `534871782097ed6d69946b887be9b4ae7bc7a105ef35d0007cec662c21c316d5`
- このrunのmodel / booster / final prediction / submissionは
  `0 / 0 / 0 / 0`。
- ユーザーの事前指示どおり、受理・起動確認後の継続監視は再開しない。

## 2026-07-28 outer-fold完了確認とStage 0 aggregate実行

- ユーザーからfold 4完了の連絡を受け、Kaggle logsで全outer-fold shardを
  再確認した。fold 0–4はいずれも`passed: true`、2 partitions、
  3,783,989 rowsで完了した。
  - fold 0: 13,837.057秒
  - fold 1: 15,614.736秒
  - fold 2: 7,493.623秒
  - fold 3: 13,728.845秒
  - fold 4 v2: 14,368.183秒
- fold 4 v2は次の5生成物をKaggle outputへ保存した。
  parent schema、partition manifest、fold manifest、outer-fold 4の
  train parquet、valid parquet。tracebackはなく、末尾のstderrは
  nbconvert/mistuneのSyntaxWarningだけ。
- 既確認PASSの`train_source`、`current_test-v2`と合わせてStage 0の
  upstream 7件が揃ったため、凍結済みaggregate packageをpushした。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-stage0-aggregate`
  - version: `1`
  - id_no: `128831850`
  - 2026-07-28 06:26 JST status: `KernelWorkerStatus.RUNNING`
  - private、CPU、internet off
  - kernel sources: train source 1、fold shard 5、current-test 1
  - model / booster / final prediction / submission:
    `0 / 0 / 0 / 0`
- push後のKaggle metadata pullでfold 4/current-testの`-v2` inputを含む
  7 sourceと上記runtime設定を確認した。
- ユーザーの事前指示どおり、aggregateの受理・起動確認後は継続監視しない。

## 2026-07-28 Stage 0 aggregate version 1失敗調査

- ユーザーから失敗連絡を受け、Kaggle logsを取得した。version 1は
  実行開始約22秒後、`run_stage0_aggregate()`内のfold 4 artifact root解決で
  `FileNotFoundError`となった。
- 直接原因:
  - configのfold 4 primary pathは旧canonical slug
    `/kaggle/input/notebooks/kentookumura/exp402-foldsafe-grwr5-train-fold4/artifacts`
    のまま。
  - 実際のinputはquota失敗後の予約済みslugを回避した
    `...train-fold4-v2/artifacts`。
  - primary path不在のためsentinel fallbackへ進んだが、全foldが同名の
    fold manifest / partition manifestを持つため、fold 0–4の5 rootを候補に
    して一意性guardがfailした。
- 分類はKaggle input path / resolver contractのcode defect。
  fold shardの欠損、SHA不一致、coverage不一致、OOM、runtime上限ではない。
  upstream 7 runのPASS結果は無効化されない。
- 既存testのaggregate synthetic検証は
  `resolve_train_fold_artifact_root`をmonkeypatchしており、同名sentinelを持つ
  5 inputが同時mountされた実環境の一意性を検証していなかった。
- 前節の「canonical identityとsentinel-based artifact探索は維持」という
  判断は不十分だった。fold 4だけは同名sentinelが5件あるためfallback不能。
- 再実行はまだ行っていない。config fileまたはcompact implementation sourceを
  変更するとupstream manifestのexecution identity SHAと不一致になるため、
  最小修正はaggregate wrapper内だけで、read済みconfigのfold 4 runtime pathへ
  `...train-fold4-v2/artifacts`を明示的に先頭追加すること。
  file SHAで使うconfigとcompact implementation sourceは変更せず、
  package notebookだけをversion 2として再生成・pushする。

## 2026-07-28 Stage 0 aggregate version 2修正・再実行

- ユーザーの「修正と再実行してください」により、version 1のpath解決defectを
  修正し、同じcanonical aggregate slugへversion 2をpushすることを承認済み。
- aggregate wrapperがscientific contractを検証した後、in-memory configの
  fold 4 artifact patterns先頭へ次を追加するwrapper-only hotfixを実装した。
  `/kaggle/input/notebooks/kentookumura/exp402-foldsafe-grwr5-train-fold4-v2/artifacts`
- config fileとcompact implementation sourceは変更していない。
  - config SHA:
    `82bcce7c7d6e0694ffc67a2898068213e68d1b52cb560f89f63f8788f701bce0`
  - compact implementation source SHA:
    `7098ebe2063faeaee2d0d9b65d910648777da7726f682679ce1f17a8548c4ac4`
  - version 2 aggregate wrapper SHA:
    `7558165751a7bd63f0f5593b4d3e1a35e3c96a7274ad289fbac9e69600e79cc0`
- 同名fold manifest / partition manifestを持つ5 rootが存在しても、明示した
  fold 4 pathが先に一意解決される回帰testを追加した。
- 検証結果:
  - 専用test: `11 passed`
  - Jupytext round-trip: PASS
  - pycompile: PASS
  - Ruff `F821,F401,E9`: PASS
  - strict `validate-exp`: PASS
  - package JSON、bootstrap manifest、fold4/current-test `-v2` source: PASS
- `task` CLIは環境に存在しなかったため、規約上同等の
  `make prepare-kaggle-notebooks`でaggregate packageだけを再生成した。
- 06:39 JSTに同じkernelへversion 2をpushした。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-stage0-aggregate`
  - version: `2`
  - id_no: `128831850`
  - status: `KernelWorkerStatus.RUNNING`
  - private、CPU、internet off
  - input: train source 1、fold shard 5、current-test 1
  - model / booster / final prediction / submission:
    `0 / 0 / 0 / 0`
- push後のmetadata pullで7 inputとfold4 runtime aliasを確認した。
- ユーザーの事前指示どおり、受理・起動確認後の継続監視は行わない。

## 2026-07-28 Stage 0 aggregate version 2完了

- ユーザーの完了連絡後に一度だけKaggle logs、終端status、公開outputを確認した。
- kernel:
  `kentookumura/exp402-foldsafe-grwr5-stage0-aggregate`
  - version: `2`
  - id_no: `128831850`
  - status: `KernelWorkerStatus.COMPLETE`
  - private、CPU、internet off
- aggregateは`8.19382479699999 sec`、peak RSS `0.24061203002929688 GiB`で
  `zero_booster_preflight_passed`となり、`18 / 18` checksをPASSした。
- manifestで次を確認した。
  - upstream train source / current-test / outer-fold 0–4: 全PASS
  - outer-role partition: `10`
  - current-test: `14,151 rows / 3 wells`
  - parent / added GRWR / final feature: `421 / 5 / 426`
  - historical GRWR values loaded: `0`
  - current-test target formation columns read: `false`
  - model / booster / prediction row / submission row:
    `0 / 0 / 0 / 0`
  - planned GPU train:
    `1 variant / 3 LightGBM configs / 5 folds / 15 boosters`
  - control再学習: `0`
- execution identity:
  - config SHA:
    `82bcce7c7d6e0694ffc67a2898068213e68d1b52cb560f89f63f8788f701bce0`
  - compact implementation source SHA:
    `7098ebe2063faeaee2d0d9b65d910648777da7726f682679ce1f17a8548c4ac4`
  - scientific contract SHA:
    `8b2befb44bc22b6e62675edda48c90c0593932b815c00fab0915e817c24e6635`
- aggregate生成物の取得ファイルSHA:
  - partition manifest:
    `704d8a9163f5a82c9b28f3866e2de6d3a7dfac78a4236b5352436431674f365b`
  - preflight manifest:
    `c8af15ad8502b172031eaa862878ba07f2a94b6eba5913259c3a4ba0e5142de8`
  - reproducibility manifest:
    `5456cfd9b0d2df3cac5848cb234cf382f9ebd1515439742fcff6afa3f0560fda`
- Stage 0 technical gateをPASSとして確定した。CV追加価値はまだ未評価。
- 次のGPU trainはStage 0とは別承認。既存exp287 OOFをbaselineに使い、
  controlを再学習せず、1 variant × 3 configs × 5 folds = 15 GPU boostersだけを
  実装・実行する。明示承認前はtrain、inference、submissionを実行しない。

## 2026-07-28 Stage 1 GPU train実装・実行承認

- ユーザーの「実装と実行を進めてください」により、Stage 1 train実装、
  正規train Notebook採用、Kaggle T4 package / push / runを承認済み。
- push前の固定実行量:
  - scientific variant: `1`（`fold_safe_grwr_5_addonly`）
  - LightGBM config indices: `[0, 1, 2]`
  - outer folds: `5`
  - GPU boosters: `1 × 3 × 5 = 15`
  - exp287 / exp264 control再学習: `0`
  - accelerator: `NvidiaTeslaT4`
  - internet: off
  - inference / submission: disabled
- Stage 0 aggregate version 2の18 / 18 PASS、10 outer-role partition、
  各fold shardのmanifest/file SHAを学習開始前に再検証する。
- clean-273、nested compact-74、fold-safe formation-74、GRWR-5を
  `273 + 74 + 74 + 5 = 426`列の固定順で組み立てる。
- exp287の3 configとJSON等価なLightGBM parameter familyだけを使う。
  保存済みexp287/exp264 OOFを固定controlとして読み、control boosterは作らない。
- promotion gateは設計済みの全AND条件を維持する。Stage 1が完了しても
  promotion PASSと別承認が揃うまではinference / submissionをfail closedにする。
- Stage 1 Jupytext source:
  `exp402_fold_safe_grwr_5_addonly_on_exp287_compact_selfcontained_stage1_train.py`
  - source SHA:
    `e4df2ed570b42fdfc32d123e136168a3390e7d4ca62ea58dbb82b560ecf14e63`
  - 9 numbered chapters、20 cells（markdown 11 / code 9）
  - `__file__`不使用
- 別名候補Notebookのround-trip後、正規train Notebookへ採用した。
  candidate / canonical SHA:
  `bc090bceb11b1cb46d58235c007ae9a5b504572f4dae166f8ac36dbaa68c057e`
- inference候補はStage 1実装済み状態を認識するよう更新したが、
  `run_inference=false`、`create_submission=false`を維持する。
- push前検証:
  - pycompile: PASS
  - Ruff `F821,F401,F811,E9`: PASS
  - Jupytext round-trip: PASS
  - 専用test: `12 passed`
  - strict `validate-exp`: PASS

## 2026-07-28 Stage 1 version 2失敗とversion 3修正

- canonical kernelへversion 2をT4指定でpushし、status `RUNNING`を確認した。
- version 2はNotebook開始`10.6418`秒後、学習前のStage 0 aggregate root解決で
  `FileNotFoundError`となった。completed boosterは`0`。
- 直接原因は、Kaggle input metadataにはaggregate kernelが含まれていた一方、
  実mount pathがconfigの固定absolute pathと一致せず、Stage 1 resolverが
  sentinel / file SHA探索へfallbackしていなかったこと。
- 科学仕様、特徴、LightGBM config、fold、promotion gateは変更しない。
  resolverへ次のfail-closed修正を加えた。
  - Kaggle input rootからrequired manifest名を探索する。
  - 同名manifestが複数あるfold shardは、Stage 0 aggregateが固定した
    manifest / partition file SHAの両方で一意に選ぶ。
  - exp287 rootも固定artifact SHAで選ぶ。
- Kaggleの実行中versionに対する`pull -m`はmaterialized previous versionの
  `enable_gpu=false / machine_shape=None`を返す場合がある。この曖昧性を
  runtime evidenceにしないため、重い入力処理より前に`nvidia-smi`で
  `NVIDIA T4`を要求するhardware guardを追加した。T4以外では0 boosterで停止する。
- version 3候補:
  - Stage 1 source SHA:
    `cbcd5f3cd7676d477dba6a46eea217e1bb83a464e91dc262c845199980a2d6b1`
  - canonical train Notebook SHA:
    `68c0d20fe66f25ae9c7376bc3c71018d933f6c87fef5c4f405bb975b6bdc5549`
  - dedicated tests: `13 passed`
- version 2でboosterを学習していないため、retry後も承認済み総学習量は
  1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0のまま。
- strict `validate-exp`、Jupytext round-trip、pycompile、Ruff、
  dedicated tests `13 passed`、package bootstrap SHA監査を再度PASSした。
- canonical kernelへversion 3をpushした。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - version: `3`
  - id_no: `128627922`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - status: `KernelWorkerStatus.RUNNING`
  - requested accelerator: `NvidiaTeslaT4`
  - private、internet off、kernel inputs 10
  - package config SHA:
    `98915abf6641da87ac7e798e135b35b0b827697856aafee4c148cec971892723`
  - package Notebook SHA:
    `5bf975ca46414332901ebc94e3953c4f3d51ecf3756e43002904c74e30a0f88a`
  - package metadata SHA:
    `324ab304a7bfd6cbff477296713ec7c87b712e62d873b0e645a02c2532e081bf`
- remote source pullでSHA-qualified resolver、T4 runtime guard、aggregate input、
  fold4-v2 inputを含む10 inputが反映されたことを確認した。
- 2026-07-28 18:29:44 JST時点でversion 2の失敗時点を越えてRUNNING。
  CLI logsはまだ空。受理・起動確認後は継続監視しない。

## 2026-07-28 Stage 1 version 3失敗とversion 4修正

- ユーザーの失敗連絡後、同じcanonical kernelの終端statusとlogsを取得した。
- version 3は物理T4を正常確認した。
  - GPU: Tesla T4 × 2
  - driver: `580.159.04`
  - memory: `15,360 MiB / GPU`
- Stage 0 aggregateとfold rootのSHA-qualified mount解決も通過した。
- `227.2871`秒後、clean-273再構築の
  `exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz`
  読み込みで`FileNotFoundError`となった。LightGBM学習前なのでboosterは`0`。
- 原因は、exp287と同じclean-273再構築に必要な4 inputのうち
  `exp145-train`だけをStage 1のkernel sourcesへ含めていなかったこと。
  必要な残り3 input（exp072、exp264 Stage C / Stage D）は既に添付済み。
- Kaggle repositoryには13 kernel sourceの既存package実績があるため、
  10 sourceを維持するための科学contract変更は行わず、
  `kentookumura/exp145-train`を11番目のinputとして追加する。
- version 4のfail-fast guard:
  - kernel sourcesが固定11件かつ重複0であること。
  - exp145のfull-train ML features、feature schema、summaryの3ファイルが
    Kaggle input上に存在すること。
  - 3ファイルのpath / bytes / SHAをStage 1 preflightへ記録すること。
- version 4候補のSHA:
  - Stage 1 source:
    `1d7666edfe81e274f1a0db58499dd465b5a5724c80b661a028e4004765774491`
  - canonical train Notebook:
    `310e3fd356d8f444761d571a78533c9a83ed2c87304798c281ca147e7874ef55`
- version 2 / 3はいずれも0 boosterのため、retry後も承認済み学習量は
  1 variant × 3 configs × 5 folds = 15 T4 boosters、control再学習0。
  inference / submissionは引き続き無効。
- version 4候補はpycompile、Ruff `F821,F401,F811,E9`、Jupytext round-trip、
  strict `validate-exp`、Kaggle notebook test `4 passed`、専用test
  `13 passed`、package bootstrap / SHA監査をPASSした。
- canonical kernelへversion 4をpushした。
  - kernel:
    `kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - version: `4`
  - id_no: `128627922`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp402-foldsafe-grwr5-on-exp287-train`
  - 2026-07-28 18:41:05 JST status: `KernelWorkerStatus.RUNNING`
  - requested accelerator: `NvidiaTeslaT4`
  - private、internet off、kernel inputs 11
  - package config SHA:
    `3aa1cdde16141acd2a385b228feefdc4e96e600f999cab036e694060b741df1c`
  - package Notebook SHA:
    `a503778e0841f5910a7600b5deb0aec789deb77acb4289d48ef8dadc1e49c29f`
  - package metadata SHA:
    `816907a0c430621df07c5d69315c7beef771011a41a7f4f35a913980f25a097e`
- post-push remote sourceでexp145 input、固定11 input guard、exp145必要3ファイル
  guard、物理T4 guard、internet offが反映されたことを確認した。
- 受理・起動確認後はユーザー指示どおり継続監視しない。version 4完了連絡後に
  OOF、fold / scope / by-well、importance、model manifest SHAを取得し、
  固定promotion gateを判定する。
