# exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264 セッションノート

> **結果無効:** exp265のregime raw contextにもtraining-only formation 6列があり、exp264 scoreも無効。
> structure/scoreの全readoutを実行再現用にだけ保持し、hidden-safeな診断には使わない。

## 目的

非TVT train-side情報と候補パス間の差・傾向からtarget-free regimeを作り、複数soft expertを
学習する価値があるかを0-boosterで先に判定する。

## 現在の状態

- Route: ensemble
- 状態: version 2完了・feature availability leakageにより結果無効
- Stage 0 CV: 無効
- LB / submission: scope外
- 親exp264: feature availability leakageによりStage B score/guard/readout無効

## 実行量契約

- Stage 0: 0 variant、0 LightGBM config、0 fold training、0 booster。
- 親/control再学習: 0。
- Kaggle runtime: CPU、GPU off、internet off。
- 入力kernel source: exp263 Stage 0 cache、exp264 Stage B score/model manifest/metricsの2件。
- Conditional Stage 1: disabled。3 regimes × 2 objectives × 5 folds = 30 CPU boostersの旧計画は破棄。
  exp264 global selectorをfallbackに使わない。

## コマンドログ

### 2026-07-17 実装

```bash
make new-steering DATE=20260717 EXP=exp265 TITLE=target-free-pairwise-candidate-divergence-soft-experts-on-exp264
make new-exp EXP=exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264
.venv/bin/ruff check src/candidate_pairwise_regime.py tests/test_candidate_pairwise_regime.py
.venv/bin/python -m pytest -q tests/test_candidate_pairwise_regime.py
```

- `docs/legacy/steering/20260717-exp265-target-free-pairwise-candidate-divergence-soft-experts-on-exp264/`
  に仮説、Stage 0/1境界、guardを固定した。
- `src/candidate_pairwise_regime.py`にprimitive cache reader、block fingerprint、outer-fold KMeans、
  centroid-matched stability、exp264 Parquet batch audit、artifact writerを実装した。
- targeted unit testは5件。15 pair、target-free schema、共通offset耐性、cluster再現性、
  in-memory/streaming score auditを検証する。
- Kaggle notebookは未実行。Kaggle pushも未実行。

## 変更点

- `regime_contract.yaml`: 6 primitiveとfamily、全15 pair、post-assignment score policy。
- `config.yaml`: 512-row block、K=3、raw/confidence/pair/bank allowlist、Stage 0 guard、実行量。
- `src/candidate_pairwise_regime.py`: 候補差分regimeの再利用実装。
- train notebook: input fail-closed、fold別fingerprint、OOF regime、streaming score audit、artifact/SHA保存。
- inference notebook: Stage 0では常に停止し、submissionを生成しない。

## 再現性メモ

- seed policy: KMeans主seed `42 + outer_fold`、監査seed `10042 + outer_fold`。
- stochastic components: KMeans initializationのみ。
- parallel RNG: sklearnの`random_state`を明示しglobal RNGを使わない。
- CPU/GPU runtime: Kaggle CPU予定、GPU off。
- Kaggle kernel id / version: 未実行。
- input SHA: exp263 manifest/catalogを固定。exp264 score/model manifest/metrics SHAは実行時に保存。
- exp264 canonical SHA: score `e51bb674...45a5a`、model manifest `12375038...4c9a`、
  selector metrics `568140aa...c16`をconfigへ固定。local score fileは過去の0-byte partialのため使わず、
  Kaggle kernel source上の実体をSHA検証して読む。
- feature schema/content SHA: 実行時に保存。
- centroid/assignment SHA: artifact SHAとして実行時に保存。
- model / prediction / submission SHA: Stage 0では対象外。
- deterministic anchor: false。Kaggle rerun確認前はanchorと呼ばない。

## 次のアクション

1. conditional Stage 1、inference、submissionは実行しない。旧記録の「exp264 global selectorを維持」は撤回し、利用禁止とする。
2. 再訪する場合はblock-length proxyと外れ値支配を除く別実験の0-booster固定監査として提案する。

## 2026-07-17 静的検証・Kaggle package監査

- Jupytext train/inference変換と`--test`、`py_compile`、`ruff --select F821`をPASSした。
- targeted test 5件、repository全74 testsをPASSした。ローカルnotebook実行は行っていない。
- `make validate-exp` strictとexperiment doc reviewをPASSした。
- canonical packageをtrain `kentookumura/exp265-pairwise-regime-audit-train`、inference
  `kentookumura/exp265-pairwise-regime-audit-inference`としてprepareした。
- metadataはprivate、CPU、GPU/TPU/internet off、`run_on_push=false`。train入力はcompetition source、
  exp263 Stage 0、exp264 Stage Bの2 kernel source。inferenceはdisabledで誤実行すると停止する。
- package内のsource parityを確認した。config SHA `ee1ea786...139b`、regime contract
  `28737b23...7d9`、`src/candidate_pairwise_regime.py` `ecd51917...6ede`はlocal/package一致。
- package notebook SHAはsupport bootstrap追加後の`ad05ac48...bf0`。source notebook SHA
  `d26a5290...955`とは用途どおり異なる。
- exp264 scoreはlocalに残る0-byte partialを入力根拠にしない。Kaggle kernel source上で
  canonical SHA `e51bb674...45a5a`を照合してからfingerprint生成へ進む。

## 2026-07-17 Stage 0実行承認

- ユーザーからKaggle Stage 0の実行承認を受領した。
- 実行対象はpairwise regime separability audit 1 notebook。LightGBM variant 0、config 0、
  fold training 0、booster 0、親/control再学習0。KMeansはouter 5 folds × 主/監査seedでfitする。
- runtimeはKaggle CPU、GPU/TPU/internet off。入力はcompetition、exp263 Stage 0、exp264 Stage B。
- `execution.run_approved=true`へ変更した。conditional Stage 1の30 CPU boosters、inference、
  submissionは引き続きdisabledで、今回の実行scopeに含めない。

## 2026-07-17 Stage 0 push前監査

- strict validate-expとtargeted 5 testsを再度PASSし、experiment summaryを更新した。
- canonical train packageを`kentookumura/exp265-pairwise-regime-audit-train` / 
  `exp265 pairwise regime audit train`として再生成した。id/titleのslugは一致する。
- metadataはprivate、CPU、GPU/TPU/internet off、`run_on_push=true`。inputはcompetition source、
  exp263 Stage 0、exp264 Stage Bの2 kernel sourcesである。
- package内configはStage 0 0 variant / 0 config / 0 fold training / 0 booster、
  control再学習0、`run_approved=true`。conditional Stage 1 30 boostersとinferenceはdisabled。
- source/package SHAは一致した。config `85decbe2...8ef2`、regime contract
  `28737b23...7d9`、pairwise regime module `ecd51917...6ede`。package notebook SHAは
  `0c5f33ee...ae0f`。

## 2026-07-17 Stage 0 Kaggle version 1実行開始

- canonical kernel `kentookumura/exp265-pairwise-regime-audit-train`へpushし、
  `Kernel version 1 successfully pushed`を確認した。Kaggle id_noは`127531288`。
- pullしたKaggle metadataはprivate、CPU、GPU/TPU/internet off、exp263/exp264の2 kernel source、
  competition sourceでpackageと一致した。
- `KernelWorkerStatus.RUNNING`を確認した。実行中の通常logsは空だが、このCLIでは完了前に空となる
  既知挙動のため、再pushやslug変更は行わない。
- 実行scopeは承認どおりStage 0 0 booster、control再学習0。Stage 1、inference、submissionは実行しない。

## 2026-07-17 Stage 0 Kaggle version 1失敗・guard修正

- version 1は約58秒、fold 0のblock fingerprint schema検査でERRORとなった。入力解決、exp263/exp264
  SHA、leakage contract表示まではPASSし、artifact生成とStage 1学習には到達していない。
- tracebackは`confidence__<primitive>__sigma_tvt__median` 6列をforbiddenと誤判定した
  `ValueError`。configではtarget-free confidenceとして`sigma_tvt`を許可する一方、guardが列名に
  substring `tvt`を含むだけで拒否していた自己矛盾が原因である。
- guardを`__`区切りsegmentの完全一致判定へ変更した。`sigma_tvt`は許可し、`tvt`、`tvt_input`、
  `true_tvt`、`candidate_tvt`、`last_known_tvt`、actual/pred error等は明示segmentで拒否する。
- production相当confidence slotsを使い、`sigma_tvt`許可、`raw__tvt` / `candidate_tvt`拒否を確認する
  回帰テストを追加した。targeted 6 tests、repository全79 tests、Jupytext、py_compile、F821、
  strict validate-expをPASSした。ローカルnotebook実行は行っていない。
- 修正はfeature schema guardとそのテストだけ。候補、512-row block、K=3、fold、seed、入力、
  Stage 0 guard、0 booster契約は変更していない。再実行は同じcanonical slugのversion 2とする。

## 2026-07-17 Stage 0 version 2 push前監査

- strict validate-exp、targeted 6 tests、repository全79 tests、Jupytext、py_compile、F821をPASSした。
- 同じcanonical id/titleでprivate CPU packageを再生成した。GPU/TPU/internet off、
  `run_on_push=true`、exp263/exp264の2 kernel sources、competition sourceはversion 1と同じ。
- package内configは`run_approved=true`、Stage 0 0 booster、conditional Stage 1 / inference disabled。
- source/package SHAは一致した。config `ce7949c9...778c`、regime contract
  `28737b23...7d9`、pairwise regime module `136012d4...6583`。package notebook SHAは
  `28df33e1...881d`。

## 2026-07-17 Stage 0 Kaggle version 2再実行開始

- 修正版を同じcanonical kernel `kentookumura/exp265-pairwise-regime-audit-train`へpushし、
  `Kernel version 2 successfully pushed`を確認した。
- 初期statusは`KernelWorkerStatus.RUNNING`。version 1の誤検出を修正したpackageが起動したことを
  確認し、ユーザー指定どおり継続監視は行わない。
- scopeはStage 0 0 booster、control再学習0のまま。conditional Stage 1、inference、submissionは
  disabledであり、今回の再実行に含めない。

## 2026-07-17 Stage 0 Kaggle version 2完了・guard監査

- canonical kernel `kentookumura/exp265-pairwise-regime-audit-train` version 2は
  `KernelWorkerStatus.COMPLETE`。生成物表示まで331.158秒で、例外なく正常完走した。
- 3,783,989 rows / 773 wells / 7,787 blocks、295 features、6 primitives / 15 pairsを監査した。
  exp264 candidate-longは22,703,934 rowsを91 Parquet batchesでstreaming集計した。
- Stage 0最終判定はFAIL。occupancyだけがFAILし、stabilityと数値separabilityはPASSした。
  block occupancyはregime 0/1/2 = 171/411/7,205 blocks = 2.20%/5.28%/92.53%。
  各foldでもregime 2が91.1--93.5%で、各regimeに10%以上を要求するguardを満たさなかった。
- primary/audit KMeansのcentroid-matched assignment agreementは全5 foldsで1.000。一方、
  global mean soft probabilityは0/1/2 = 0.021999/0.052997/0.925004、平均最大確率は0.997257、
  entropy中央値は`6.04e-08`で、soft membershipは実質hard assignmentだった。
- block assignmentとfingerprintを選択取得して原因診断した。regime 1相当は5/5 foldsで
  median 151--162 rows、full 512-row block率0%のterminal partial block。dominant regime 2は
  median 512 rows、full block率96.5%。outer-train RobustScaler空間のcentroid距離は、terminal対
  dominantで`raw__md__range`、`raw__md__end_minus_start`、`raw__md__std`がほぼ100%を占めた。
- 残る0--2.2%の極小clusterは、存在するfoldでは主に`selfgr_hmm_a070`対`exact_hmm`の
  `gap_mean`/`gap_end`/`gap_slope`外れ値が距離を支配した。地質的・一般的な候補trend regimeではなく、
  partial-block機構とrare path failureを分離した結果である。
- terminal clusterのbest candidateは5/5 foldsで`exp226_k16`。dominant clusterも4/5 foldsで
  `exp226_k16`、fold 3だけ`selfgr_hmm_a070`。stage summaryのpooled best family 2件と
  calibration bias range 1.013964 ftは数値guardを通ったが、fold 2でterminal cluster labelが
  regime 1から0へ入れ替わり、極小clusterとpooledされたためexpert根拠として採用しない。
- target-free schemaは295 features、forbidden hits 0。feature schema SHAは`f703ce4e...de4c`、
  block fingerprint logical SHAは`be391fec...6cf`、block/row assignment SHAは
  `0e548140...730` / `b1fb5271...7eec`。exp263/exp264入力SHAはconfig固定値と一致した。
- guard監査に必要な小さいsummary/CSV/manifestと13MB block fingerprint、320KB block assignmentだけを
  選択取得した。3,783,989-row assignmentやoutput archive全体は取得していない。
- Trust assessment: 実行とoccupancy FAIL判定はTrustworthy。pooled regime family/calibration PASSは
  fold間label非整合と極小cluster混入のためexpert separability証拠としてはNot trustworthy。
- 当時のdecision文字列`reject_conditional_expert_training_keep_saved_exp264_global_selector`は撤回した。
  正式には`reject_conditional_expert_training_structure_occupancy_failed_exp264_score_invalidated`。
  Stage 1の30 CPU boosters、inference、submissionは未実行のまま閉じる。
