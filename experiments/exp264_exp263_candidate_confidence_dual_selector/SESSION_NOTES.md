# exp264_exp263_candidate_confidence_dual_selector セッションノート

## 目的

exp263候補bankを入力の正とし、候補数拡張、候補confidence追加、selector特徴量整理、dual scoreからcompact meta-featureへのfold-safe変換を一つの実験契約へ固定する。

## 現在の状態

- Route: `ml_model`
- 状態: 旧Stage A/B/CとStage D compact add-onlyをfeature availability leakageで無効化。
  修正版Stage A v4で88列を凍結し、修正版Stage B v5とnested Stage C v6を完了。
  Stage Cはscore guard / leakage audit PASS、hard top1 FAIL。clean 273 downstreamは固定済み。
  修正版Stage D version 3は30/30 GPU boostersを完走。overall/fold/bucket/hidden-likeは改善したが、
  worst-well +14.482873により事前guard FAIL。ユーザー明示overrideの修正版推論・提出は完了し、
  Public LB 7.562で直前ML anchor exp274 7.715を更新
- CV: 修正版Stage Bのselector score OOF、Stage C nested compact、Stage D downstream TVT OOFは有効。
  Stage Dはclean 273 control 10.476169、347列add-only 8.460811（-2.015358）。
  旧Stage B/C/D数値は比較・診断・採用根拠として使用禁止
- LB: 7.562、submission ref 54818932。Public-LB上のML route anchor
- 完了scope: 修正版Stage Dとcorrected inference/reference submission。Stage Dは
  2 variants × 3 configs × 5 folds = 30 GPU boosters。
  内訳はclean 273 matched control 15本、clean 273 + compact 74 = 347列add-only 15本。
  corrected inferenceは学習0本で完了。competition submissionもCOMPLETE。学習local run gateは閉鎖済み。

## 2026-07-19 修正版推論・提出override

- ユーザーから「今回の実験の推論・提出に進む」明示指示を受領した。Stage D worst-well
  `+14.482873`のguard FAILは保持し、hidden-safe推論とsubmit-check PASS後のcompetition submit 1件だけを
  例外scopeとする。再学習、hard selector、Viterbi、candidate softmax平均は含めない。
- 推論は修正版Stage C v6の88列・40 selector modelと、修正版Stage D v3のclean 273 + compact 74 =
  347列`selector_compact_addonly` 15 TVT modelだけを使用する。新規学習0、matched control推論0。
- 旧Stage C v3 bundleの代わりにv6の40 model、88列schema、74列compact schema、manifestを含む
  44-entry bundleを生成した。bundle SHAは`0e1ae1a5...b5adf`、サイズ約13 MB。
- private Dataset `kentookumura/exp264-stage-c-selector-models`へ
  `corrected Stage C v6 88-feature selector models for exp264 inference`として新versionをuploadした。
- 推論configをStage C manifest `3f28b04a...e2d2`、feature schema `b91ec151...1035`、Stage A catalog
  `3a443a1a...1235`、Stage D v3 manifest `c3b22481...5fcc`へ更新した。
- 推論model contractをselector 88、compact 74、source base 380、clean base 273、final 347へ修正した。
  source 380 catalogとclean 273 allowlistのSHA・件数・一意性・モデル列順をfail-closedで確認する。
- notebook自身のsubmit API呼び出しはfalseのままにし、Kaggle output取得後のsubmit-check PASSを外部提出条件にした。
- py_compile、ruff F821/F401/E9、exp264 targeted 15 tests、YAML/JSON parseをPASSした。

## 2026-07-16 設計判断

- candidate countは7ではなく12とする。
- 12 = raw-test-ready primitive 6 + raw-test pair 5 + fixed `exp226_w500_50_50` 1。
- full exp263 inventory 33、Stage 0 core 12、virtual/namedを含む23 surfaceは全体inventoryであり、現行Stage 1で生成・parity確認済みの12 surfaceとは分ける。
- exp264のscore対象12本はすべてexp263 Stage 1でcurrent-test生成済みである。追加6 primitiveは`stage0_oof_only_not_in_current_stage1_primitives`として記録し、原理的な生成可否ではなく「現行Stage 1出力に未収録」であることを表す。
- 追加6 primitive、追加3 pair、outer-fold fitted formula 2本は現行current-test bankの分布を変えるため、Stage 1を拡張してparity確認するまではselectorへ混ぜない。
- 12 scoreは保持するが、hard-domainはprimitive+pair 11とprimitive+fixed 7に分ける。
- candidate IDはstringをartifactに残し、modelへはone-hot 12を入れる。
- source-native confidenceは全候補に同一scalarを要求しない。全候補へuniversal proxyを作り、common learned confidenceをdual scoreとする。
- exp251 v4 295列はraw-test-safe context seed。旧候補固有列を落とし、exp263 bankへ再計算する。
- outputはexp251型candidate別dual scoreを正とし、exp238型compact metaは同じprocess内のadapterとする。
- Viterbiを作らない。full score CSVを推論依存にしない。HMM+LGBもscope外。

## 2026-07-16 候補tier表記の修正

- exp263 Stage 1の実出力`current_test_formula_parity.parquet`を確認し、exp264のscore対象12本がすべてcurrent-test生成済みであることを再確認した。
- 旧`train_only_primitives` / `train_only_pairs`表記を廃止し、`stage0_oof_only_not_in_current_stage1_primitives` / `stage0_oof_only_not_in_current_stage1_pairs`へ変更した。
- このtierは生成不可能という意味ではなく、現行Stage 1出力へ未収録であることだけを表す。将来Stage 1を拡張してparity確認できればexp264候補追加の再評価対象にできる。
- candidate contract、config、metrics、README、steering、調査docs、backlogを同期し、契約テスト4件とstrict experiment validationを通した。train/inference Kaggle packageも再生成済みで、pushは未実行。

## 学習コスト契約

| Stage | 状態 | variant | objective/config | fold | booster | device | control再学習 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| A feature audit | Kaggle v1完了 | 0 | 0 | 0 | 0 | CPU | なし |
| B selector-only OOF | Kaggle v2完了 | 1 | 2 objectives | 5 outer | 10 | CPU | なし |
| C nested compact | Kaggle v3完了 | 1 | 2 objectives | 5 outer × 4 inner | 40 | CPU | なし |
| D TVT matched ablation | 修正版Kaggle T4 v3完了・guard FAIL | 2 | 3 configs | 5 outer | 30 | GPU | あり、15 control + 15 add-only |

Stage Bの承認は消化済みで再実行しない。Stage Cは承認scopeの40 CPU boostersだけを実行する。
Stage Dは承認scopeどおり30本だけを実行する。追加variant/config/foldは別承認とする。

> 上表と直前の説明はavailability audit前の履歴。旧Stage B/C/Dは無効で、その実行承認も
> 修正版へ自動継承しない。現契約は以下の2026-07-18修正版を正とする。

### 2026-07-18 clean 273選択と修正版Stage B承認

- ユーザーはdownstreamにclean 273列案を選択。非fold-safe 107列を再生成して380列へ戻す案は不採用。
- allowlist: `artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv`
- allowlist SHA256: `d01a73cc28485345dd86ed56ad6276f1727dca6b270d87685e1cf578afb677bf`
- Stage D実装は履歴的source 380列を組み立てた後、SHA・source 380列・selected 273列・重複0・
  allowlistとの列順完全一致をfail-closedで検証し、モデル入力を273列に限定する。
- 将来のStage D比較面はmatched control 273列 / selector compact add-only 347列。
- 現在承認済みの実行は修正版Stage Bの1 variant × 2 objectives × 5 folds = 10 CPU boostersだけ。
- 修正版Stage Cは40 CPU boosters、Stage Dは30 GPU boostersだが、どちらも未承認で実行しない。
- Stage Bにはmatched control、exp218 TVT model、GPU boosterを含まない。

### 2026-07-17 Stage D実装と実行承認

- ユーザーからfull matched ablationの明示承認を受領した。
- 実行対象は`matched_control` 380列と`selector_compact_addonly` 454列、exp218固定3 config、
  Stage C固定5 downstream outer folds。合計30 GPU boostersで、内訳はcontrol 15、add-only 15。
- Stage C version 3のmetrics/model/compact/schema固定SHAに加え、25 compact partitionのbyte SHAを
  学習前に検証する。metadataだけの不完全downloadは入力として採用しない。
- outer-trainはStage C inner OOF compact、outer-validは4-inner-model ensemble compactだけを使う。
- exp218 base surface、LightGBM family、`gpu_repro_guard_dp_threads8`、early stopping 250を固定した。
- OOF、30 model manifest、gain/split重要度、fold/bucket/hidden-like/by-well readoutを保存する。
- hidden-like assignmentは事後評価専用で、学習、early stopping、model選択へ使用しない。
- Stage D実装に合わせて契約テストを9件へ拡張し、canonical Jupytext train notebookを同期した。

### 2026-07-16 Stage A実行承認

- ユーザーからStage A実行の明示承認を受領した。
- 実行対象は`execution.stage=feature_contract_audit`のみ。0 variant、0 objective/config、0 fold training、0 booster、Kaggle CPU、親/control再学習なし。
- `execution.run_approved=false`を維持し、Stage B 10 CPU boosters、Stage C 40 CPU boosters、Stage D 30 GPU boostersは今回実行しない。
- push直前にsource/packageの`config.yaml` SHA一致、`candidate_contract.yaml` SHA一致、private/CPU/internet off/run_on_push、canonical kernel id/titleを確認した。

### 2026-07-16 Stage A初回push復旧

- 初回slug `kentookumura/exp264-exp263-candidate-confidence-dual-selector-train`（54文字）はKaggle `SaveKernel 400`で実行前に失敗した。
- 同IDの`kaggle kernels pull -m`は403で未作成を確認し、親`kentookumura/exp263-last-anchor-pair-cache-train`のpullは成功した。入力source欠落ではない。
- 過去の53文字以上のslugで同じ400になった記録に基づき、同じexp264のまま意味を保持した44文字の`kentookumura/exp264-exp263-confidence-dual-selector-train` / `exp264 exp263 confidence dual selector train`へID/titleを同時に短縮する。
- Stage Aの設定、候補、特徴量、booster数は変更しない。新slugでpackageを再生成してから再pushする。

### 2026-07-16 Stage A Kaggle v1結果

- kernel: `kentookumura/exp264-exp263-confidence-dual-selector-train` version 1、id_no `127485868`、private、CPU、internet off。
- 153.612秒で正常完了し、ログ上で`0 variants / 0 configs / 0 folds / 0 boosters`を確認した。
- exp263入力は3,783,989 rows / 773 wells / 5 folds。manifest `85e60ac1...a26bb9e`、catalog `7cd74866...e9e6e0`が期待値と一致した。
- fold別合計600,000 candidate-long rowsを監査し、162特徴候補から100列を採用した。全欠損41、定数5、完全重複16を除外し、|Pearson|または|Spearman|が0.999以上の35組はreport-onlyで維持した。
- feature schema SHAは`766cfcf10a14fdcd0aa6f6ff78f347c1fb4f3eb86f95138b8676d11da96d4deb`、audit-long content SHAは`62d84bcf39c86846d6fe58dbd35781b904bbdfc06089a0816a48c1f11cb4896d`。
- compact metaは74列、schema SHA `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
- 採用100列中21列がsource-native confidenceまたはformula親confidenceに依存する。Stage AのOOF feature contractはPASSだが、Stage B前にexp263 Stage 1 current-testへnamespaced confidenceを追加してparityを閉じる。
- 実行中の`DataFrame is highly fragmented`はfeature列追加時の性能警告で、監査値・生成物・終了statusには影響しなかった。Stage B前に`pd.concat`型の一括構築へ最適化する候補とする。
- 必要な実ファイルを`kaggle/output/stage_a_v1/artifacts/`へ取得した。selector model、OOF prediction、submissionは生成していない。

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp264_exp263_candidate_confidence_dual_selector
make new-exp EXP=exp264_exp263_candidate_confidence_dual_selector
.venv/bin/pytest -q experiments/exp264_exp263_candidate_confidence_dual_selector/tests/test_exp264_candidate_selector_pipeline.py
.venv/bin/python -m py_compile src/candidate_selector_pipeline.py
.venv/bin/ruff check src/candidate_selector_pipeline.py experiments/exp264_exp263_candidate_confidence_dual_selector/exp264_exp263_candidate_confidence_dual_selector_train.py experiments/exp264_exp263_candidate_confidence_dual_selector/exp264_exp263_candidate_confidence_dual_selector_inference.py --select F821
make prepare-kaggle-notebooks EXP=exp264_exp263_candidate_confidence_dual_selector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp264-exp263-candidate-confidence-dual-selector-train --title 'exp264 exp263 candidate confidence dual selector train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp264_exp263_candidate_confidence_dual_selector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp264-exp263-confidence-dual-selector-train --title 'exp264 exp263 confidence dual selector train' --run-on-push --strict"
make push-kaggle-train EXP=exp264_exp263_candidate_confidence_dual_selector
kaggle kernels pull kentookumura/exp264-exp263-confidence-dual-selector-train -p /tmp/kaggle-pull/exp264-exp263-confidence-dual-selector-train -m
kaggle kernels logs kentookumura/exp264-exp263-confidence-dual-selector-train -f --interval 10
kaggle kernels output kentookumura/exp264-exp263-confidence-dual-selector-train -p experiments/exp264_exp263_candidate_confidence_dual_selector/kaggle/output/stage_a_v1
```

### Stage A完了時点の未実行・予定（履歴）

```bash
task prepare-kaggle-notebooks EXP=exp264_exp263_candidate_confidence_dual_selector EXTRA_ARGS="--strict"
task push-kaggle-train EXP=exp264_exp263_candidate_confidence_dual_selector  # 当時はStage B承認後のみ
```

この時点ではKaggle Stage A train v1だけが完了し、Stage B train、inference、submissionは未実行だった。

`task` executableが環境に無かったため、同等の`make prepare-kaggle-notebooks`を使用した。初回prepareはbootstrap fileをexperiment-relative pathとして誤指定して失敗したため、repo-relative sourceと安全なdestinationを持つ`bootstrap_dependency_files`へ修正した。再prepareはPASSした。

## 変更点

- `candidate_contract.yaml`: 12候補、alias、confidence、legal domain。
- `feature_contract.yaml`: exp251 295列からの整理規則、feature group、重複/相関監査。
- `output_contract.md`: dual score、compact adapter、nested stacking、scope外。
- `config.yaml`: stage、booster数、guard、再現性を機械可読に固定。
- `src/candidate_selector_pipeline.py`: exp263 12候補formula、raw context、candidate-long、監査、dual LightGBM、streaming OOF、compact adapter、current-test adapterを実装。
- train notebook: Stage Aを同runで必須実行し、Stage Bだけ`run_approved=true`を要求する。
- inference notebook: 5-fold saved selectorを平均してcompact化する。native confidence採用時はexp263 Stage 1 parityを必須にする。
- unit test: 4件PASS。12 one-hot、label隔離、exact duplicate、2 legal domain、74 compact featureを確認。
- COPCF: exp263 cache単体ではtrain cross-fit/current-test同一生成を保証できないため初回schemaではdefer。raw horizontal/typewell contextを共通生成する。
- 親exp251 train notebook 259行 / 7章に対し、exp264 train notebookは261行 / 7章。input contract、audit、training、metrics、SHAの同等role slotを維持し、薄い`main()`呼び出しにはしていない。
- train/inference Jupytext sourceに`__file__`は残っていない。
- Stage A train package metadataはcanonical id/title、private、CPU、internet off、competition source、exp263 Stage 0 kernel source 1件。bootstrap内configは`feature_contract_audit`、`run_approved=false`で、exp251 v4 schemaと`src/candidate_selector_pipeline.py`のSHAをmanifestへ含む。
- inference packageもcanonical id/title、private、CPU、internet offで準備した。`run_on_push=false`、`inference_enabled=false`のまま、入力sourceをexp263 inferenceと将来のexp264 trainへ固定したため、誤実行しない。

## 2026-07-18 hidden-safe inference再開

- ユーザーから推論へ進む明示指示を受領した。Stage D worst-well guard FAIL（+17.446742）は変更せず、例外scopeを保存済みmodelによる未提出推論成果物生成だけに固定した。
- 再学習は0。Stage C version 3の40 selector modelとStage D version 2の`selector_compact_addonly` 15 TVT modelだけを使う。matched control modelは推論しない。
- 既存exp264 inferenceはStage B 10 modelと静的exp263 public-test Parquetを使う旧scaffoldだったため、Stage C outer別8 selector model→74 compact列→対応Stage D 3 modelの構成へ更新する。
- exp263候補は静的row artifactを入力にせず、raw competition testから6 primitive、5 pair、fixed 1、21 confidence列を同じnotebookで再生成する。
- Stage C model取得はKaggle CLI既定page sizeのため当初3/40ファイルだけだった。version 3を`page-size=200`で再取得し、40/40 SHA一致を確認した。40 modelは39.07 MB、完全性確認済みの決定的tar.xz bundleは13.03 MB（44 entries）、SHA `1697e1f762540673ce19656134c84da0d6528fc1aebfb328ba8dc3572f246b21`。
- 最終`submission.csv`は形式互換の推論成果物として生成するが、notebookもCodexもKaggle competition submitを行わない。

## 2026-07-18 inference push前監査

- 実行する新規学習variant 0、LightGBM config 0、training fold 0、合計学習booster 0。保存済みStage C selector 40本と保存済みStage D add-only TVT model 15本をpredictionへ使うだけで、parent/control再学習は0。
- 推論計算はouter 5 × inner 4 × 2 objectivesの40 selector predictionと、outer 5 × 3 configsの15 TVT prediction。matched control 15 modelはmanifest確認対象だがprediction対象外。
- 初回はStage C archiveをnotebook supportへ埋め込み、21,750,498 bytesのpackageをpushしたが、Kaggle SaveKernel APIが400で拒否した。実行versionは作られておらず、推論も開始していない。
- サイズ制約を回避するためprivate Dataset `kentookumura/exp264-stage-c-selector-models`を作成した。dataset fileは13,029,920 bytes、SHA `1697e1f7...6b21`で、CLI上のfiles一覧も確認した。
- 再生成したcanonical inference packageはprivate、CPU、internet off、run-on-push true、competition source 1、kernel source 9、private dataset source 1。GPU/machine shape指定は無効化した。
- package notebookは約416 KB、SHA `ef83297d...4e6a7`。package config SHA `03f304c0...64f77`、metadata SHA `0ab5942e...fcfab`。
- 小型packageの初回pushは`Notebook not found`でversion作成前に拒否された。9 kernel + 1 dataset + 1 competitionの計11 sourceが上限を超えた可能性を切り分け、1.3KBのexp112 feature schemaをbootstrapへ移してexp112 kernel sourceを削除した。推論値・model・feature schemaは変更していない。
- sourceを10件へしても旧slugは`Notebook not found`を返したため、最初の400でslugがghost状態になったと判断した。実験内容を変えず、新slug `kentookumura/exp264-stage-d-hidden-safe-inference`へ切り替えた。

## 2026-07-18 inference Kaggle version 1開始

- `kentookumura/exp264-stage-d-hidden-safe-inference` version 1のpush成功を確認した。直後のCLI statusは`KernelWorkerStatus.RUNNING`。
- 実行packageはprivate、CPU、internet off、run-on-push true。8 kernel source + 1 private Dataset + 1 competition sourceで合計10 input source。
- push済みpackage SHAはnotebook `d3ef7364...06bf8`、config `bab22c4b...cb07c`、metadata `27012472...6906`。
- 実行scopeは保存済み40 selector + 15 add-only TVT prediction、学習booster 0。`submission.csv`は生成するがcompetition submitはしない。

## 2026-07-18 inference version 1 preflight ERROR

- 起動ログを一度確認し、27.0秒でStage D model manifest SHAのfail-closed guardが停止したことを確認した。候補生成、selector prediction、TVT predictionは未開始で、学習boosterは0。
- 原因はsummaryの短縮SHAからconfigへ転記した完全値の誤り。configは`064684b66646...`だったが、取得済みmanifestとStage D reproducibility manifestが一致して記録する正しいSHAは`064684b65cde075d1be8d0200b2d6da54939891013370b2b9e48170c566d6159`。
- model、feature、推論方式、input sourceは変更せず、期待SHAの誤記だけを修正してversion 2へ再pushする。

## 2026-07-18 inference Kaggle version 2開始

- 正しいStage D manifest SHAを入れた同じcanonical notebookを、同じkernelへversion 2としてpushした。push成功と`KernelWorkerStatus.RUNNING`を確認した。
- package SHAはnotebook `32ab0876...60803`、config `0d03f86b...3bb0f`、metadata `27012472...6906`。
- 起動直後のlogs APIは空だったため、version 1ログとの混同を避けて再pushせず、Kaggle runを継続させる。長時間follow監視は行わない。
- Stage C archiveは44 entries、SHA `1697e1f7...6b21`でconfig/support manifestと一致。targeted 10 tests、Jupytext test、py_compile、F821/F401/E9、strict experiment/project validationをPASSした。
- 出力はprediction、fold別compact、監査sample、`submission.csv`、metrics/SHA manifest。Kaggle competition submitは行わない。

## Stage A完了時点の再現性メモ（履歴）

- seed policy: LightGBM 42、samplingはstable key SHA256。
- stochastic components: LightGBM row/column sampling、candidate-long row sampling。
- CPU/GPU runtime: Stage AはKaggle CPU、153.612秒。Stage B/C CPU、Stage D GPU。
- Kaggle kernel id/version: `kentookumura/exp264-exp263-confidence-dual-selector-train` version 1、id_no `127485868`。
- input SHA: exp263 Stage 0 manifest `85e60ac1...a26bb9e`、catalog `7cd74866...e9e6e0`。
- feature schema/content SHA: schema `766cfcf1...d4deb`、audit-long `62d84bcf...a4896d`。
- model manifest/model SHA: 未生成。
- prediction SHA: 未生成。
- submission SHA: scope外、未生成。
- rerun check: 未実行。

## Stage A完了時点の次アクション（完了済み）

1. 承認済みStage B packageで`execution.run_approved`、package metadata、0 control再学習を再確認する。
2. 同じcanonical kernelへpushし、完了後にselector score、feature importance、重複・相関、SHAを監査する。

## 2026-07-16 current-test confidence parity実装

- exp263 Stage 1に21 namespaced confidence列を追加し、exp264 candidate contractにもprimitive別required
  field mappingを同じ順序で固定した。
- inferenceはcandidate-long feature生成前に21列の存在、numeric finite、`confidence_valid`の型、
  native fieldを持つ5候補の全行validを確認する。likPFだけはnative scalarなしのためvalid falseを許可する。
- selected featureの暗黙NaN補完に到達する前にfail-closedにしたため、1列だけ欠けるケースもテストで停止する。
- Stage Aで警告が出たconfidence/formula列の逐次insertを、列dict + 1回の`pd.concat`へ変更した。
  特徴列の順序と値は維持し、`PerformanceWarning`が出ないことを回帰テストへ追加した。
- exp263/264 targeted 19 tests、repo全69 tests、ruff、py_compile、Jupytext inference regenerationをPASSした。
- Stage Bの学習設定は変更していない。1 variant × 2 objectives × 5 folds = 10 CPU boosters、
  parent/control再学習0のまま未承認である。

## 2026-07-17 exp263 current-test confidence parity完了

- exp263 Stage 1 Kaggle inference v3が14,151 rows × 36列で完走し、必要な21 namespaced confidence列を
  全行non-null/finiteで確認した。likPFの`confidence_valid=False`はnative scalarなしの契約どおり。
- v2の旧15値列はexact equality、最大絶対差0。拡張Parquet SHAは
  `bda0502894d6a20cc3c332d729cf120b17ceed2e1773093bd7140c6df71e360c`。
- exp264 current-test adapterの前提は満たされた。Stage Bは未承認のため、学習・inference・submissionは
  引き続き実行しない。

## 2026-07-17 Stage B実行承認

- ユーザーへStage Bの実行量を再提示し、Kaggle実行の明示承認を受領した。
- 実行対象はstandard LightGBM 1 variant、`pred_abs_error` / `p_within10`の2 objectives、
  5 outer folds、合計10 CPU boosters。親実験・baseline・controlの再学習は0。
- `execution.stage=selector_outer_oof`、`execution.run_approved=true`へ変更した。
- Stage Cの40 CPU selector boosters、Stage Dの30 GPU boosters、inference、submissionは無効のままで、
  今回の実行scopeに含めない。
- push前にJupytext、構文、ruff、targeted tests、experiment validation、package内config SHA、
  canonical kernel id/title、private/CPU/internet off、exp263 Stage 0 inputを再確認する。

## 2026-07-17 Stage B push前監査

- targeted 19 tests、repository全69 tests、Jupytext train/inference test、py_compile、ruff、
  template validation、experiment validationをPASSした。ローカルnotebook実行は行っていない。
- canonical packageを`kentookumura/exp264-exp263-confidence-dual-selector-train` /
  `exp264 exp263 confidence dual selector train`として再生成した。
- metadataはprivate、CPU、GPU/TPU/internet off、`run_on_push=true`。inputはcompetition sourceと
  `kentookumura/exp263-last-anchor-pair-cache-train`の1 kernel sourceだけである。
- source/packageのSHAは一致した。config `e8d63f53...6a5fc`、candidate contract
  `4f4d3f77...86c20`、feature contract `ca58014a...4f16`、selector pipeline
  `4abf3e88...2372`。package notebook SHAは`dd9c11c6...6119`。
- package内configは1 variant、2 objectives、5 folds、10 CPU boosters、control再学習0、
  Stage C/D disabled、inference disabledであることを確認した。

## 2026-07-17 Stage B Kaggle version 2実行開始

- 同じcanonical kernel `kentookumura/exp264-exp263-confidence-dual-selector-train`へpushし、
  `Kernel version 2 successfully pushed`を確認した。既存kernel id_noは`127485868`。
- push対象packageのconfig SHAは`e8d63f53...6a5fc`、notebook SHAは`dd9c11c6...6119`。
- 実行scopeは承認どおり10 CPU boosters、control再学習0。Stage C/D、inference、submissionは実行しない。
- 完了まで同じslugを監視し、空ログや一時的status API errorだけを理由に再pushしない。
- `RUNNING`を確認後、少なくともouter fold 0/1のbinary/L1計4 boosterが学習ログを出し、
  各objectiveで200 roundまで正常に進んだ。tracebackやOOMは観測していない。
- ユーザー指示によりCLIのfollow監視だけを停止した。Kaggle version 2は停止せず継続する。
  完了連絡後に同じslugのlogsと必要成果物を取得して最終監査する。

## 2026-07-17 Stage B Kaggle version 2完了監査

- ユーザーの完了連絡後、CLIで`KernelWorkerStatus.COMPLETE`を確認した。10 CPU boostersを完走し、
  生成物一覧表示は1,633.209秒、notebook変換完了は1,644.601秒。traceback/OOMなし。
- best iterationは`p_within10=[127,149,194,149,198]`、
  `pred_abs_error=[137,208,143,123,210]`。各modelはtrain 720,000 / early-stop 360,000 long rows。
- score guardはPASS。pooled expected-error MAEは5.788783→3.742231、within10 loglossは
  0.510131→0.355298、Brierは0.165095→0.110596で、3指標すべて5/5 folds改善した。
- hard top1は8.362844で、fixed `exp226_w500_50_50` 8.238332より+0.124512悪化。
  3/5 foldsでは改善したが、near +0.088746、1000+ +0.135728、worst-well +18.258274で
  hard readout guardはFAIL。
- 完了後に独立assignmentを`selector_by_well.csv`へpost-hoc joinした。assignmentは学習・calibration・
  threshold選択に未使用。spatial valid 200 wellsは9.186219 vs fixed 8.748108（+0.438111）、
  typewell-purged valid 200 wellsは9.101735 vs 8.694132（+0.407604）で、hidden-likeもFAIL。
- 重要度group shareはpred-abs-errorでbank 53.87%、ctx 28.01%、formula 8.83%、cand 4.91%、
  conf 4.03%、id 0.34%。within10はbank 54.13%、ctx 36.35%、cand 4.25%、formula 3.43%、
  conf 1.46%、id 0.38%。`conf__native__sigma_tvt`は予測誤差重要度5位。
- Stage A監査は再現し、162候補→100採用、全欠損41、定数5、完全重複16、高相関35組、
  feature schema SHA `766cfcf1...d4deb`でv1と一致した。
- 一括output取得は411MB OOFの`IncompleteRead`で失敗したため、diagnostic CSV/JSON/PNGと10 modelを
  pattern指定・個別取得した。途中1件の一時DNS失敗は同じfileを再取得して解消した。
- model manifest SHA `12375038...4c9a`と10 model SHAはすべて一致。Kaggle側で記録したcandidate score
  OOF SHAは`e51bb674...45a5a`、compact meta OOF SHAは`1ab4cff4...45ba`。巨大OOF本体のlocal取得は
  未完了で、0-byte partial fileを信頼根拠には使わない。
- 全100特徴の説明・objective別重要度、top1選択率、0 gain、完全重複、全欠損、定数、高相関35組を
  `selector_feature_readout.md`へ記録した。
- 運用判断は「score guard PASS / hard readout FAIL」。hard selector、Viterbi、softmax平均、inference、
  submissionは行わない。scoreはStage C nested compact add-only候補にだけ残す。Stage C/Dは別承認。

## 2026-07-17 Stage B記録同期・検証

- `config.yaml`、`metrics.json`、`result.md`、`README.md`、steering tasklistをStage B結果へ同期した。
- `backlog/KAGGLE_DIRECTION.md`はhard selector branchを閉じ、Stage C compact add-onlyを別承認候補として更新した。
  sequence decoder案はscore guard通過済みだが、保存scoreによる0-booster switch attribution待ちへ下げた。
- `make update-summary`で262実験の`experiment_summary.md`を再生成し、exp264 statusを
  `stage_b_score_guard_passed_hard_readout_failed`へ反映した。
- JSON/YAML parse、`make validate-exp EXP=exp264_exp263_candidate_confidence_dual_selector`のstrict validation、
  `experiments/exp264_exp263_candidate_confidence_dual_selector/tests/test_exp264_candidate_selector_pipeline.py`の5 testsをすべてPASSした。
- Stage C 40 CPU boosters、Stage D 30 GPU boosters、inference、submissionは引き続き未承認・無効。

## 2026-07-17 Stage C実行承認・実装

- Stage B結果とStage Cの内容を説明後、ユーザーから`Stage Cに進んでください`との明示承認を受領した。
- 実行scopeは1 selector variant、2 objectives、5 outer × 4 inner、合計40 CPU boosters。
  親実験・baseline・control再学習0、GPU 0、Stage D/inference/submissionなし。
- `run_stage_c`を追加し、各outer-trainのwellを行数balancedなinner 4-foldへdeterministic割当する。
  outer-trainはinner OOF score、outer-validはouter-train内4 selector ensembleからcompact化する。
- 出力はdownstream outer fold × role × source foldの25 Parquet partition。期待base rowsは
  18,919,945、outer-valid score auditは45,407,868 candidate-long rows、modelは40本。
- `nested_fold_manifest.csv`、`nested_selector_model_manifest.json`、
  `nested_compact_partition_manifest.csv`、`nested_compact_manifest.json`、
  `nested_outer_valid_candidate_score.parquet`、`nested_selector_metrics.json/csv`、nested重要度を保存する。
- outer-valid wellをinner assignment、fit、early stoppingから除外し、inner train/valid well disjoint、
  model/partition/row coverage、全score finiteをfail-closedで検証する。
- `config.yaml`を`nested_compact_meta`、`run_approved=true`、Stage C enabledへ変更した。
  Stage D/inference/submissionはdisabledのまま。
- pipeline/train sourceのpy_compile、F821、exp264契約テスト6件をPASSした。ローカルnotebook実行はしていない。
- Jupytext train sourceは325行 / 7章で、方法親exp251 trainの259行 / 7章に対し、
  input contract、audit、training、metrics、SHAの同じrole slotを維持しつつStage C orchestrationを展開した。
  canonical train ipynbをJupytextから再生成し、薄い`main()`呼び出しにはしていない。

## 2026-07-17 Stage C push前監査

- JSON/YAML parse、strict experiment validation、Jupytext round-trip、py_compile、F821、repo全78 testsをPASSした。
- canonical train packageを既存kernel `kentookumura/exp264-exp263-confidence-dual-selector-train` /
  title `exp264 exp263 confidence dual selector train`、private、CPU、internet off、run-on-pushで再生成した。
- 実行契約は1 variant、2 objectives、5 outer × 4 inner、40 CPU boosters、親/control再学習0。
  Stage D、GPU、inference、submissionはdisabled。
- package sourceはcompetitionと`kentookumura/exp263-last-anchor-pair-cache-train`だけ。
- source/package SHAは一致した。config `c0c9cc5a...e6ce2`、candidate contract
  `4f4d3f77...86c20`、feature contract `ca58014a...4f16`、selector pipeline
  `56695bbc...76be2`。bootstrap ZIP内の同4 filesもsource SHAと一致した。
- canonical source notebook SHAは`81ce4445...8206`、bootstrap付きpackage notebook SHAは
  `3bdf9468...b9b`。train source/packageに`__file__`は残っていない。
- credential checkerはOAuth/legacy CLI credentialを確認した。API tokenは未設定だが、既存OAuthでCLI実行可能。
- Stage B compact 581.9MBの5 downstream-fold面に相当するため、Stage C compactは概算約2.9GBに
  outer-valid score約0.4GBと40 modelsが加わる。Kaggle outputは必要なmanifest/metricsを優先監査する。

## 2026-07-17 Stage C Kaggle version 3実行開始

- push前に既存canonical kernelをpullし、id `kentookumura/exp264-exp263-confidence-dual-selector-train`、
  id_no `127485868`、title、CPU/internet off、exp263 kernel sourceが一致することを確認した。
- 同じcanonical kernelへpushし、`Kernel version 3 successfully pushed`を確認した。
- push後pullしたnotebookは`run_stage_c`、`nested_compact_meta`、40 booster contractを含み、
  id_no `127485868`のまま。Kaggle statusは`KernelWorkerStatus.RUNNING`。
- 実行scopeは承認どおり40 CPU boosters、親/control再学習0。Stage D、inference、submissionは実行しない。
- push後は重複実行防止のためlocal sourceをstatus `stage_c_kaggle_v3_running`、
  `run_approved=false`へ戻した。実行中version 3のbootstrap config SHAはpush前記録を正とする。
- 空logや一時status errorを理由に再pushしない。完了後にlogsとmanifest/metricsを優先取得し、
  40 models、25 partitions、18,919,945 compact rows、45,407,868 outer-valid long rows、SHA、
  leakage/score guardを監査する。

## 2026-07-17 Stage C Kaggle version 3完了監査

- ユーザーの完了連絡後、CLIで`KernelWorkerStatus.COMPLETE`を確認した。生成物一覧表示は
  4,329.795秒、notebook変換完了は4,338.238秒。traceback、exception、OOMは観測していない。
- 1 variant × 2 objectives × 5 outer × 4 innerの40 CPU boostersを完走し、親/control再学習は0。
  downstream TVT学習、GPU、inference、Viterbi、submissionは実行していない。
- pooled expected-error MAEは5.788783→3.762776、within10 loglossは0.510131→0.354702、
  Brierは0.165095→0.110137で、3指標すべて5/5 folds改善しscore guardはPASSした。
- hard top1は8.420613でfixed `exp226_w500_50_50` 8.238332より+0.182281悪化。3/5 foldでは
  改善したがoverall guardはFAILで、hard inference不採用を維持する。
- leakage auditはPASS。outer-valid wellはinner assignmentから除外、inner train/validはwell-disjoint、
  outer-train compactはinner OOF、outer-valid compactは4 inner model ensembleで生成された。
- `nested_fold_manifest.csv`はouter 5 × inner 4の20組、model manifestは40組と40一意SHA、
  compact manifestは5 valid + 20 trainの25 partition、18,919,945 rowsを記録した。outer-valid scoreは
  45,407,868 candidate-long rowsで期待値と一致した。
- model manifest SHA `b2d8def7...aab1`、compact manifest SHA `c95d9ea4...c06e`はlocal取得ファイルと
  metrics/reproducibility記録で一致した。outer-valid score SHAは`4a77ceb7...2777`。
- 巨大Parquet本体は運用ルールどおり取得しなかった。Kaggleログで全40 modelの非ゼロsizeを確認し、
  manifest全40 path/SHAの一意性を検証、選択取得した3 modelはbyte-level SHA一致。途中の切断でできた
  0-byte partial modelは削除し、監査根拠に含めていない。
- nested重要度group shareはpred-abs-errorでbank 51.49%、ctx 30.66%、formula 8.33%、cand 5.18%、
  conf 4.04%、id 0.31%。within10はbank 54.57%、ctx 36.15%、cand 4.25%、formula 3.29%、
  conf 1.42%、id 0.33%。`conf__native__sigma_tvt`はpred-abs-error 6位、2.64%。
- 判定は「Stage C score/leakage PASS、hard FAIL」。74列nested compactはStage D add-only入力として成立。
  Stage Dは30 GPU boostersと15 control再学習を含むため、別承認までdisabledを維持する。

## 2026-07-17 Stage C記録同期・検証

- `config.yaml`、`metrics.json`、`README.md`、`result.md`、`output_contract.md`、
  `selector_feature_readout.md`、steering 3文書、調査docs、`backlog/KAGGLE_DIRECTION.md`を実測値へ同期した。
- `make update-summary`で263実験の`experiment_summary.md`を再生成し、exp264 statusを
  `stage_c_completed_score_guard_passed_nested_compact_ready`へ反映した。
- config YAML / metrics JSON parse、strict experiment validation、exp264契約テスト6件、
  repository全84 testsをPASSした。
- Stage D、inference、submissionは有効化していない。次の外部実行は別承認後にのみ行う。

## 2026-07-17 Stage D 30 GPU boosters実行承認

- ユーザーからStage D full matched ablationの明示承認を受領した。
- 実行scopeは`matched_control`と`selector_compact_addonly`の2 variants、exp218固定
  `lgb0/lgb1/lgb2` 3 configs、5 outer folds、合計`2 × 3 × 5 = 30` GPU boosters。
- 30本の内訳はmatched control 15本、74列compact add-only 15本。Stage C selector 40本、
  parent PF/HMM/Beam、exp218 historical modelは再学習しない。downstream base-only controlだけを15本再学習する。
- control再学習が必要な理由は、historical exp218 OOFとStage C downstream outer foldが一致せず、
  同一fold・同一config・同一runtimeで74列の有無だけを比較しなければfeature効果を帰属できないため。
- targetはexp218と同じ`TVT - last_known_tvt` residual。controlはexp218 380列、add-onlyは380+74=454列。
- GPUはT4、internet off、deterministic/gpu double precision/thread 8、early stopping 250を固定する。
- Stage D実装・package監査後に別kernelからStage C version 3をinputとして実行する。
  inference、Viterbi、submissionは今回のscope外。

## 2026-07-17 Stage D push前監査・Kaggle version 1開始

- exp264対象契約テスト9件、strict experiment validation、Jupytext、py_compile、F821をPASSした。
  repository全体は87件中86件PASSで、残る1件は今回変更していないexp267の
  `execution.run_approved=true`と既存テストの期待値`false`の不一致である。
- 新しいStage D kernelを`kentookumura/exp264-exp263-confidence-dual-selector-tvt-train` / title
  `exp264 exp263 confidence dual selector tvt train`として作成した。既存Stage C kernelは上書きしない。
- package metadataはprivate、T4、internet off、run-on-push。inputはcompetition、Stage C version 3系、
  exp072 feature cache、exp145 learned likelihoodの3 kernel source。
- push時のconfig SHAは`519d1a138e1f267244b34d714b808a8d58b8cf8164ac411f87bce4a93a68cfa6`、
  selector pipeline SHAは`71b8da4a384b5e0be97b9010b27b4f04d5d33ceeaff6b5c664223815c48ce44a`、
  package notebook SHAは`3a35b551107933552b1c832abe002683929299872bc85ff189c25a0af1e48e66`。
- bootstrap内のexp218 source/config SHAは`6712d5b4...f7a33` / `fd547494...fa09d`、
  hidden-like assignment SHAは`5f9ac9fa...a6597`でsourceと一致した。
- target kernelのpush前pullは403で未作成を確認し、version 1のpush成功後にid_no `127577193`、
  T4、private、internet off、入力3件をpull metadataで再確認した。
- push後notebookには`downstream_tvt_ablation`、`run_stage_d`、30-booster contract、bootstrap manifestが
  残り、`__file__`依存はない。Kaggle statusは`KernelWorkerStatus.RUNNING`。
- 実行scopeは承認どおりcontrol 15 + add-only 15 = 30 GPU boostersだけ。inference、Viterbi、
  submissionは実行しない。重複push防止のためlocal `execution.run_approved=false`へ戻した。

## 2026-07-17 Stage D version 1即時失敗・version 2修正実行

- ユーザーのGUI確認を受けてstatus/logsを再取得し、version 1が`KernelWorkerStatus.ERROR`であることを
  確認した。初回push直後の一時的な`RUNNING`を確定扱いした記録を訂正する。
- version 1は22.525秒、`run_stage_d`のStage C入力検証で停止し、LightGBM fitへ到達していない。
  したがって学習済みGPU boosterは0本、control/add-only OOFも未生成。
- 原因は`compact_meta_schema.json`のファイルbyte SHAと、JSON内
  `compact_meta_schema_sha256`の論理schema SHAを同一値として扱ったこと。実測byte SHAは
  `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74`、論理SHAは
  `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
- configを`stage_c_expected_compact_meta_schema_file_sha256`と
  `stage_c_expected_compact_meta_schema_logical_sha256`へ分離し、pipelineもbyte/logicalを別々に検証する
  よう修正した。unit testに両SHAの正常系とlogical mismatch fail-closedを追加した。
- exp264対象テスト9件、py_compile、F821/E9、strict experiment validation、package bootstrapの
  config/pipeline SHA一致を再確認した。
- 同じkernel IDへversion 2を`--accelerator NvidiaTeslaT4`明示でpushした。v1失敗時刻を超えた
  30秒後もstatusは`KernelWorkerStatus.RUNNING`。実行中CLI logsは空で、即時schemaエラーは再発していない。
- version 2の実行scopeは同じ30 GPU boostersで追加variantはない。重複push防止のためlocal
  `execution.run_approved=false`へ戻した。

## 2026-07-18 Stage D version 2完了・成果物監査

- ユーザーの完了連絡後にKaggle statusを一度確認し、version 2が`KernelWorkerStatus.COMPLETE`、
  ログが`completed_boosters=30 / planned_boosters=30`まで到達したことを確認した。
- 30本目完了は32,344.668秒、生成物一覧は32,352.136秒、notebook変換完了は32,361.299秒。
  Traceback、OOM、DeadKernelはなく、DataFrame fragmentationのPerformanceWarningだけだった。
- matched control `lgb_mean`は8.545568072、selector compact add-onlyは7.805644167、差は
  -0.739923904。fold差は-1.272861/-0.670234/-0.358390/-0.083249/-1.241044で5/5改善。
- near 0–250、mid 250–1000、1000+の差は-0.222916/-0.419414/-0.807155。
  hidden-like spatial/typewell-purgedは-1.174830/-1.193025で、ここまでの事前guardは全てPASS。
- well単位は470改善、303悪化。243 wellが+0.25を超え、worst `70925e23`はcontrol
  5.804539からadd-only 23.251280へ悪化し、差+17.446742。事前上限+0.25を超えたため総合guardはFAIL。
- OOF Parquet 3,783,989 rowsを取得し、5 fold coverage、全予測/targetの欠損・nonfinite 0、8 pooled RMSEを
  streaming再計算してmetricsと一致した。OOF SHAは`7367983f...dafee`。
- model manifestは2 variants × 3 configs × 5 foldsの30組、30 pathとも一意。取得した30 modelのbyte SHAは
  全件manifestと一致した。manifest SHAは`064684b6...6159`。8 output SHAもreproducibility manifestと一致。
- 74 compact列は15 add-only modelの平均正規化gainで全gainの70.96%。上位4列は2 legal domainの
  `p_within10`/`pred_abs_error` top1候補値とlast-known anchorとの差で、合計59.96%。全74列の説明と重要度を
  `stage_d_feature_importance_readout.md`へ記録した。
- global改善を理由にworst-well guardを事後変更しない。compact inference、hard inference、Viterbi、
  softmax TVT平均、submissionは不採用。追加GPU再学習やcontrol再実行もしない。

## 2026-07-18 hidden-safe inference version 2失敗・欠損契約修正

- CLIで`kentookumura/exp264-stage-d-hidden-safe-inference` version 2が
  `KernelWorkerStatus.ERROR`であることを確認した。395.586秒、candidate生成後・selector model predict前に
  `current-test selector matrix contains non-finite values`で停止した。新規学習は0 booster、competition submitも0。
- Stage A catalogを実値監査し、採用100列中29列が学習時からNaNを含むことを確認した。内訳は`conf` 11、
  `formula` 11、raw context 7。feature contractもsource-native unavailableをNaNで保持する設計であり、
  version 2の一律`np.isfinite` guardが学習・推論契約と矛盾していた。
- NaNを0/median補完せず、Stage A `feature_catalog.csv`をSHA
  `83c8b953410c19b3170e180a4e1f28deb7fc8d898504c6c00b494f17a1e5639d`でpackageへ追加した。
- 新guardは`±inf`、training missing率0の列への新規NaN、`conf__`/`formula__`構造欠損率ずれ、
  current-test全欠損化を停止し、feature別・candidate別missingnessを保存する。Stage D base 380列は学習時も
  finite必須だったことをコードで確認し、compact 74列・最終454列のfinite guardは変更していない。
- exp264対象回帰テストは10件から11件へ増やし、期待NaNの通過、dense新規NaN、構造欠損率ずれ、`±inf`の
  fail-closedを確認した。Kaggle version 3のpushはnotebook/package同期とstrict validation後に行う。
- canonical Jupytext inferenceを`.ipynb`へ同期し、targeted 12 tests、repository 146 tests、py_compile、
  F821/F401/E9、Jupytext test、strict experiment/project validationをPASSした。ローカルnotebook実行はしていない。
- 同じkernel ID/title、private、CPU、internet off、run-on-push true、既存8 kernel + 1 dataset + 1 competition
  sourceでversion 3 packageを再生成した。Stage A catalog 48,271 bytes / SHA `83c8b953...639d`がbootstrap
  manifestに含まれた。共有pipelineの別作業更新を巻き戻さず再同期し、package内pipeline SHAもsource
  `6034bb389eb28f6c761db5f7310c28a4b08cc8f431750b41839b780ba67fb784`と一致した。
- 修正版package SHAはnotebook `9a01bea901aa4d62854186d3af6d3c4f57ef1ab3ec2595647d8681a388886d63`、
  config `b4495b05b749eaef09fa6b0e68f50b71e260a7da7d3a6594474461b3b9427321`、metadata
  `270124722f4ff43001cc8ccb3c914d7cfb43529e10770a7a169084269de76906`。

## 2026-07-18 hidden-safe inference version 3 push

- ユーザーの実行指示を受け、既存kernelを事前pullしてID `kentookumura/exp264-stage-d-hidden-safe-inference`、
  private、CPU、internet off、入力1 dataset + 8 kernels + competitionを再確認した。
- 正規sourceとpackageのpipeline/inference/config一致、package SHA、strict experiment validationを再確認し、
  同じkernel IDへversion 3をpushした。直後statusは`KernelWorkerStatus.RUNNING`。
- 実行scopeは保存済みStage C selector 40本とStage D add-only TVT model 15本のpredictionだけ。
  新規variant 0、LightGBM config 0、training fold 0、学習booster 0、親/control再学習0。
- `submission.csv`は成果物として生成するがcompetition submitは0。Stage D worst-well guard FAILも保持する。
- ユーザー方針どおり継続監視は行わない。完了連絡後にlogs/outputとmissingness、model SHA、454列finite、
  row/order、prediction/submission SHAを監査する。

## 2026-07-18 hidden-safe inference version 3失敗・formation契約監査

- ユーザーの失敗連絡後にCLI status/logsを取得し、version 3が`KernelWorkerStatus.ERROR`、378.938秒で
  selector predict前に停止したことを確認した。候補生成は完了、新規学習boosterとcompetition submitは0。
- tracebackはtraining missing率0の`ctx__raw__astnu/astnl/egfdu/buda`と各delta、計8特徴へcurrent-testで
  NaNが発生したことを報告した。公開test 3 wellのhorizontal CSVは全て6 formation列を持たない。
- config allowlistとStage A catalogを全件監査し、`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`のraw値と
  last-known差の12特徴がselectedであると確認した。ANCC/EGFDL系4特徴はtrainにも少数NaNがあるため最初の
  dense guardには出なかったが、current-testでは同様に全欠損となる。
- repoの公式データ記録は6 formation列をtraining onlyと明記しており、Stage Aの
  `generated_from_raw_or_exp263_stage1`分類が誤りだった。12特徴のselector gain share合計は
  pred-abs-error 5.657%、within10 7.622%。欠損許可、0/median補完、test-only KNN補完は学習/推論parityを
  回復しないため実施しない。
- 既存Stage C/D OOFは後段の訂正により診断値としても無効化した。正規修正候補は
  (A) direct formation 12特徴を削除してStage A/B/Cとadd-only 15本だけ再学習、または
  (B) outer-fold cross-fit formation imputerをtrain/test共通で導入して同じ範囲を再学習すること。
  複数の妥当案で結果とGPUコストが変わるため、ユーザー選択前に実装・pushしない。

## 2026-07-18 OOF trust判定の訂正

- 「既存Stage C/D OOFは診断値として保持できる」という先の記録を撤回する。outer-validにはhidden testで
  利用できないformation raw/delta 12特徴そのものが入っており、これはfeature availability leakageである。
- Stage A feature contract PASS、Stage B/C score guard PASS、Stage C leakage PASS、Stage D add-only
  7.805644、controlとの差-0.739924、fold/bucket/hidden-like/by-well、重要度をすべて無効化する。
  数値は失敗の再現記録としてのみ残し、モデル比較・候補選択・backlogの根拠に使わない。
- Stage D matched control 8.545568はcompact 74列を使わない別surfaceだが、exp264 add-onlyのgain根拠には
  ならない。再利用する場合はexp218側のraw-test feature contractを独立に再監査してから扱う。
- exp263の候補値・confidence生成そのものは今回のavailability leakageとは別契約であり、自動的には
  無効化しない。修正版selectorの入力候補として使う前に改めてparityを確認する。

## 2026-07-18 修正版feature availability監査・Stage A実装

- `studies/exp264_feature_availability_audit.py`で旧selector 100列とexp218 downstream 380列を
  feature-levelに追跡した。actual raw schema、生成コード、保存manifest/metrics/SHAを根拠にする。
- selectorは88 pass / 12 fail。`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA` raw/deltaを削除し、
  `features.raw_context.horizontal_numeric_allowlist=[MD,X,Y,Z,GR]`へ変更した。
- `audit_raw_context_availability`を追加し、actual train/current-testの全horizontal file headerでallowlistの
  全file存在をfit前に検証する。欠けた列はNaN化せずfeature生成前に停止する。
- exp218は273 pass / 107 fail。内訳はfull-train formation reference 74、exp111 fold0 scoreの
  non-nested stacking 27、その推移依存GRWR 6。既存380列matched control/OOFは再利用しない。
- audit生成物は`artifacts/feature_availability_audit/`。selector/exp218 feature CSV、clean 273 allowlist、
  summary JSON、readout READMEを保存した。
- targeted testは13件PASS。py_compile、ruff F821/F401/E9、Jupytext convert/test、strict exp validationをPASS。
- 次のKaggle Stage A scopeは0 variant / 0 LightGBM config / 0 fold / 0 booster、CPU、internet off、
  親/control再学習0。Stage AはGPU cost approval対象外だが、Stage B/C/Dは新たに承認を取り直す。

## 2026-07-18 corrected Stage A version 4 push

- `runtime.kaggle.train_kernel_sources`に残っていた旧Stage D用の自己参照・exp072・exp145を削除し、
  corrected Stage Aの実入力である`kentookumura/exp263-last-anchor-pair-cache-train`だけに固定した。
- repository test 155件、strict experiment validation、canonical/packageのconfig・train source byte一致をPASS。
  packageはprivate、CPU、internet off、run-on-push true、competition source 1、kernel source exp263 1件。
- package SHAはconfig `7163aa2bd1d3f7b75e2f1592f5ac311ed59f41de21641748c3d19213a4fa14bc`、
  train source `99ff589901c6b9de96dde18c98596612caee17afa54d3be99c31324506195b46`。
- 既存kernelをmetadata付きで事前pullし、同じID
  `kentookumura/exp264-exp263-confidence-dual-selector-train`へversion 4をpushした。直後statusは
  `KernelWorkerStatus.RUNNING`。実行scopeは0 variant / 0 LightGBM config / 0 fold / 0 booster、
  親/control再学習0で、raw schema availability監査と88列Stage A catalog/cache生成だけを行う。

## 2026-07-18 corrected Stage A version 4完了・生成物監査

- ユーザーの完了連絡後、同じkernel IDのstatusが`KernelWorkerStatus.COMPLETE`であることと、通常logsで
  `Stage A completed: 0 variants / 0 configs / 0 folds / 0 boosters`を確認した。生成物一覧表示は
  152.891秒、notebook HTML書き出しは162.681秒。Traceback、OOM、DeadKernelはなく、nbconvert/mistuneの
  SyntaxWarningだけだった。
- 必要な小規模outputを取得し、`kaggle/output/stage_a_v4/artifacts/`へ保存した。600,000 candidate-long
  rows、変換後150特徴から全欠損41、定数5、完全重複16を除き88特徴を採用した。採用側の完全重複は0、
  高相関14組は事前契約どおりreport-only、学習時missing率>0の採用特徴は25列。
- raw context gateは`MD/X/Y/Z/GR`の5列についてtrain 773/773 file、current-test 3/3 fileで全件PASS。
  旧formation raw/delta 12特徴はschemaに存在しない。candidate IDの`id__candidate__pf_ancc`は候補識別であり、
  training-only formation列`ANCC`とは別物なので保持する。
- logical feature schema SHAは`aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`。
  file SHAはfeature schema `b91ec151...1035`、feature catalog `3a443a1a...1235`、raw availability
  `cead0e03...e1c3`。exp263 manifest/catalog SHAと3,783,989 rows / 773 wells / 5 foldsも期待値と一致した。
- 旧Stage B/C/Dの承認・model・OOF・重要度・matched controlは引き続き無効。次の共通ステップは88列を使う
  Stage B 1 variant × 2 objectives × 5 folds = 10 CPU boosters。Stage B後のdownstreamは、無効107列を
  削除する273列base案か、formation/exp111/GRWRをfold内再生成する案かをユーザー確認後に決める。

## 2026-07-18 clean 273固定・corrected Stage B version 5 push

- ユーザー指示によりdownstreamはclean 273列案に確定。既存380列の「完全復旧」は行わない。
- allowlistは`artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv`、SHA256
  `d01a73cc28485345dd86ed56ad6276f1727dca6b270d87685e1cf578afb677bf`。
- Stage D helperは履歴的source 380列からこの273列をsource順に選択し、source/selected列数、
  allowlist SHA、重複、未知列、列順の不一致をfit前に拒否する。将来の比較面はcontrol 273 / add-only 347列。
- targeted 15 tests、repository 157 tests、Jupytext convert/test、py_compile、ruff F821/F401/E9、
  strict experiment/project validationがPASS。Kaggle notebookのローカル実行はしていない。
- 承認scopeはcorrected Stage Bの1 variant × 2 objectives × 5 folds = 10 CPU boosters、
  control/親再学習0本。corrected Stage C/DとGPU boosterは0本で未承認。
- packageはprivate、CPU、internet off、competition source 1、kernel sourceはexp263 1件だけ。既存kernelの
  metadataとID/title/runtime/input契約が一致した。
- package SHA256: notebook `b9aa7e93...6919`、config `2ebf35ff...9e6a`、metadata
  `b50f33c0...1c9a`、pipeline `9c1b77cb...c0b5`。support manifestのallowlistも`d01a73cc...77bf`で一致。
- canonical kernel `kentookumura/exp264-exp263-confidence-dual-selector-train` version 5をpushし、
  `KernelWorkerStatus.RUNNING`を確認。重複実行防止のためlocal `execution.run_approved=false`へ戻した。
- 完了後には88列schemaのparity、10 selector model SHA、score guard、hard diagnostic、全特徴重要度を監査する。

## 2026-07-18 corrected Stage B version 5完了・生成物監査

- canonical kernel `kentookumura/exp264-exp263-confidence-dual-selector-train` version 5はKaggle CPU、
  internet offで完了した。生成物一覧表示まで1,433.462秒、notebook HTML完了まで1,444.407秒。
  実行scopeは1 variant × 2 objectives × 5 folds = 10 CPU boostersで、control/親再学習は0本。
- 修正版88列のscore guardはPASS。expected-error MAEはprior 5.788783から3.795801、within10
  logloss/Brierは0.510131/0.165095から0.359972/0.112451へ改善し、3指標すべて5/5 foldsで改善した。
- hard top1はFAIL。RMSE 8.587004でfixed `exp226_w500_50_50` 8.238332より+0.348673、改善foldは0/5。
  near +0.079326、1000+ +0.389208、worst-well +14.684481。事後評価専用hidden-likeでもspatial
  +0.768585、typewell-purged +0.721137だった。hard selector、Viterbi、softmax TVT平均は禁止を維持する。
- `candidate_score_oof.parquet` 45,407,868行は12候補が各3,783,989行で、actual/pred errorの
  nonfinite、負のpred error、確率範囲外、binary label違反、model-fold違反はいずれも0。
  `compact_meta_oof.parquet`は3,783,989行、6 key + 74特徴でnull/nonfinite 0だった。
- candidate score OOF SHA `9a91b625...d48a`、compact OOF SHA `5485ede1...512a`、model manifest
  file SHA `d5159ed1...a07d`を確認し、manifest記載の10/10 model byte SHAも一致した。
- confidence groupのgain shareはpred-abs-error 4.267%、within10 1.461%。
  `conf__native__sigma_tvt`はpred-abs-error 4位、2.958%であり、confidence追加は誤差校正には寄与した。
  hard選択の正当化には使わない。
- 全88特徴の説明・objective別重要度、zero-gain、除外した完全重複16組、高相関14組、候補別score、
  top1選択率は`selector_feature_readout_corrected_stage_b_v5.md`を正とする。
- 監査に必要なmetrics、feature importance、schema、manifest、10 modelとOOF本体を
  `kaggle/output/stage_b_v5/`へ取得した。大きなOOFはKaggle生成物でありGit管理対象にしない。
- Stage Bの実行承認は消化済み。次候補は同じ88列・dual objectiveをouter 5 × inner 4で生成する
  修正版Stage C 40 CPU boostersだが、旧承認は継承せず、新規承認が得られるまでpushしない。

## 2026-07-18 corrected Stage C実行承認

- ユーザーから修正版Stage Cへ進む明示承認を受領した。
- 実行scopeは1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU selector boosters。
  control/親再学習0本、GPU booster 0本。Stage D、inference、competition submissionは含めない。
- 同一run内で修正版Stage Aを再実行し、training-only formation raw/deltaを含まない88列schema
  `aaef4ffd...ddd3a4`を再発行・照合してからStage Cを開始する。旧100列Stage C model・compactは読まない。
- outer-trainはinner OOF、outer-validはouter-train内4 inner model ensembleとし、40 model、
  25 compact partition、18,919,945 compact rows、45,407,868 outer-valid candidate-long rowsを期待する。
- CPU、internet off、入力kernel sourceはexp263だけ。push前にcanonical package、bootstrap config、
  feature/schema契約、variant/objective/fold/booster数をfail-closedで確認する。

## 2026-07-18 corrected Stage C version 6 push

- credential checkerはKaggle CLI OAuthとlegacy credentialを利用可能と確認した。token実値は出力していない。
- canonical kernelを事前pullし、ID `kentookumura/exp264-exp263-confidence-dual-selector-train`、
  id_no `127485868`、private、CPU、internet off、exp263単一kernel sourceを確認した。
- strict experiment/project validation、JSON/YAML、Jupytext test、py_compile、ruff F821/F401/E9、
  repository 157 testsをPASSした。Kaggle Notebookのローカル実行はしていない。
- packageは1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boosters、control/親再学習0本。
  Stage D、GPU、inference、competition submissionは含めない。
- package SHA256はnotebook `d06d7cde...fddf`、config `b6118326...bf24`、metadata
  `b50f33c0...1c9a`、pipeline `9c1b77cb...c0b5`、train source `927d5723...1bbe`。
- notebook bootstrap ZIPをpush前後に展開監査し、埋め込みconfig SHA `b6118326...bf24`、
  `stage=nested_compact_meta`、`run_approved=true`、40 CPU boosters、GPU/internet offが一致した。
- 同じcanonical IDへversion 6をpushし、直後statusは`KernelWorkerStatus.RUNNING`。post-push pullでも
  id_no、runtime、sourceとbootstrap configを再確認した。重複実行防止のためlocal `run_approved=false`へ戻した。

## 2026-07-18 corrected Stage C version 6完了・成果物監査

- ユーザーの完了連絡後、CLIで`KernelWorkerStatus.COMPLETE`を確認した。生成物一覧表示は
  3,443.356秒、notebook HTML完了は3,451.631秒。Traceback、ERROR、OOM、DeadKernelはなく、
  nbconvert/mistuneのSyntaxWarningだけだった。kernelはcanonical ID、version 6、id_no `127485868`、
  CPU、internet off、exp263単一sourceのまま。
- 1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boostersを完走した。control/親再学習、
  GPU、downstream TVT学習、inference、Viterbi、submissionは0。
- pooled expected-error MAEはprior 5.788783から3.798819、within10 loglossは0.510131から
  0.359412、Brierは0.165095から0.111830へ改善。3指標ともpooledかつ5/5 folds改善し、
  score guardはPASS。Stage B比ではMAE +0.003018、logloss -0.000560、Brier -0.000621で、
  nested化しても校正品質をほぼ維持した。
- leakage auditはPASS。outer-valid wellはinner assignment、fit、early stoppingから除外され、
  inner train/validはwell-disjoint。outer-train compactはinner OOF、outer-valid compactは4 inner model
  ensembleから生成された。
- hard top1 RMSEは8.652532でfixed `exp226_w500_50_50` 8.238332より+0.414200悪化し、改善foldは
  1/5。fold差は+0.800053 / -0.059620 / +1.041497 / +0.068543 / +0.216519でhard guardはFAIL。
  hard selector、Viterbi、softmax TVT平均、submissionには使わない。
- 出力は40 models、25 compact partitions、18,919,945 compact rows、outer-valid candidate-long
  45,407,868 rows。40/40 modelの組合せ・byte SHAはmanifestと一致し、best iterationは78–176、
  model総量は31,171,472 bytes。25 partition manifestはpath/role/source/rows/wells/model count/SHAが
  partition CSVと一致し、各downstream foldにtrain 4 + valid 1が存在する。
- 巨大compact Parquet本体は取得していない。Stage Dを実行する場合、fit前にKaggle入力上の25ファイル
  すべてをmanifestのbyte SHAで照合し、不一致・metadataだけの入力・欠損partitionを拒否する。
- feature schema logical SHAは`aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`、
  compact schema logical SHAは`23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
  nested metrics / model manifest / compact manifest file SHAは順に
  `421376abe55378ac3e0d9c37d2a882a97d6fd4ebf866ec4395e7346c1d4f478b` /
  `3f28b04a017f0edadb815be44bf6f6f039dbd115bb72b3401e2681d3002422d2` /
  `f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c`。
- 40-model平均gain shareはpred-abs-errorでbank 54.909%、ctx 26.848%、formula 8.756%、
  cand 4.902%、conf 4.247%、id 0.337%。within10はbank 56.684%、ctx 33.899%、cand 4.328%、
  formula 3.294%、conf 1.446%、id 0.349%。`conf__native__sigma_tvt`はpred-abs-error 5位・2.841%、
  within10 33位・0.477%で、confidenceは主に誤差校正へ寄与した。
- 判定は「nested compact生成に合格、hard選択に不合格」。次の外部実行はclean 273 control 15本と
  clean 273 + compact 74 = 347列add-only 15本を同条件比較するStage Dだけだが、30 GPU boostersと
  control再学習を含むため別承認までdisabledを維持する。

## 2026-07-18 corrected Stage D実行承認

- ユーザーから修正版Stage Dを実行する明示承認を受領した。
- 実行scopeは`matched_control` clean 273列と`selector_compact_addonly` 347列の2 variants、
  exp218固定`lgb0/lgb1/lgb2` 3 configs、Stage C固定5 downstream folds。合計
  `2 × 3 × 5 = 30` GPU boostersで、control 15本、add-only 15本。
- 旧380列controlは107列が非fold-safeで無効であり、clean 273 controlの信頼できる保存済みOOF/modelは
  存在しない。74 compact列の効果を同一行・fold・config・GPU runtimeで帰属するため、control 15本を
  再学習する。親exp218 model、旧Stage D model、Stage C selector、PF/Beam/HMMは再学習しない。
- 入力Stage Cはcanonical kernel version 6だけを使う。metrics/model/compact/schema SHAに加え、
  25 compact Parquetをfit前に全件byte SHA検証し、metadata-only、欠損、0 byte、SHA不一致を停止する。
- baseは履歴的380列名を再構築後、SHA固定allowlist `d01a73cc...77bf`で273列だけを順序保持して選ぶ。
  source 380、selected 273、drop 107、重複0、全finiteをfit前に検証する。
- runtimeはKaggle T4、internet off、`gpu_use_dp=true`、deterministic / force-col-wise、thread 8、
  early stopping 250。出力は30 models、OOF、fold/bucket/hidden-like/by-well、重要度、manifest/SHA。
- hard selector、Viterbi、softmax TVT平均、inference、competition submissionはscope外。
- 旧Stage D承認欄だけではcost guardを通せないよう、`corrected_run_approval_received_at/scope`と
  clean 273/347列の完全一致を`stage_d_cost_contract`へ追加した。

## 2026-07-18 corrected Stage D push前監査

- credential checkerはKaggle CLI OAuthとlegacy credentialを利用可能と確認した。credential実値は出力していない。
- canonical Stage D kernel `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train`を事前pullし、
  id_no `127577193`、title、private、T4、internet offを確認した。入力Stage C kernelもcanonical
  `kentookumura/exp264-exp263-confidence-dual-selector-train`、id_no `127485868`として存在を再確認した。
- 実行契約は2 variants × 3 configs × 5 folds = 30 GPU boosters。clean 273 control 15本と
  clean 273 + compact 74 = 347列add-only 15本。親/control以外の再学習、追加variant/config/fold、
  inference、submissionは0。
- JSON/YAML、`stage_d_cost_contract`、py_compile、ruff F821/F401/E9、Jupytext test、strict experiment/project、
  exp264対象15 tests、repository全157 testsをPASSした。Kaggle Notebookのローカル実行はしていない。
- canonical packageはprivate、T4、internet off、run-on-push。kernel sourceはStage C v6、exp072 cache、
  exp145 trainの3件、competition sourceはROGII 1件。dataset/model sourceは0。
- source/package SHAはconfig `5ea36d32...daa7`、pipeline `382ed81c...05c4`、train source
  `927d5723...1bbe`で一致。package notebook `ddc2061d...8cd2`、metadata `4925dfe7...6c0e`。
- bootstrap ZIPは34 entries。埋め込みconfig/pipeline/train SHAもsourceと一致し、stage
  `downstream_tvt_ablation`、`run_approved=true`、修正版approval scope、T4、3 kernel sourcesを確認した。
- bootstrap内exp218 source/config、clean 273 allowlist、hidden-like assignment SHAは順に
  `6712d5b4...f7a33` / `fd547494...fa09d` / `d01a73cc...77bf` / `5f9ac9fa...a6597`でsourceと一致。
- Stage D開始時にStage C v6 metrics/model/compact/schemaの固定SHAと25 Parquet本体のbyte SHAを
  notebook内で検証する。検証完了前には1本目のLightGBM fitへ進まない。

## 2026-07-18 corrected Stage D Kaggle T4 version 3開始

- canonical kernel `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train`へ
  `--accelerator NvidiaTeslaT4`を明示してpushし、`Kernel version 3 successfully pushed`を確認した。
- push直後のCLI statusは`KernelWorkerStatus.RUNNING`。同じkernelを再pullし、id_no `127577193`、
  private、T4、internet off、Stage C v6 / exp072 / exp145の3 kernel sourcesを確認した。
- remote notebookはKaggle再serializationによりfile SHA `d086c7ce...7c57`となったが、埋め込み
  config/pipeline/train SHAはpush前の`5ea36d32...daa7` / `382ed81c...05c4` / `927d5723...1bbe`と一致。
  stage、`run_approved=true`、修正版approval scope、T4も一致した。
- 実行scopeはclean 273 control 15 + 347列add-only 15 = 30 GPU boostersだけ。Stage C/exp218/PF/Beam
  の再学習、追加variant/config/fold、inference、submissionは0。
- 重複実行防止のため、push後のlocal configは同じStage Dのまま`execution.run_approved=false`へ戻した。
  実行中version 3の正はremote notebook bootstrap内のpush前config SHAとする。
- CLIの`RUNNING`は起動確認であり、preflight/学習完了を意味しない。完了連絡後にlogsと必要な
  metrics/manifest/model/OOFを監査する。同じkernelへ再pushしない。

## 2026-07-19 corrected Stage D Kaggle T4 version 3完了・結果監査

- CLI logsとstatusで`KernelWorkerStatus.COMPLETE`、30/30 boosters、生成物一覧、notebook HTML変換完了を確認した。
  30本目は26,366.334秒、生成物一覧は26,373.956秒、HTML完了は26,384.758秒。
- Stage C v6の40 models / 25 compact partitions / 18,919,945 rows、nested score guard PASS、nested
  leakage audit PASSをreproducibility manifestで再確認した。25 partitionは学習前のbyte SHA gateを通過した。
- clean 273 control `lgb_mean`は10.476169、347列add-onlyは8.460811で、deltaは-2.015358。
  fold deltaは-2.103882 / -1.322387 / -1.432684 / -2.090369 / -2.989973で5/5 folds改善した。
- distance bucketはnear 0-250が-0.445903、250-1000が-0.756786、1000+が-2.233208。
  hidden-like spatial / typewell-purgedは-3.073014 / -3.091639で、事前の非悪化条件を通過した。
- 773 well中518改善、255悪化。+1 ft超135、+3 ft超39、+5 ft超14。worst `70925e23`は
  11.825487から26.308360へ+14.482873悪化し、上限+0.25を超えたため総合guardはFAIL。
- compact 74列はadd-only全体の15-model平均正規化gain 76.9258%、split 25.2013%。上位4つの
  legal-domain top1候補値-minus-anchorがgain 61.0343%を占め、`beam_mean`予測誤差scoreが5位5.8196%。
- 小さい生成物だけを`kaggle/output/stage_d_v3_corrected/artifacts/`へ保存した。171 MB OOFとmodel本体は
  取得せず、Kaggle outputのSHAをreproducibility manifestから記録した。metrics SHAは
  `29cecbf7...0acb`、OOF SHAは`b11c5005...9ae2`、model manifest SHAは`c3b22481...5fcc`。
- 判定は「有効なmatched ablationだがworst-well guard FAIL」。hard selector、Viterbi、softmax TVT平均、
  corrected Stage C v6 / Stage D v3を使うinference、competition submissionは実行しない。

## 2026-07-19 corrected inference・提出のユーザー承認とpush前監査

- worst-well guard FAILを保持した参考提出であることを明示した上で、ユーザーからcorrected Stage C v6 /
  Stage D v3によるinferenceと、submit-check PASS後のcompetition submission 1件の明示承認を受領した。
- inferenceは88 selector特徴から40 selector modelsで候補別scoreを推定し、同じ処理内で74 compact列へ
  決定的変換する。hard selector、Viterbi、候補softmax平均は行わない。最終TVTはclean 273 + compact 74
  = 347列を入力するStage D add-only 15 modelsの等重み平均とする。再学習boostersは0本。
- corrected Stage C v6 bundleをprivate dataset `kentookumura/exp264-stage-c-selector-models`へ更新し、
  Kaggleから再downloadしたbundle SHAがlocalと一致することを確認した。bundle SHAは
  `0e1ae1a59047ddbba8037b63e06bddbd8b903b378f944e37ecb75f52440b5adf`、44 entries、12,616,690 bytes。
- canonical inference packageはprivate、CPU、internet off、run-on-push。config/package SHAは
  `497cb3e95f6e68f87eaf75e02f5ed35bf20dd8dca8cf6f7cad24adb298dad897`、notebook SHAは
  `b54d044f541f73ae304fe03ea77f6f3dd332a2ad6c661c65e0e15ddb2eae7c2b`、metadata SHAは
  `270124722f4ff43001cc8ccb3c914d7cfb43529e10770a7a169084269de76906`。
- bootstrapは35 support files。埋め込みcatalog / clean allowlist / source availability SHAは順に
  `3a443a1a...1235` / `d01a73cc...77bf` / `6f93a502...9c67`で正規sourceと一致した。
- py_compile、ruff F821/F401/E9、Jupytext test、strict experiment/project、exp264対象15 tests、
  repository全161 testsをPASSした。Kaggle Notebookのローカル実行はしていない。
- notebook自体からsubmit APIは呼ばない。Kaggle完了後に実ファイル`submission.csv`をsample submissionと
  照合し、PASSした場合だけ外部CLIからversion/fileを固定して1件提出する。

## 2026-07-19 corrected inference Kaggle version 4開始

- 既存canonical kernel `kentookumura/exp264-stage-d-hidden-safe-inference`を事前pullし、id_no
  `127732365`、private、CPU、internet off、正しい8 kernel sources、Stage C private datasetを確認した。
- 同じslugへ検証済みpackageを1回pushし、`Kernel version 4 successfully pushed`と
  `KernelWorkerStatus.RUNNING`を確認した。再pushせず、このversion 4だけを監視する。
- 実行scopeはcorrected Stage C v6 40 selector modelsによる74 compact列生成と、corrected Stage D v3
  add-only 15 TVT modelsの等重み推論だけ。学習boosters、hard selector、Viterbi、候補平均、submit APIは0。

## 2026-07-19 corrected inference version 4完了・提出

- status `KernelWorkerStatus.COMPLETE`。runtime 424.511秒、14,151 rows / 3 wells / 12 candidates、
  namespaced confidence 21列、selector 88列・40 models、compact 74列、source base 380列からclean 273列、
  final 347列、Stage D TVT 15 models、学習boosters 0本を実測確認した。formula parityの最大絶対誤差は0。
- `submission.csv`は`id,tvt`の14,151行でsample submissionとheader・行数・ID順が完全一致、重複ID 0、
  empty/NaN/Inf 0。skill checkerとrepository checkerがともにPASSした。SHAは
  `cbaad9a3603008f4adaaf0c53a3369aa47f0fd95db8711ad0d005116663297b7`。
- user-authorized scopeに従い、kernel version 4 / `submission.csv`をcompetitionへ1回提出した。
  submission refは`54818932`、descriptionは`exp264 corrected Stage C v6 Stage D v3 compact add-only reference`。
- 提出時点のstatusは`PENDING`。ユーザー指示によりローカル監視プロセスは停止したが、Kaggle側の採点は
  継続する。完了連絡後にpublic scoreを取得して実験記録と提出台帳を確定する。
- worst-well +14.482873のStage D guard FAILは維持する。この提出はreferenceであり、scoreだけで
  train-side guardをPASSへ変更しない。
- submission一覧には本提出の約3分前にdescriptionなしの別pending ref `54818883`も存在した。
  この作業で明示的に送信した提出はdescription付きref `54818932`の1回だけで、これを正として追跡する。

## 2026-07-19 Public LB確定

- monitorのread-only確認でref `54818932`は`SubmissionStatus.COMPLETE`、Public LB **7.562**、
  private scoreなしと確定した。submitted UTCは`2026-07-19 00:46:31.027000`。
- ML submitted anchor exp274 / 7.715から-0.153改善したため、Public-LB上のML anchorをexp264へ更新する。
  別routeのensemble anchor exp082 ref `53885305` / 7.601も-0.039で上回るが、ensemble anchorはexp082に維持する。
- version 4完了時に作成されたdescriptionなしref `54818883`もCOMPLETE / 7.562だった。同一kernel runに
  由来する自動submission recordと判断し、明示submit ref `54818932`のduplicateとして同じv070へ記録する。
- Stage Dのworst-well +14.482873 guard FAILは維持する。LB anchorの更新であり、train-side guard PASS、
  hard selector、Viterbi、候補softmax平均の採用を意味しない。

## 2026-07-20 route分類修正

ユーザー指示によりrouteを`ensemble`から`ml_model`へ修正した。PF/HMM/Beam候補はselectorの補助meta
featureに限定され、direct blend、hard-path、Viterbi、softmax TVT平均は使わず、最終予測をdownstream
LightGBMが生成するため、主目的基準ではML routeとする。CV、LB、生成物、SHA、guard判定は変更しない。
`kaggle/output/`配下の実行済みJSONに残る旧route文字列は、byte SHA証拠を改変しないため履歴として保持する。

## 2026-07-19 OOF診断notebook・viewer CSV作成

- 参照元はexp238 selector-confidence probe `scriptVersionId=336248071`とLikPF 128-path probe
  `scriptVersionId=336196931`。現行Kaggle sourceと既存exp238実装を構成参照し、exp264用の別名Jupytext sourceと
  正規`.ipynb`を作成した。既存notebookは上書きしていない。
- selector-confidenceはcorrected Stage C v6 strict nested outer-valid candidate scoreを使う。入力は
  45,407,868行 / 12 candidates、SHA `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`。
  2 legal domainを分離し、primary/fixedのpredicted-error・within10 marginを表示する。hard top1 RMSE
  8.652532 / guard FAILを診断として保持する。
- final overlayとviewer CSVはcorrected Stage D v3 OOFの
  `selector_compact_addonly__lgb_mean__pred_tvt`だけを使う。入力Parquetは3,783,989行 / 773 wells、
  SHA `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`。旧Stage D v2は使わない。
- LikPF notebookはexp072と同じ500 particles × 128 stable seedsをraw trainから再生し、保存済み
  `likpf_mean_d`とのexact parity後にcorrected Stage D v3 final OOFをoverlayする。学習・blend・submitはしない。
- viewer CSVは`id,tvt`、3,783,989 unique ID、773 wells、NaN/Inf 0、RMSE 8.460811237612477。
  repository viewer loaderで読込を確認し、SHAは
  `9fe0cfceda8b8e3d852c74352e0e4d7d6748f057b79354133b110e77173ce04b`。
- canonical notebook source SHAはselector `1b9280c9...e0f`、LikPF `1258ba2a...01c`。
  `.ipynb` SHAはselector `bcf23f96...d9e1`、LikPF `32e1d7af...623`。両方16 cells、output/execution count 0。
- Kaggle packageは`kentookumura/exp264-oof-selector-confidence-probe`と
  `kentookumura/exp264-oof-likpf-128-paths-probe`としてprivate・CPU・internet off・run-on-push falseで準備した。
  push、Kaggle実行、773 well plot生成、competition submitは実施していない。
- py_compile、ruff F821/F401/E9、Jupytext convert/test、strict exp validation、exp264 15 tests、
  Kaggle notebook 4 testsをPASSした。viewer CSVはheaderを含め3,783,990 linesでmanifest SHAと一致した。

## 2026-07-19 OOF診断2本のKaggle実行承認

- ユーザーからselector-confidenceとLikPF 128-path notebookの実行指示を受領した。前回TODOにしたCPUコストと
  773 well plot scopeを再確認し、この指示を2本とも全773 wellsで実行する明示承認として扱う。
- selector-confidenceはcorrected Stage C v6 / Stage D v3を読み、773 well PNG、manifest、distribution、
  summary、plots zipを生成する。exp238参照runtimeは約925.583秒。
- LikPFは500 particles × 128 stable seeds × 773 wellsをraw trainから再生し、保存済みexp072 meanとの
  exact parity後に773 well PNG、manifest、summary、plots zipを生成する。exp238参照runtimeは
  14,067.881秒（約3時間54分28秒）。Kaggle CPU core数に応じて`n_jobs`を縮小する。
- 2本ともprivate、CPU、internet off。学習variant 0、model config 0、fold学習0、booster 0、
  parent/control再学習なし、GPU 0、prediction blend 0、submission生成・competition submit 0。
- PF seedは`likpf / train / well_id / seed_index`からstableに生成し、global RNGやthread schedulingへ依存しない。
  corrected Stage C/D・exp072入力とbootstrapをSHA/row/well/parityでfail-closedに検証する。

### OOF診断2本のpush前監査

- canonical slugの事前pullは2本とも`403 Forbidden`。初回作成前の状態として想定どおりで、別slugは作らない。
- `run_on_push=true`でpackageを再生成し、両方17 cells、output 0、private、CPU、internet offを確認した。
  selectorはkernel source 3、LikPFはkernel source 2。id/titleのslugは完全一致する。
- selector package notebook SHAは`21c7c99a...6635`、metadata SHAは`451b2929...7d12`。
  LikPF package notebook SHAは`5232ce34...a4da`、metadata SHAは`d44446b5...527d`。
- strict experiment validationとexp264/Kaggle notebookの19 testsをPASSした。canonical sourceに`__file__`はない。

### OOF診断version 1 pushとselector起動時ERROR

- selector `kentookumura/exp264-oof-selector-confidence-probe` version 1とLikPF
  `kentookumura/exp264-oof-likpf-128-paths-probe` version 1を各1回pushした。URLは各canonical slug。
- post-push pullでselector id_no `127868315`、LikPF id_no `127868327`、private / CPU / internet off、
  source 3 / 2、remote/local 17 cell source完全一致を確認した。
- LikPF version 1は`RUNNING`。selector version 1は29.737秒で`ERROR`。bootstrap、corrected Stage C/D、
  exp065 typewell入力、Stage D 3,783,989行 / 773 wells / RMSE 8.460811まではPASSした。
- 原因はStage C strict nested scoreがouter-fold順、Stage D OOFがglobal well/row順なのに、row-groupの連続offsetが
  同じID順だと誤って仮定したこと。row group 0の3,836行目から順序が分岐し、既存guardが描画前に停止した。
- 修正はStage D unique ID indexで各Stage C base rowを配置し、missing ID、row-group内重複、row-group間重複、
  well/fold/downstream fold、全3,783,989行coverageをfail-closedで検証する。candidate block順は変更しない。
- 最初の全行ローカルsmokeはメモリ常駐実装のためexit 137。監査側だけmemmap streamingへ変更し、
  Stage C 45,407,868 candidate-long行から3,783,989 / 3,783,989 base rows、773 wellsを重複なく対応させ、
  primary hard top1 RMSE `8.652531955610227`が既存値へ完全一致した。
- ID alignment回帰テストを追加し、py_compile、ruff F821/F401/E9、Jupytext test、strict validation、
  exp264/Kaggle notebook 20 testsをPASSした。同じselector canonical slugへversion 2をpushする。

### selector version 2開始

- 修正版package notebook SHA `4db359da...7807`を同じcanonical kernelへversion 2として1回pushした。
  post-push pullは同じid_no `127868315`、remote/local 17 cell source完全一致、private / CPU / internet off、
  kernel source 3を確認した。
- version 2は`RUNNING`。CLI logsは実行中の既知挙動どおり空だが、version 1の29.737秒を越えた後も
  `RUNNING`であり、旧offset guardでの即時停止は再発していない。別slug・追加pushは行わない。
- LikPF version 1も`RUNNING`を維持する。実行中logsが空であることを失敗根拠にせず、同じslugを追跡する。

### selector version 2完了・描画契約による無効化

- version 2は`COMPLETE`。3,783,989 rows / 773 wellsをcoverageし、773 plotsを生成した。実測runtimeは
  669.522秒、final OOF RMSE 8.460811237612477、primary hard top1 RMSE 8.652531955610227で、ID修正は有効。
- ただしユーザー確認により、exp238にないmatched-control/fixed-domain線、probability twin axis、変更した
  panel比率を含み、exp238のPF/Beam/LikPF/exp226/exact-HMM/-Z比較とHMM ±2sigma帯を欠くため、plot成果物は無効。
  selectorの候補集合・primary top1結果だけはexp264へ変更してよいという補足を受領した。

### exp238描画契約への復元と修正版package

- selector結果はcorrected Stage C v6 primary-domain predicted-error top1とtop2-top1 marginを維持した。
  図はexp238と同じ3段比`[7.0, 1.5, 0.65]`へ戻し、main panelをtrue TVT、exp264 final OOF、selector top1、
  LikPF、PF ANCC、Beam、exp226 K16、exp209 exact HMM、exact HMM ±2sigma、-Z min-maxだけに固定した。
- `REFERENCE_LINE_COLORS`、linewidth、linestyle、alpha、zorder、margin panel、top1 bandの構成をexp238へ一致させ、
  fixed-domain/p-within10/controlはplotせずsummary-only監査値とした。exp209の`hmm_mean_tvt`/`hmm_std`と
  exp072/exp226入力を追加した。
- 回帰テストで色辞書、panel比、main path集合、HMM ±2sigma、twin axis不在を固定した。py_compile、ruff、
  Jupytext convert/test、strict validation、exp264/Kaggle notebook 21 testsをPASS。packageは17 cells、output 0、
  private/CPU/internet off/run-on-push、kernel source 6、notebook SHA `1cd157eb...b182`。
- 同じcanonical slugへのpushは4回試したが、Kaggleの`Maximum batch CPU session count of 5 reached`で
  version作成前に拒否された。LikPF v1と他の承認済みCPU runを停止せず、空きができ次第同じpackageを再pushする。

### exp238描画契約版selector version 3開始

- ユーザーから再度実行指示を受領し、credential checkerでOAuth/legacy CLI認証を確認した。API tokenは未設定だが
  OAuth credentialでCLI操作可能。CPU枠解放後、既存canonical kernelを事前pullし、id_no `127868315`、
  version 2の3 kernel sources、private/CPU/internet offを確認した。
- notebook SHA `1cd157eb6f70f5f1504dda8656300f2a52098ca4617c774b3887e55c3a68b182`の検証済みpackageを
  同じ`kentookumura/exp264-oof-selector-confidence-probe`へpushし、version 3として実行を開始した。
- post-push pullでid_no不変、6 kernel sources（exp072、corrected Stage C/D、exp209、exp226、exp065）、
  private/CPU/internet off、17 cells、remote/local cell source完全一致、HMM ±2sigma実装を確認した。
- 起動後90秒超で`RUNNING`を維持し、version 1の29.737秒alignment ERROR時間を通過した。実行中logsは
  Kaggle CLIの既知挙動どおり空であり、再pushせず同じversion 3を追跡する。LikPF version 1も`RUNNING`。

### OOF診断2本の完了・成果物監査

- selector-confidence version 3とLikPF 128-path version 1はともにKaggle status `COMPLETE`。
  private / CPU / internet off、model fit 0、booster 0、submission生成・competition submit 0の診断scopeを維持した。
- selector version 3は3,783,989 rows / 773 wellsをcoverageし、773/773 PNGを生成した。summary runtimeは
  732.203秒、notebook全体は約1,026.305秒。final OOF RMSEは8.460811237612477、primary predicted-error
  top1 RMSEは8.652531955610227、margin mean / p50 / p90は0.311003 / 0.086398 / 0.932687だった。
- selector summaryの`plot_contract`は3 panels、height ratio `[7.0, 1.5, 0.65]`、
  `reference_paths_and_colors_unchanged=true`、`exact_hmm_sigma_band=mean_plus_minus_2sigma`を記録した。
  代表PNGを目視し、true TVT、exp264 OOF、selector top1、LikPF、PF ANCC、Beam、exp226 K16、
  exp209 exact HMM、exact HMM ±2sigma、-Z min-maxと色・線種がexp238契約どおりであることを確認した。
  version 2の無効plotは採用しない。
- selector plot manifestは773 rows / 773 wells / 773 unique plot paths、対象行合計3,783,989。
  summary SHAは`70eab1a000caf89fed7f6bc9f2138806f5180b27f75d95c993c1b304e0f3f869`。
- LikPF version 1は500 particles × 128 stable seedsを773 wellsで再生し、773/773 PNGを生成した。
  summary runtimeは12,526.219秒、notebook全体は約12,754.892秒。exp264 OOF RMSEは
  8.460811237612477、PF seed-mean RMSEは11.594897672217703。
- LikPF plot manifestは773 rows / 773 wells / 773 unique plot paths、対象行合計3,783,989。
  全wellでseed count 128、particles 500、保存済みexp072 meanとのexact parityがtrue、最大絶対差と
  weighted mean absolute差はいずれも0。summary SHAは
  `ffc85e804d564dc0c1ade8245dceafb87b7e78b446e05495d5c36ceb6bec94d0`。
- output archive全体は取得せず、実ファイル確認に必要なsummary JSON、plot manifest、各代表PNGだけを
  `kaggle/output/oof_selector_confidence_probe_v3/`と`kaggle/output/oof_likpf_128_paths_probe_v1/`へ取得した。
  viewer CSVは既存監査どおり3,783,989 unique ID / 773 wells、finite、loader互換、SHA
  `9fe0cfceda8b8e3d852c74352e0e4d7d6748f057b79354133b110e77173ce04b`で確定する。
