# exp510_exp413_exact_public_preoverride_hedge セッションノート

## 目的

最終提出2枠の第2枠として、LB 7.159公開sourceのpre-override dual-pipeline componentを
exp413へ固定10%だけ加えるPublic分布hedgeを実装する。

## 現在の状態

- Route: `ensemble`
- 状態: hidden-safe version 4 technical PASS / code submission COMPLETE
- CV / LB: なし / Public 7.201（exp413と公開3桁同値、full precision不明）
- implementation / Kaggle package / run: `1 / 1 / 1`
- Kaggle output files listed / output archive retrieved / submission: `1 / 1 / 2`（v2 hidden rerun失敗、v4成功）
- 正規train / inference notebook: template placeholder、科学ロジックなし
- Jupytext候補inference: `.py/.ipynb`実装済み、専用current-test copyをKaggleで実行済み
- blocker: なし。正規notebook採用と最終portfolio採否は別判断。

## 2026-08-04 設計確定

- public予測をdual-pipeline core、guarded contact、Gold overlayの3層へ分解した。
- final source cellの実際の入力に合わせ、public componentを
  `0.55 * sp45_projection_submission + 0.45 * submission_B`へ固定した。
- architecture説明にある`submission_A.csv`はfinal source cellで未使用のため、代入を禁止した。
- freeze pointをguarded contact override呼び出し前に固定し、Goldも完全OFFとした。
- exp413とのfinal weightを`0.90 / 0.10`へ固定した。
- 対応するhonest OOFがないため、CV promotionを主張せず第2枠hedgeに限定した。
- public output copy、artifact欠落時のinference-time training、weight tuningを禁止した。
- PF/likelihood-PFはstable SHA256 per-well seedへ移植し、公開artifact byte parityは目的外とした。

## 根拠

- exp082 hidden-compatible source-portではSP45単枝Public LB `7.857`からdual-pipeline
  `0.55/0.45` blendで`7.601`へ`0.256`改善した。
- exp413 Public LBは`7.201`。
- public `7.159`はnotebook title/lineage情報で、Gold/contact/seed/artifact差を同条件で
  分解した提出結果はない。
- exp497 strict public-coreは平均相補性を示したがfold/hidden/tail gateをFAILしており、
  exact public componentの大weightを正当化しない。
- exp508 SG61/p3のexp413 OOF gainは`0.006133728 ft`で、大きな公開差の主因ではない。

## source identity

- archived source:
  `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260627/degnonguidi__public-score-rogii-lb-7-159/public-score-rogii-lb-7-159.ipynb`
- SHA256: `4d0712983788dc7d9b97fdb8e5dc7c30b6d3634a9c64597d84d21da28e9623eb`
- source kernel: `degnonguidi/public-score-rogii-lb-7-159`

## 実行予定inventory

| scientific variant | new model config | fold | booster | inference training | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |

PF/Beam run数はhidden testのnonempty well数で動的に決まるため、実装後かつpush前に正確に記録する。
今回はpushしない。

## コマンドログ

- `task new-steering ...`は環境に`task`がなく失敗した。代替の`make new-steering`でsteeringを作成した。
- `make new-exp EXP=exp510_exp413_exact_public_preoverride_hedge`でtemplate scaffoldを作成した。
- `sha256sum`でarchived public source SHAを記録した。
- `make validate-exp EXP=exp510_exp413_exact_public_preoverride_hedge`はstrict PASS。
- `make validate-template`はPASS。
- `make update-summary`で`experiment_summary.md`へdesign-only実験を反映した。
- notebook実行、Kaggle API、学習、推論、提出は行っていない。

## 再現性メモ

- seed policy: `stable_sha256(split, family, well, seed_index)`。
- stochastic components: PF、likelihood-PF。
- parallel RNG: immutable per-well seed bank、global RNG禁止。
- CPU/GPU runtime: 将来のKaggle CPU、GPU 0、internet off、inference-time training 0。
- source SHA: 記録済み。dataset/model/feature SHAはpreflight待ち。
- prediction / submission SHA: 未生成。
- deterministic anchor: false。rerun一致前は昇格しない。

## 次のアクション

1. 技術実装・hidden rerun・scoring・記録は完了。追加提出やweight変更は行わない。
2. 正規notebook採用と最終portfolioでの採否は引き続き別判断とする。

## 2026-08-04 実装

### 承認範囲

- ユーザーの「exp510を実装してください」を、dataset/model preflight、候補source/notebook、
  contract test、実験記録更新までの承認として扱った。
- 既存steeringどおり、正規`*_inference.ipynb`への採用、Kaggle package/push/run、output取得、
  submit-check、外部提出は承認範囲に含めていない。

### artifact preflight

- archived kernel metadataは6 datasetと1 kernel sourceを列挙するが、可視pre-override境界の
  実行に必要なartifactをコード参照単位で再監査した。
- `phongnguyn23021656/koolbox-offline`: current version 1。候補ではprivate helperをimportしない。
- `fleongg/rogii-claude-models-pub`: current version 1。runtime必須。
  - `features.json`: `ea9042f88cb3d8716b83e40c5c5ecb39f8bc8fcfeb52edb40d1871cd99496308`
  - `lgb0.pkl`: `a6451b3c42aeace6778e952b088287654946dca5412b818990d3f6b397e501e1`
  - `lgb1.pkl`: `4d61ab162af864bd3cfe37bde4421299746f28147faa3239e1ad14f15453f547`
  - `lgb2.pkl`: `1ee24121ecf455d904f3433bba49857d076fc33ca0b6b7a71ff9d538b3b8acf5`
- `ravaghi/wellbore-geology-prediction-artifacts`: current version 6。元sourceではPipeline-A
  tabular fallback用の7GB train featureとtrainerを提供するが、exp510はSP45 fallback rows 0を
  必須とするためruntime routeから削除した。
- metadataにある`nina2025/rogii-03`、`thbdh5765/rogii-v10-fresh-artifacts`、
  `needless090/rogii-tabicl-mirror`、package-manager kernelは可視pre-override candidateの
  runtime非依存としてconfigへ記録した。
- Kaggle CLI `datasets status`はpublic datasetに404を返したため、Kaggle公開dataset view APIの
  `currentVersionNumber`を用いてversionを確認した。

### 実装内容

- archived source SHA `4d071298...623eb`を再確認し、final blend cellが
  `sp45_projection_submission.csv`と`submission_B.csv`を読む境界をcontract testで固定した。
- source notebookの該当cellだけをJupytext候補へ抽出し、学習cellと境界後のcellを含めなかった。
- `*_compact_selfcontained_inference.py` 1486行と対応`.ipynb`を別名で作成した。正規notebookは未変更。
- projected-SP45をsample全IDで厳格検証し、元sourceのPipeline-A tabular fallbackを削除した。
- Pipeline BはSHA一致するversion 1の3 boosterだけを読む。欠落、複数候補、SHA不一致、
  unexpected feature gapで停止し、inference-time booster trainingを持たない。
- source/model schema差として事前固定した39列だけを0補完し、列名、feature schema/content SHA、
  zero-filled列をmanifestへ保存する。
- PF/likelihood-PFを`SHA256(base_seed, split, family, well, seed_index)`へ移植し、同じ
  likelihood bankをSP45とPipeline Bで再利用する。global RNGは使わない。
- test側のFormationPlaneKNN/DenseANCCでも同じwell IDを除外し、same-ID空間lookupを遮断した。
- exp413入力はraw gzip SHA `52ffb491...f136`とdecompressed content SHA
  `875a1334...dc4`の両方を照合する。
- fixed float64式、sample order、one-to-one ID、finite、duplicate、fallback 0、formula parity
  `<=1e-12`を全ANDとした。
- overall / well / MD horizon / start continuity readoutとsource/model/feature/prediction/submission SHA
  manifestを実装した。

### 実行inventory

| scientific variant | new model config | fold | new booster | saved booster | inference training | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

- per test well: likelihood-PF 48 seed runs + feature PF 3 single-seed runs = PF 51 runs。
- per test well: feature builder 7 + selector ensemble 14 = beam path 21 runs。
- public sample 3 wellsではPF 153、beam 63。hidden totalはpush前の動的well数で確定する。
- Pipeline-A/CV controlや親exp413は再学習しない。

### 検証ログ

- dedicated tests: `10 passed`。
- `py_compile`: PASS。
- Ruff `F821/F401/F811`: PASS。
- 禁止route AST/text scan: PASS。候補sourceに禁止symbol、public notebook output path、
  global RNG、booster training class/callはない。
- Jupytext round-trip、strict experiment validation、template validation: PASS。
- repository-wide testsは、既知の別実験contract test 2ファイルを除外して
  `1821 passed, 8 skipped`。除外対象を含む実行ではexp293の記録SHA不一致2件とexp296の
  既存status/error文言不一致2件だけが失敗し、対象2ファイル単独では`4 failed, 21 passed`だった。
  exp510 dedicated testsと実装ファイルには波及していない。
- notebookローカル実行、Kaggle API push/run、output取得、提出は行っていない。

### 親compact比較

- 親のcurrent-test inference sourceは1563行、exp510候補は1486行。
- exp510候補はImports、runtime/SHA、public physics/features、setup/input、SP45、Pipeline B、
  fixed blend、metrics/manifestの8役割を持ち、同一exp helperをimportする薄いnotebookではない。

## 2026-08-04 Kaggle current-test実行承認・push前監査

### 承認範囲

- ユーザーの「実行してください」により、Kaggle CPU/private/internet-off package、push、run、
  完了までのlogs監視を承認済みとした。
- 正規`*_inference.ipynb`は上書きせず、候補とbyte-identicalな
  `*_current_test_inference.ipynb`を実行対象として生成した。
- output archive取得、submit-check、competition submitは未承認のため行わない。

### 実行inventory

| scientific variant | new model config | fold | new booster | saved booster | inference training | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |

- 公開current testは14,151 rows / 3 wells。PF 153 runs、beam 63 paths。
- hidden code rerunではsample由来のnonempty well数から動的に増減する。
- kernel: `kentookumura/exp510-exp413-exact-public-preoverride-hedge-inference`。
- title: `exp510 exp413 exact public preoverride hedge inference`。

### package readback

- Kaggle credential checker: API tokenは未設定、OAuthとlegacy credentialは利用可能。
- metadata: CPU、private、internet off、`run_on_push=true`、competition source 1、
  dataset source `fleongg/rogii-claude-models-pub`、kernel source
  `kentookumura/exp413-scale5-likpf-current-test-inference`。
- bootstrap support files: 34。生成notebookのbootstrap以降のcellは候補と完全一致。
- embedded config / compact source / execution notebook SHA:
  `f119d5c7cfef5e4d17c1a26ea3ba318d9ef2affffcef2e13df9857dd61d3d031` /
  `0788998f7bee74b87af1753f8a5f1705581eb47710c8bf8323fcad81242425fa` /
  `2159852252f7446871a8b6ec970420c80d1d884d2907a4c6c042b73145d1cbd7`。
- bootstrap済みNotebook / metadata SHA:
  `5c99ca24dc353bcf2cc2551ac4504007109cc99a87162a4d1c36475e4664f34f` /
  `396ef945eb74624d9243db8d56e777c1fa35a37cc9805cc07b6fa3f9a666666a`。
- strict experiment validation、dedicated tests `10 passed`、Jupytext round-tripを再確認した。

### push attempt 1失敗

- kernel id/titleをfull experiment suffixで一致させてpushしたが、Kaggle `SaveKernel`は
  詳細なし`400 Bad Request`を返し、実行は開始しなかった。
- 初回slugは54文字。旧kernelのpull確認も`403`で、作成済みversionは確認できない。
- 科学条件、notebook、input、CPU/internet設定は変更せず、50文字以下かつexp510を一意に示す
  `kentookumura/exp510-exp413-exact-public-preoverride-inference`（48文字）へ短縮する。
- 新titleは`exp510 exp413 exact public preoverride inference`で、title由来slugとidを一致させる。
- 短縮slug packageのreadbackも34 support files、source cell保持、CPU/private/internet off、
  `run_on_push=true`でPASSした。embedded config / source / bootstrap済みNotebook / metadata SHA:
  `66c51060b9a6a738c44601f0f4e5919f162d380445f173e218716e212466777b` /
  `0788998f7bee74b87af1753f8a5f1705581eb47710c8bf8323fcad81242425fa` /
  `6f772989ad790a67de53643d387ca4d68feee672783fdd6a38c7694830e28c1e` /
  `b852bbd3d5cdc442129adc2c4b8f4d0edb093b27a73a9bf50d27f527c3bf9681`。

### Kaggle version 1失敗・version 2修正

- version 1（id_no `129634723`）は`ERROR`。約52.218秒で
  `NameError: run_particle_filter is not defined`となった。
- 原因はJupytext sourceの`## 3. Public-source physics and feature helpers`見出し直後に
  code cell markerがなく、`run_particle_filter`を含む定義群がmarkdown cellへ変換されたこと。
- `# %%`を1行追加し、markdown cell内に非comment Python行がないことを専用回帰testへ追加した。
  科学条件、seed、input、model、blend、runtimeは変更していない。
- version 2 push前はdedicated tests `11 passed`、Jupytext round-trip、py_compile、Ruff、
  strict experiment validationがPASS。生成notebookでも`run_particle_filter`がcode cellであることを
  readbackした。
- version 2 embedded config / source / bootstrap済みNotebook / metadata SHA:
  `d8dde96a97873ffd6b3c6ea85f4989199e09851c2903803c87cb8a6f46e0be1b` /
  `d5197c77b92dc6d365164053110647ba31c28531551d7e5e5a65411e0cd5e60d` /
  `af155afba6e6f1417811b1655c5d297a4dc99548b7e3b3adb7c73cf7c4b0129f` /
  `b852bbd3d5cdc442129adc2c4b8f4d0edb093b27a73a9bf50d27f527c3bf9681`。

### Kaggle current-test version 2完了

- kernel `kentookumura/exp510-exp413-exact-public-preoverride-inference` version 2
  （id_no `129634723`）は`KernelWorkerStatus.COMPLETE`。
- inference内部runtimeは`151.53196215629578 sec`、log最大timestampは`177.618570932 sec`。
- 14,151 rows / 3 wells、保存Pipeline-B booster 3、新規model/booster/GPU/親再学習0。
- technical gateはPASS。fallback / duplicate / nonfiniteは`0 / 0 / 0`、public/final式の
  max absolute parityは`0.0 / 0.0`。
- feature rows/count/zero-fill countは`14,151 / 196 / 39`。feature content/schema SHA:
  `d46fa8cfc1b0ba64b30e2a818bd8abcd4daf316f020fd1b0e677b10eb3c4ddb8` /
  `2702f0fed5a1ee9663e2ef81e15c6744a337cdb3609e70b73212a72f1b602abb`。
- likelihood-PF content/schema SHA:
  `a089855ba7ca1a126673b8cbd152f350fa806d406e6e37317eb84f38924c1cb8` /
  `278ebaa86b6e3d074a4c32aa90f371ed76fac23efc3aadd8bd0883a8dc579879`。
- component prediction content SHA:
  - SP45: `cf9ef1d693fdc12cce76b1665af91c4751941897a8402e110a285341725414f4`
  - Pipeline B: `f47676a254769adffc9baaf3e8fc7cfaf4c9679eb7f00edd3fec3238dd6bf8af`
  - public preoverride: `ab6fc2008202b0a88829d6ec854f9a6f1ea2c1290dba729be396070a4bdaf642`
  - exp413: `2346abc3458a126115ff2c8c4e7f8716dae2a17f8992ca6977c9455100af91fb`
  - final: `ea61118d3cc9cd41cb1519c368f7ec9ccece0c048694aad41d731922c369e9d1`
- Kaggle生成file SHA:
  - `sp45_projection_submission.csv`: `9b134d9619727f9484d51b6c89003bfc674edf530078b499ef36aec0545c9ca4`
  - `submission_B.csv`: `cbb6d23842bd1492ca762ca326d0837dd406cabec97367bc74ce9e476b8c99e2`
  - `public_preoverride_submission.csv`: `a849e23f8e851a5b4e188fd94006a5b1db747c54ace28e494f867471c54e16b4`
  - `submission.csv`: `7209a4bd89665f795bd4223fff5f4d9026f022e0e694465088749e8651e44e52`
- Kaggle file listingでsubmissionと全readout/manifestの存在を確認した。output archiveは取得せず、
  submit-checkとcompetition submitも行っていない。
- 初回runだけではrerun SHA一致がないため、`deterministic_anchor=false`を維持する。
- 実行完了後はlocal `inference_enabled=false`、sealed package `run_on_push=false`へ戻した。
  sealed embedded config / source / Notebook / metadata SHA:
  `5aeabd1f01d2b5696d014c6896dabf5a2dc0c508a53a6485c9ee50fdf7dbb555` /
  `d5197c77b92dc6d365164053110647ba31c28531551d7e5e5a65411e0cd5e60d` /
  `d08d1a4b2482416a971e68cceeb95f90daeedf90daa4dfcb00827d7bd821b411` /
  `2ba9fd3c3681d26f1be541651f99943959ba1ffba11ea42ae56f97490ed1693c`。

## 2026-08-04 output取得・submit-check・提出承認

- ユーザーの「提出してください」により、version 2 output取得、submit-check、code submission、
  scoring監視、LB/提出履歴記録を承認済みとした。
- Kaggle outputを一時ディレクトリ`/tmp/exp510-submit-v2.STK3as`へ取得した。
- skill checkerとrepo checkerはいずれもPASS。FAIL/WARNは`0 / 0`。
- `submission.csv`は14,151 rows、`id,tvt`、unique ID 14,151、sample ID順完全一致、
  duplicate / missing / nonfinite `0 / 0 / 0`。
- downloaded submission SHAはruntime manifestと一致:
  `7209a4bd89665f795bd4223fff5f4d9026f022e0e694465088749e8651e44e52`。
- downloaded manifest / log SHA:
  `b9cce40f61c22f6818e01ea2cb79e951ad8fabd2b610cc55d255a13816888905` /
  `e824166cb39199897d45793c21e1d552bcacf8deacbca40ac5f957769bfe3544`。

### code submission作成

- command: `kaggle competitions submit rogii-wellbore-geology-prediction -k
  kentookumura/exp510-exp413-exact-public-preoverride-inference -v 2 -f submission.csv`。
- message: `exp510 exact public preoverride 10pct hedge v2`。
- ref `55225634`、submitted at `2026-08-04 01:24:00.857000 UTC`、初期status `PENDING`。
- monitor scriptはsystem Python 3.8で`datetime.UTC` import errorとなったため、repo `.venv`
  Python 3.11で再起動した。submission自体への影響はない。
- monitor log: `logs/submission_exp510_exp413_exact_public_preoverride_hedge.log`。

## 2026-08-04 code submission ref 55225634 hidden rerun失敗

- ref `55225634`はCLI上`COMPLETE`だがPublic/Private Scoreは空欄。monitorでは提出から
  151分後の`2026-08-04 03:55:57 UTC`にcompleteを観測した。
- raw APIのscriptVersionIdは`340025138`。`errorDescription`は
  `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`。
- Kaggleはhidden tracebackを公開しないため、例外行の直接確認はできない。
- 静的監査では、public componentのSP45/Pipeline Bはhidden sampleから動的再生成する一方、
  exp413だけはkernel sourceに保存された公開test固定
  `exp413_current_test_predictions.csv.gz`をSHA一致で読み込んでいた。
- `load_exp413_component()`はこの静的artifactのID集合とhidden
  `sample_submission.csv`のID集合を完全一致させる。hidden datasetの行数/IDが公開testと異なると
  `RuntimeError("exp413 ID mismatch: ...")`で停止するため、これを高確度の原因と診断する。
- kernel sourceをmountしても上流exp413 notebookがhidden sample上で再実行されるわけではなく、
  公開commit outputは静的入力のままである。visible current-test technical PASSとsubmit-checkは、
  このhidden互換性を検証していなかった。
- 必要な修正はexp413 version 4のhidden-safe current-test生成をexp510内へ持ち込み、動的sample上で
  exp413 predictionを再生成してから固定`0.90 / 0.10` blendを適用すること。修正・再push・再提出は
  ユーザー承認前には行わない。

## 2026-08-04 hidden-safe修正・再実行承認

- ユーザーの「まず修正・実行してください。提出はまだです」により、exp510内のhidden-safe修正、
  同じKaggle kernelへのCPU/private/internet-off push/run、完了監視と生成物検証までを承認済みとした。
- competition submit、code submission再実行、正規`*_inference.ipynb`採用は承認範囲外。
- 科学条件は変更しない。public component `0.55 / 0.45`、final `0.90 / 0.10`、Gold/contact/same-well
  overlay禁止を維持し、公開test固定exp413 sidecarだけを動的exp413 v4生成へ置換する。

### 修正内容

- exp413 version 4 source SHA
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388`をguardした生成script
  `scripts/prepare_exp510_hidden_safe_runtime.py`を追加した。
- 生成helper `exp510_exp413_hidden_safe_runtime.py`はparent sourceをimport-safeな関数へ機械変換し、
  `exp413_runtime/`の固定config/settings/input bundleと11 kernel sourcesからcurrent sample上の
  12候補・21 confidence・exp218 clean273を再生成する。
- 保存済みparent selector 40、signed selector 20、TVT LightGBM 15をSHA検証して適用する。
  public Pipeline-B 3本と合わせてsaved booster読込は78、新規model/config/fold/booster、親再学習、
  inference-time training、GPUは`0 / 0 / 0 / 0 / 0 / 0 / 0`。
- `exp413_current_test_predictions.csv.gz`の公開commit SHA探索を削除した。exp413 runtime出力を
  dynamic sample IDへ完全一致検証し、その場でfixed blendへ渡す。
- visible reference decompressed SHA `875a1334...dc4`は固定入力guardではなく、公開run parityの
  report-only比較に変更した。hiddenではsampleに応じて異なるSHAを許容する。
- current-test inputsは`fleongg/rogii-claude-models-pub` version 1とexp413 v4の11 upstream
  kernel sources。`kentookumura/exp413-scale5-likpf-current-test-inference`の公開output mountは削除した。

### 修正後実行inventory

| scientific variant | new model config | fold | new booster | saved booster | inference training | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 78 | 0 | 0 | 0 |

- exp413 replacement likelihood-PFはtest wellあたり500 particles × 128 stable seeds。
- public componentはtest wellあたりPF 51 runs、beam 21 paths。合計well数と総run数はsampleから動的。
- 修正時点でcompetition submit authorizationはfalse。Kaggle run完了後も提出せず停止する。

### Kaggle version 3 実行と追加修正

- Kaggle kernel version 3は`COMPLETE`、runtime `385.163 sec`、`technical_gate_pass`だった。
- 14,151行・3 well、duplicate/nonfinite/fallbackはすべて0、public/final formula parityは0.0、
  external submissionはfalseだった。
- 動的exp413のdecompressed SHAはvisible reference `875a1334...dc4`と一致した。一方、version 3は
  parentが返したin-memory float32 `pred_tvt`を直接blendしたため、従来のCSV component boundaryとの
  差が最大`4.359375e-4`生じ、final SHAがversion 2と一致しなかった。
- exp510のexact契約を維持するため、動的に生成した`exp413_current_test_predictions.csv.gz`を
  その場で読み戻し、ID・finite・roundtrip driftをfail-closed検証してからblendするよう追加修正した。
- この修正はhidden sample対応を維持し、parent/public componentやblend weightを変更しない。
  competition submitは引き続き未承認のため実施しない。

### Kaggle version 4 最終実行

- 同一kernel `kentookumura/exp510-exp413-exact-public-preoverride-inference` version 4をCPU / private /
  internet offで実行し、`385.107887 sec`で`COMPLETE`、`technical_gate_pass`。
- exp413 runtimeは`290.261 sec`。14,151 rows / 3 wells、candidate 12、saved selector 40、
  signed selector 20、TVT model 15、booster training 0。
- generated exp413 decompressed SHAは`875a1334...dc4`、visible referenceと一致。
  CSV serialization roundtrip最大差は`4.84375e-4`で、1e-3 guard内。
- fallback / duplicate / nonfiniteは`0 / 0 / 0`、public/final formula parityは`0.0 / 0.0`。
- component content SHAはexp413 `2346abc3...1fb`、public `ab6fc200...642`、
  final `ea61118d...e9d1`。`submission.csv` SHAは`7209a4bd...4e52`でversion 2と完全一致。
- output archive取得後のsubmission形式検証はPASS。manifest SHA `ae7114e6...92c5`、
  downloaded log SHA `76324f57...988f`。
- `external_submission_performed=false`。ユーザー指示どおりcompetition submitは行っていない。

## 2026-08-04 hidden-safe version 4提出

- ユーザーの「提出に進んでください」によりversion 4のcompetition submissionを承認済みとした。
- Kaggle remote kernel `kentookumura/exp510-exp413-exact-public-preoverride-inference` version 4は
  `COMPLETE`、CPU / private / internet off、id_no `129634723`を再確認した。
- skill checkerとrepository checkerはいずれもPASS。14,151 rows / `id,tvt`、duplicate / missing /
  nonfinite 0、sample header・row count一致、FAIL/WARN `0 / 0`。
- submission SHA `7209a4bd89665f795bd4223fff5f4d9026f022e0e694465088749e8651e44e52`、
  manifest technical gate `PASS`を再確認した。
- submit commandはkernel version `4`、output `submission.csv`、message
  `exp510 hidden-safe exact public preoverride 10pct v4`。
- ref `55231514`、submitted at `2026-08-04 06:16:40.563000 UTC`、初期status `PENDING`。
- monitor logは`logs/submission_exp510_exp413_exact_public_preoverride_hedge.log`。hidden rerun・scoring完了後に
  Public LBと最終状態を追記する。

## 2026-08-05 version 4 scoring完了・実装再監査

- ref `55231514`はmonitor上`2026-08-04 16:48:01 UTC`に`COMPLETE`を観測した。submitted at
  `2026-08-04 06:16:40.563000 UTC`から627分、Public LB `7.201`、Private LB空欄。
- Kaggle UI表示は`Your latest submission scored 7.201, matching your best.`。exp413 ref
  `55080377`もCLI/APIの公開値は`7.201`で、公開3桁では同値。
- Kaggle CLI 2.2.3のtable/JSON API responseはいずれも`publicScore: "7.201"`までしか返さない。
  UIの`matching your best`がfull-precision RMSEの完全一致を比較しているか、leaderboard表示値を比較して
  いるかは公開情報から判別できないため、`full_precision_exact_match=unknown`として記録する。
- remote kernelを再pullし、提出notebookのsupport zipから展開したcompact inference source SHA
  `f3f699e9...81d1`とhidden-safe runtime SHA `0eea5b11...`がローカルversion 4と一致した。
- notebookは26 cellsで、最終cellが`run_inference()`を同期実行し、その後にcellはない。parent exp413
  runtimeが途中で`/kaggle/working/submission.csv`を書くが、return後にexp510がpublic componentと
  exp413をID/order完全一致でblendし、`final.to_csv(/kaggle/working/submission.csv)`で上書きする。
- final式は全行float64の`0.90 * exp413 + 0.10 * (0.55 * SP45 + 0.45 * Pipeline B)`。式計算後の
  fallback、early return、subprocess/background write、後続上書きはなく、parity failure時はfail-closeする。
- visible version 4 outputを再計算し、finalはexp413と同一ではない。差分MAE `555.138482 ft`、RMSE
  `912.373771 ft`、最大絶対差`3125.435635 ft`、100 ft超8,564行、1,000 ft超3,020行。
  visibleはhidden scoring rowsの証拠ではないが、payload・式・出力pathの実装回帰を検出する証拠である。
- hidden rerunは完走してscoreが付いたため、hidden sample上でfail-close guardを通過したことは確認できる。
  hidden `submission.csv`自体はKaggleから取得できず、hidden component差分やsubmission SHAは未確認。
- 結論: 技術実装は再監査PASS。Public hedgeはexp413比で公開3桁の改善を出さず、科学的仮説は支持されない。
  exp413をML/overall anchorとして維持し、exp510はnegative/neutral LB evidenceとして完了扱いにする。
