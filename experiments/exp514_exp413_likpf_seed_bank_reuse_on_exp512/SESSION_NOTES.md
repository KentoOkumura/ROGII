# exp514_exp413_likpf_seed_bank_reuse_on_exp512 セッションノート

## 目的

exp413 stable-seed likelihood-PF bankをSP45とexp413で共有し、SP45の重複128-seed PFを除去する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage A PASS、Stage B v2 scientific FAIL、Stage D v3 hidden rerun ERROR、Stage D v4 visible technical PASS
- 親: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- CV / LB / submission: なし / scoreなし / ref `55266559` ERROR
- 実装 / Kaggle package / run / output取得: `1 / 1 / 1 / 1`
- 正規train / inference Notebook: 非実行placeholder

## 2026-08-05 Stage D version 4 OOM修正・visible実行

- ユーザーの`実行してください`をStage D v4 visible technical validationのpackage/run/output確認承認として
  記録した。competition submission / hidden再提出は承認されておらず、実施しない。
- 実行量はscientific variant 1、runtime Ridge 1 config x 5 fits、保存済み83 model files / 103 estimators。
  LightGBM training config / trained fold / new booster / parent-control retrainingは`0 / 0 / 0 / 0`。
  Goldは最大4 process、shared PF/SP45は最大4 threadで実行する。

- ユーザー指示により、Stage D v3の数値契約を変えずにmemory lifetimeだけを修正した。Ridge出力`sub_1`
  確定後、train feature、OOF、保存trainer、Ridge trainer、予測matrix、冗長test aliasを明示解放し、
  `gc.collect()`とLinux `malloc_trim(0)`をbest-effort実行する。
- shared likelihood-PFとSP45 Beam/selectorを同じwell workerへ統合した。4-thread並列は維持し、各wellの
  SP45処理直後に`sp45_full`、`row_index`、`evaluation_index`、`known_mask`を除去する。全well完了後に
  保持するのはexp413用`id / likpf_scale_5 / likpf_mean` float32とmanifest/ledgerだけで、exp413 adapter
  消費時にcompact frameもrecordから除去する。full payloadの同時保持上限はeffective worker数の4 wells。
- Ridge用一時frameを`id/well/md_since/pred`へ縮小した。SP45 deterministic frameはdeep copyせずHJYACTへ
  in-place所有権移譲し、candidate reuse SHA確定後は`globals().pop`でexp413へ唯一所有権を渡す。
  exp413入口、exp145、exp218のcaller-side全行copyと、不要なprediction frame returnを除去した。
- 末尾のStage D v2 visible 5-SHA witnessはvisible sample ID SHA一致時だけfail-closeし、hidden dynamicでは
  `SKIPPED_HIDDEN_DYNAMIC`を記録する。これにより、hidden出力をvisible固定SHAへ比較する既知の必敗経路を除去した。
- generator/source/notebook SHAは`30a0500c...047f / ff668d88...542d / 15d4ebeb...1316`。
  sourceは8,640行、Notebookは55 cells。構文、Ruff F821、Jupytext round-trip、19 contract tests、
  strict experiment validationをPASSした。
- Kaggle packageは`--no-src`で再生成した。56 cells / 870,908 bytes、Notebook SHA
  `4d3696b8...6018`、normalized source-list SHA `49f120da...d5eb`。bootstrapはv3と同じ26 files /
  978,989 uncompressed bytesで、private T4 / internet off / 6 datasets / 11 kernelsを確認した。
- `2026-08-05 11:38:55 UTC`に同じcanonical kernelへversion 4をpushし、`RUNNING`を確認した。
  remote id_noは`129770672`。Kaggleがcell sourceをlistからstringへ正規化した後の56/56 sourceは完全一致し、
  remote source-list SHAも`49f120da...d5eb`。hidden code submission / 再提出は実施しない。
- version 4は`COMPLETE`。3 wells / 14,151 rowsをbootstrap込み`879.386897秒`で処理し、v2固定の
  Gold / HJYACT / exp413 / component readout / final submissionの5 SHAはすべて完全一致した。
  final submission SHAは`9974c3fa...e192ad`。sampleのheader / 14,151 rows / ID順に完全一致し、
  duplicate / NaN / Infは0でsubmit-check `PASS`。外部competition submissionは実施していない。
- runtime report、submission、logを`kaggle/output/stage_d_v4/`へ取得した。report / metrics / log SHAは
  `76b548f1...25c5f / 16157b80...1a424 / ff67f620...a606`。
- Ridge解放ではparent current RSSが`12,979.207→791.668 MiB`、exp413前は`953.895→865.621 MiB`、
  shared bank終了時は`1,359.234→1,037.676 MiB`へ低下した。raw 128-seed bankは保持せず、visibleでは
  full payload最大3 wells、hidden設定上限は4 wells。全well full payload保持はfalseだった。
- parent process peak RSSは`25,123.293 MiB`でv3から`1.309 MiB`低いだけである。peakはRidge解放前の
  高水位を記録し、Gold child process peakも含まないため、visible完走はhidden OOM不発の保証ではない。
- 200-well工程別外挿は`6.289658〜8.057147時間`で上限9時間未満の
  `estimated_pass_not_hidden_runtime_guarantee`。v3よりvisibleで`39.690306秒`（`4.727%`）遅いが、
  上限推定は約`5.99分`増に留まる。Stage B scientific FAILは変わらず、hidden再提出は実施していない。

## 2026-08-05 Stage D version 3 code submission hidden rerun ERROR

- ユーザー実施のcode submission ref `55266559`をKaggle APIで確認した。submitted atは
  `2026-08-05 10:05:07.813 UTC`、kernelは`kentookumura/exp514-shared-likpf-stage-d-visible`
  version 3、scriptVersionId `340328874`。
- API statusは`COMPLETE`だがPublic/Private Scoreは空。`errorDescription`はhidden datasetでNotebookが
  unhandled errorになったという一般表示で、hidden tracebackや最初の例外行はAPIから取得できない。
- 静的監査で、Stage D v3末尾の`_stage_d_v2_equivalence_targets` 5件がvisible sample判定でguardされず、
  hiddenでもvisible v2固定SHAとのexact一致を無条件要求することを確認した。hidden出力はvisible SHAと異なるため、
  このblockへ到達すれば必ず`RuntimeError("Stage D v3 runtime-only output parity failed ...")`になる。
- したがって、以前の`submission-ready`判定は誤りだった。visible HJYACT/exp413 reference guardはsample ID SHAで
  分岐していたが、後付けしたv2 output-equivalence guardに同じhidden分岐が欠け、contract testも
  hidden sampleでこのblockを通す検査を持っていなかった。
- ただしKaggle APIからhidden tracebackを取得できないため、今回実際に最初に発生した例外がこのguardか、
  それより前のOOM/別例外かは未確定。visible v3 parent peak RSSは`25,124.602 MiB`でchild processを含まず、
  Gold最大4-processのhidden 200-well memoryリスクも残る。
- Stage B scientific gateは既にFAILしているため、guardだけを外して再提出することは事前規約上の採用判断を
  変えない。Codexは提出・修正・再提出を行っていない。

## 2026-08-05 Stage B version 2完了・scientific FAIL

- canonical kernel `kentookumura/exp514-shared-likpf-fixed32-stage-b` version 2 / id_no
  `129762632`は`KernelWorkerStatus.COMPLETE`。Kaggle reportの本体runtimeは`4,845.475189秒`、
  report出力はlog `4,865.784秒`、32 wells / 129,906 rowsを評価した。
- v2のprediction freeze content SHAはv1と同じ`62c78373...8a280`。truth/foldはfreeze後にjoinされ、
  repaired source SHA `a510d17b...e0748`もremote outputと一致した。よってv2は採点修正だけで、
  v1から科学predictionを変更していない。
- primary pooled RMSEはlegacy control `9.010759361`、shared-bank candidate `9.060439859`、
  candidate-control `+0.049680497 ft`で、上限`+0.02 ft`をFAILした。
- fold gateはnonworse `2/5`。deltaはfold 0--4で`-0.092700 / +0.267956 / +0.081856 /
  +0.062790 / -0.032173 ft`となり、必要な4/5を満たさない。
- fixed scopeは`raw_gr_observed`が`+0.060618 ft`で上限`+0.05 ft`を超えた。その他4 scopeは
  上限内、hidden-like 2面はともに`-0.117665 ft`改善した。
- by-well delta p95は`+0.647871 ft`で上限`+0.25 ft`をFAIL。worst wellは`+1.192164 ft`で
  上限`+5.0 ft`をPASSし、fold/scope nonemptyもPASSした。全7 gateのうち精度4条件がFAIL、
  技術3条件がPASSで、`all_and_gate_passed=false`。
- report / metrics SHAはともに`906fcdaf...7ba0`、execution log SHAは`005f6bb4...f742`。
  必要な3ファイルだけを`kaggle/output/stage_b_v2/`へ保存した。submission生成・外部submitは0。
- これは実装ERRORではなく、SP45 legacy stochastic bankをexp413 stable-seed bankへ置換する科学仮説の
  FAIL。事前規約どおりscale、seed、selector、well subsetで救済せずexp514を終端する。Stage D v3の
  runtime/readiness PASSはStage Bの精度FAILを代替しない。後にユーザーがStage Dをcode submissionしたが、
  hidden rerun ERRORとなりscoreは付かなかった。

## 2026-08-05 Stage D version 2完了と正式runtime評価

- canonical kernel version 2は`COMPLETE`。3 wells / 14,151行を`929.929790秒`で処理し、
  HJYACT、Gold、exp413、最終50/50 submissionまで生成した。外部competition submitは0。
- 工程時間はshared likelihood-PF `39.7344秒`、SP45 after shared PF `1.085秒`、
  HJYACT learned total `69.610950秒`、Gold逐次`123.116097秒`、exp413 after shared PF
  `274.684秒`、固定overhead `421.699344秒`。parent process peak RSSは`25,124.188 MiB`。
- 固定式で200 wellsへ外挿したlower/upperは`8.068150 / 9.528814時間`。upperが9時間を超えるため
  `estimated_fail`。visible 3 wellsからの高不確実性推定で、hidden runtime実測ではない。
- v3同値性witnessとしてGold balanced `2b86386f...5a815`、HJYACT `6b3e1c57...37b3`、
  exp413 `04e6da90...5908`、component readout `c3a9b217...e1fd`、最終submission
  `9974c3fa...ad`を固定した。

## 2026-08-05 Stage D version 3 runtime-only最適化

- ユーザー指示により、Goldをwell単位`joblib` multiprocessing、最大4 processへ変更した。
  worker内BLAS/OpenMP/cKDTree threadは1に制限し、親processが固定well順で結果をmergeする。
- SP45が生成済みの決定論test feature frameと`FI`/`DI` imputerをHJYACTへ再利用する。
  HJYACT固有の`pf_ancc`、`pf_z`とその依存列だけを再生成し、learned x1.3 PF、Gold PF、
  shared likelihood-PF、model、selector、weight、profile、最終式は変更しない。
- Stage A generator/source SHAは固定したまま、Stage B v2 sourceも変更していないため、再実行はStage Dだけ。
- Stage D v2の5出力SHAをv3でfail-close完全一致検査する。構文、Ruff F821、Jupytext、
  16 contract tests、strict validationは全てPASS。
- v3 sourceは8,404行 / 55 cells、packageは56 cells / 858,227 bytes、SHA
  `bdeb3f78...4c2c`。support bundleは26 files / 978,989 bytes、private T4 / internet off。
- `2026-08-05 09:22:07 UTC`に同じcanonical kernelへversion 3をpushし、`RUNNING`を確認した。
  readbackした56/56 cell sourceはlocalと完全一致し、source-list SHAは`52b00894...ed29f`、
  id_noは`129770672`。Stage B v2には変更を加えていない。

### Stage D version 3完了

- version 3は`COMPLETE`。3 wells / 14,151行を`839.696591秒`で処理し、v2比`90.233199秒`
  （`9.703%`）短縮した。parent process peak RSSは`25,124.602 MiB`でv2とほぼ同じ。
- Goldはrequested 4 / effective 3 process、inner BLAS 1 thread、cKDTree worker 1、入力well順mergeで
  `77.080117秒`。v2逐次`123.116097秒`から`46.035980秒`（`37.392%`）短縮した。
- HJYACTはSP45の決定論176列と同じimputer instanceを再利用し、full `build_features` callは0。
  HJYACT固有PF refreshは`1.844313秒`、totalは`38.367232秒`で、v2比`44.883%`短縮した。
- Gold balanced、HJYACT、exp413、component readout、最終submissionの5 SHAはv2と完全一致。
  最終SHAは`9974c3fa...ad`。submission checkは14,151行、`id,tvt`、sample ID順完全一致、
  duplicate/NaN/Inf 0で`PASS`。外部competition submitは行っていない。
- 200-well lower/upper推定は`6.174531 / 7.957332時間`となり、9時間上限に対して
  `estimated_pass_not_hidden_runtime_guarantee`。v2 upper `9.528814時間`から`1.571482時間`改善した。
  hidden 200 wellsは未実測であり、visible 3 wellsとの差、child process peak RSS、競合の不確実性はhigh。

## 2026-08-05 Stage D version 1完了確認

- Kaggle最終状態は`ERROR`。推論時間超過ではなく、log `783.969秒`のvisible-only parent SHA
  guardで停止した。Kaggle output file listingは空で、runtime reportは生成されていない。
- shared PFは`49.020秒`、SP45 Beam/selectorは`1.259秒`、learned HJYACTはlog上約`84秒`、
  Gold visible-prefixは`140.638秒`まで実行済み。3 wells / 14,151行を処理した。
- HJYACT componentはID順一致・finiteで、SHAは`6b3e1c57...37b3`。親exp512の固定SHA
  `b192d3f3...ed4a`とのexact一致を要求したため停止した。
- exp514はlegacy SP45 bankを科学的に置換する候補なので、親最終出力のexact一致要求は不適切な
  継承guardである。exp413 component、最終50/50、Stage D runtime reportには未到達。
- よってStage D v1はsubmission-ready PASSでも9時間runtime FAILでもない。修正・再実行は未承認。

## 2026-08-05 Stage D v1からの200-well暫定評価とv2修正

- ユーザー指示により、修正前にv1ログから200 wellsを暫定評価した。実測済み4工程と固定overheadに、
  未実行exp413工程は親exp512 v7の`297.388秒`をproxyとして補った。
- shared PF短縮分`49.020秒`をproxyから差し引く推定は`8.448〜9.831時間`、差し引かない保守推定は
  `9.129〜10.739時間`。上限で判定するため現時点は`estimated_fail`。
- これは正式Stage D reportではなく、不確実性highの暫定値。exp413実測を得るv2で置き換える。
- v1 HJYACT component SHA `6b3e1c57...37b3`をexp514候補のvisible witnessとして固定し、
  親exp512 SHA一致は参考記録だけに変更した。予測、model、PF/Beam、blend、runtime式は不変。
- v2実行は1 variant、runtime Ridge 1 config × 5 fits、保存model 83 files / 103 estimators、
  新規LightGBM config/fold/boosterと親/control再学習は全て0。
- v2 packageは56 cells / 840,045 bytes、SHA `dcad740c...7f9c4`、source-list SHA
  `88dc3319...47a9b`。private T4 / internet off / run-on-pushと入力source数はv1から不変。
- `2026-08-05 08:45:43 UTC`に同じ`kentookumura/exp514-shared-likpf-stage-d-visible`へ
  version 2をpushし、`RUNNING`を確認した。remote/local source-list SHAは一致、id_noは`129770672`。

## 2026-08-05 Stage B v2修正・再実行承認

- ユーザーの`Stage Bを修正して再実行してください`を、Stage B評価バグの修正、同じcanonical
  kernelへのversion 2 push、必要output取得の承認として記録した。
- v1のERROR原因は、pre-branch列を既存post-branch列名へrenameし、pandas上で同名列が2本に
  なったこと。`metric_bundle`だけをcopy + 1次元配列の明示代入へ変更した。
- 32 wells、well selection SHA、legacy/shared各32 bank、PF/Beam/selector/branch hedge、全精度閾値は
  変更しない。2 variants、LightGBM config 0、fold 0、booster 0、親/control再学習0、合計64 bank。
- generator/source/notebook SHAはそれぞれ`adf01165...214eb7`、`a510d17b...e0748`、
  `4a361ec1...ca261`。重複列を作らないcontract testを追加した。
- v2 packageは15 cells / 159,310 bytes、SHA `63f47433...dc82`。Stage B sourceとhidden-like
  assignmentの2 files / 121,377 bytesだけをbootstrapし、Stage D source等を含めない。
- `2026-08-05 08:31:05 UTC`に同じ`kentookumura/exp514-shared-likpf-fixed32-stage-b`へ
  version 2をpushし、`RUNNING`を確認した。remote/local 15 cellsのsource-list SHAは
  `c27f3173...e86a6`で一致。kernel id_noはversion 1と同じ`129762632`。

## 2026-08-05 設計記録

- `kaggle-review-exp`に従い、実験実装前にsteeringを作成した。
- `docs/06_reproducibility.md`を読み、well別stable seed、thread schedule独立性、raw-test regeneration、
  content SHA、rerunを設計へ反映した。
- ユーザー指定の5手順をproducer/consumer DAGとして固定した。
- exp413 scale5はexact parity、SP45はscientific variantとしてpaired精度gateを必須とした。
- learned x1.3、Gold masked-prefix、`pf_ancc`、`pf_z`、Beamを共有禁止とした。
- 親SP45のvisible physical overrideは変更対象外だが、positive evidenceから除外した。
- fixed32 technical、fixed200 paired accuracy、200-well end-to-end runtime、hidden readinessの4段階に分けた。
- visible 3-well runtimeを9時間判定に使用しない。
- 200-well runtimeはobserved `<=8.5h`、bootstrap p95 `<=9.0h`を固定gateとした。

## source identity（設計時）

- exp512 compact inference:
  `16982879716918811dfa9915c4862d45836bd9360efafbaee41046c3e1b6240f`
- exp512 config:
  `951a665b209880267cd0a8d603006f1ffcd9270fdd4e00348a66b212195e8d15`
- exp073 replay source:
  `4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e`
- exp413 config:
  `d12e6d74a7f567f0873d5513883b3a7d36d0cd5be5231037e7db12f1a74036a7`

実装開始時に再計算し、driftがあれば暗黙更新せず停止する。

## 実行量契約

- active scientific variant: 1
- LightGBM config / trained fold / new booster: `0 / 0 / 0`
- parent/control retraining: 0
- inference-time booster training: 0
- candidate shared likelihood-PF: `1 bank / eligible well`
- legacy SP45 likelihood-PF: `0 bank / well`
- exp413 duplicate likelihood-PF: `0 bank / well`
- fixed32 Stage A、fixed32 Stage B、200-well Stage C、hidden Stage Dはそれぞれ別承認とする。

## 再現性メモ

- seed policy: `SHA256(feature_family::split::well) + seed_index`
- shared stochastic component: exp413 x1.0 likelihood-PF、500 particles×128 seeds
- parallel policy: private per-well Numba seed state、well/row/scale merge order固定
- raw seed pathsはwell内でaggregate/branch summary生成後に解放する。
- aggregate、branch summary、generation ledger、model manifest、prediction、submission SHAを記録する。
- fixed32のthread `1/4` parityと同一package 2-runを必須にする。
- 完全なhidden stochastic pathが再現するまでdeterministic anchorとは呼ばない。

## 2026-08-05 Stage A実行承認

- ユーザーの`Stage Aを実行してください`により、Stage A fixed32専用Notebookのpackage、
  Kaggle実行、監視、report/metrics JSON取得を承認済みとして記録した。
- 実行対象は`exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.ipynb`だけとする。
- 実行量はactive scientific variant `1`、LightGBM config / trained fold / new booster
  `0 / 0 / 0`、parent/control retraining `0`、inference-time booster training `0`。
- PF実行量は`32 wells x 2 thread settings x 2 reruns = 128 well-bank generations`、
  各bankは`500 particles x 128 seeds`。正規推論、Stage B/C/D、submissionは実行しない。
- Kaggle kernel idは
  `kentookumura/exp514-exp413-likpf-seed-bank-reuse-on-exp512-stage-a-fixed32`、
  GPU T4 runtime（PF本体はCPU）・internet disabled・run-on-pushを使用する。
- Stage AではSHA/ledgerが後続gateの実証拠なので、output取得は
  `exp514_stage_a_fixed32_report.json`、`metrics.json`、監視用execution logに限定する。

### package記録

実行前検証:

- source identity 4件とgenerator/Stage A source/Notebook SHAは設計・実装時記録と一致した。
- `py_compile`、Ruff F821、Jupytext `--test`、exp514 + package tool tests `11 passed`、
  strict `validate-exp`がPASSした。
- Kaggle CLI `2.2.3`、OAuth credentialとlegacy credentialを確認した。credential実値は記録しない。

package command:

```bash
make prepare-kaggle-notebooks \
  EXP=exp514_exp413_likpf_seed_bank_reuse_on_exp512 \
  EXTRA_ARGS="--notebook stage_a_fixed32 --kernel-id kentookumura/exp514-exp413-likpf-seed-bank-reuse-on-exp512-stage-a-fixed32 --title 'exp514 exp413 likpf seed bank reuse on exp512 stage a fixed32' --run-on-push --strict"
```

- package Notebook SHA256: `42604be8...9c8629`（bootstrap込み8 cells）
- metadata SHA256: `994f7f37...1d285c`
- metadata: private、T4、internet disabled、run-on-push、competition sourceのみ。
- 正のStage A Notebook SHA256 `9916807b...81fa9`はpackage前後で不変。

### 初回push 400とcanonical slug短縮

- 最初のkernel id/titleはslug一致していたが61文字で、`SaveKernel`が詳細なし400を返した。
- 同じidを`kaggle kernels pull -m`すると403、mineの`exp514`検索は`Not found`だった。
  したがって初回要求によるremote kernel作成はない。
- Kaggleのslug長制約を原因候補と判断し、別の科学実験名にはせず、意味を保持した50文字未満の
  `kentookumura/exp514-shared-likpf-fixed32-stage-a` / `exp514 shared likpf fixed32 stage a`
  にcanonical名を短縮する。Notebook source、実行量、runtime、入力、gateは変更しない。
- 短縮後package Notebook SHA256は`87c97b4a...67436`、metadata SHA256は
  `8622bf4f...afe3f`。packageは700 KiBでKaggle source 1 MB制約内。
- `2026-08-05 05:34:38 UTC`に短縮後canonical kernelへversion 1をpushし、
  run-on-pushでStage Aを開始した。別slugのremote kernelは作成していない。
- remote id_noは`129757357`。readbackした8/8 cell sourceはlocal packageと完全一致し、
  source-list SHA256は`1cd49e0e...9a886`。remote metadataもprivate、T4、internet off、
  competition source 1件で一致した。push直後statusは`RUNNING`。

### Stage A version 1 完了

- Kernel `kentookumura/exp514-shared-likpf-fixed32-stage-a` version 1 / id_no `129757357`は
  `COMPLETE`。report内elapsedは`2,363.410299秒`、log上のreport出力は`2,378.627551秒`。
- fixed32選定SHAは`86157959...6ae58b`。32 unique wells、500 particles、128 seeds、
  thread `1/4`、各2 rerun、合計128 well-bank生成をreportで再確認した。
- well-bank実行時間はthread1が`892.045002 / 885.807420秒`、thread4が
  `282.509160 / 290.578054秒`。
- 4 runすべてで次のsignatureが完全一致した。
  - aggregate content: `68c5dc68...c8e74a`
  - branch summary: `904a3e00...6d0285`
  - generation ledger: `5a3a81f8...a42942`
- `truth_read=false`、all-AND `true`、new booster `0`、parent/control retraining `0`、
  submission file `0`、external submit `0`をJSON assertionで確認した。
- exp413 scale5は、独立した二重PF実行を追加せず、SHA固定したexp073 x1.0 sourceとのAST/source一致、
  adapter契約、generation ledger fail-closeを根拠にexact contractとした。
- reportとmetricsはbyte-identical、SHA256 `87387d9a...7b8612`。log SHA256は
  `49283a64...7efba9`。必要証拠3ファイルだけを`kaggle/output/stage_a_v1/`へ残した。
  Kaggle CLIが同時取得したbootstrap重複filesは削除せず
  `/tmp/exp514-stagea-v1-output-extras-20260805/`へ退避した。
- 結論: Stage A fixed32 technical / determinism gateはPASS。これはStage Bのlegacy/shared paired精度、
  Stage Cの200-well full runtime、hidden inference readiness、deterministic anchorのPASSを意味しない。
- 次は別承認がある場合だけStage B/Cへ進む。正規Notebook、hidden inference、submitは実行しない。

## 2026-08-05 Stage B fixed32設計変更

- ユーザーの`200wellsは時間がかかるので32wellsにしてください`を、直前に説明したStage Bの
  well数変更として記録した。Stage Cの200-well end-to-end runtime shadowは今回変更しない。
- Stage BはStage Aでtruthを読まずに固定した同じ32 wellsをselection SHA
  `86157959...6ae58b`ごと再利用し、別subsetの選び直しを禁止する。
- legacy SP45 control 32 bank + shared candidate 32 bank = 合計64 well-bank生成を予定する。
  各bankは500 particles × 128 seeds。LightGBM config / trained fold / new booster / 親再学習は
  `0 / 0 / 0 / 0`。
- pooled / fold / scope / by-well p95 / worstの全AND閾値は変更しない。ただし32 wellsのため、
  PASSしても200-well一般化の証明ではなく、小規模accuracy screeningと解釈する。
- Stage Bの実装・Kaggle package/run・output取得は未承認であり、今回は設計・記録だけを変更した。

## 未承認の将来作業

- 正規Notebook採用
- Stage A/B report/metrics以外のoutput取得
- Stage D visible-test package/run
- submit-check、competition submission、monitoring

## 次のアクション

1. Stage Bの監視を再開する場合だけ、完了結果を取得してpaired RMSEと全AND gateを確認する。
2. Stage Bの判断後、別承認があればStage D visible testを実行する。
3. Stage Cは実行せず、competition submissionへは別承認なしに進まない。

## 2026-08-05 Stage B fixed32実行承認と実装

- ユーザーの`StageBを実行してください`により、Stage B fixed32専用Notebookの実装、package、
  Kaggle実行、監視、report/metrics JSON取得を承認済みとして記録した。
- 実行対象は別名の
  `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_b_fixed32.ipynb`だけであり、
  正規inference Notebook、Stage C、hidden inference、submissionは対象外。
- Stage Aのtarget-free selection 32 wells / SHA `86157959...6ae58b`をそのまま再利用し、
  Stage Bの結果やtruthによるwell再選択を禁止した。
- 実行量はactive scientific variant `2`（legacy control / shared candidate）、
  legacy SP45 `32` bank + shared candidate `32` bank = 合計`64` well-bank生成、
  各bankは500 particles × 128 seeds。common Beamはwellごとに1回だけ生成して両variantで共有する。
- LightGBM config / trained fold / new booster / parent-control retraining / inference-time trainingは
  `0 / 0 / 0 / 0 / 0`。
- selector bin、scale、Beam blend、hold、profile、branch hedge
  `strength/min_mass/sep_low/sep_high/cap = 0.60/0.25/4/40/2 ft`を親exp512から固定した。
- raw horizontalのprediction側は`MD/Z/GR/TVT_input`だけを読み、control/candidateのprediction CSVを
  deterministic gzipで保存してdecompressed content SHAを固定した後にだけsuffix `TVT`と5 reporting foldをjoinする。
- primaryはbranch hedge適用後。pooled / 5 fold / 6 fixed scope / by-well p95 / worstを同じ固定閾値で
  all-AND判定し、hedge前も診断として保存・表示する。
- hidden-like scope用assignmentはSHA
  `5f9ac9fa...ca6597`でbootstrapし、predictionには使わずpost-freeze評価scopeにのみ使う。
- generator SHA `e604da04...83608`、Stage B sourceは1,702行 / SHA
  `6d09c42a...b1883`、Jupytext Notebookは14 cells / SHA `c3c91a8a...afa83`。
- 正規Notebookはplaceholderのまま変更していない。
- package前の構文、Ruff F821、Jupytext round-trip、exp514 contract `10 passed`、
  Kaggle package tests `4 passed`、strict `validate-exp`、template validationがPASSした。

### Stage B package記録

package command:

```bash
make prepare-kaggle-notebooks \
  EXP=exp514_exp413_likpf_seed_bank_reuse_on_exp512 \
  EXTRA_ARGS="--notebook stage_b_fixed32 --kernel-id kentookumura/exp514-shared-likpf-fixed32-stage-b --title 'exp514 shared likpf fixed32 stage b' --run-on-push --strict"
```

- package Notebookはbootstrap込み15 cells / 835,987 bytes、SHA256
  `f261b4d4...6f2a5`。metadata SHA256は`dd7ebfb3...3a047`。
- metadataはprivate、T4、internet disabled、run-on-push、competition source 1件。
- bootstrap 37 files内にhidden-like assignmentがあり、SHA
  `5f9ac9fa...ca6597`、Stage B承認=true、64 banks、training `0/0/0`をreadback確認した。
- 正のStage B Notebook SHA `c3c91a8a...afa83`はpackageで変更していない。

### Stage B version 1開始

- `2026-08-05 06:43:31 UTC`に
  `kentookumura/exp514-shared-likpf-fixed32-stage-b` version 1をpushし、run-on-pushで開始した。
- remote id_noは`129762632`。pullしたremote metadataはprivate、T4、internet off、
  competition source 1件でlocal packageと一致した。
- remote/localは15/15 cell sourceが完全一致し、source-list SHA256は
  `5c5dc1e0...7216f`。完了までは同じkernel idを監視し、logs空やstatus 500だけで再pushしない。

### Stage B version 1 ERROR

- ユーザーの`Stage Bが失敗しました`を受けて同じkernelのstatus/logsを取得した。
  statusは`KernelWorkerStatus.ERROR`。
- shared/legacy predictionは129,906行×16列を`5,081.673128秒`でtruth/fold join前にfreeze済み。
  content SHAは`62c78373...8a280`。実行ERRORは`5,083.549175秒`で、重い予測生成後の採点だけで発生した。
- `metric_bundle`がpre-branch列を既存`control_tvt` / `candidate_tvt`へrenameしたため同名列が2本ずつでき、
  `control`がshape `(129906, 2)`、targetが`(129906,)`となってbroadcast ERRORになった。
- これはscientific gateのFAILではなく実装ERRORで、Stage Bの精度結論は未評価。Kaggle files一覧は空で、
  frozen prediction CSVを再利用できる出力としては確認できなかった。
- ユーザーからStage B修正・再実行の指示は受けていないため、Stage B sourceは変更せず、再pushもしない。

## 2026-08-05 Stage D visible test実行承認と実装

- ユーザーの`先にStage Dを実行しておき実行時間を確認しておきたいです`により、Stage Dの別名
  submission-ready visible-test Notebook実装、package/run、必要output取得を承認済みとした。
- Stage B ERRORはpost-freeze採点bugで推論ロジックのscientific FAILではないため、Stage Dを止める根拠にはしない。
- 実行量はactive scientific variant `1`、LightGBM training config / trained fold / new booster /
  parent-control retraining `0 / 0 / 0 / 0`。保存model 83 files / 103 estimatorsを再利用し、
  source Ridgeだけ1 config × 5 runtime fitsを行う。
- shared likelihood-PFはdynamic wellごとに1 bank、legacy SP45 bankとexp413 duplicate bankは0。
  外部competition submissionは行わない。
- 正規inference Notebookはplaceholderのまま維持し、7,920行のSHA固定候補から別名
  `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_d_visible.py/.ipynb`を生成した。
- Stage D sourceは8,079行、Notebookは55 cells。工程別runtime、visible well/row数、parent process peak RSS、
  200-well lower/upper estimateをJSONへ保存する。推定はhidden runtime実測や完走保証ではない。

### Stage D package / version 1開始

- package Notebookはbootstrap込み56 cells / 839,181 bytes、SHA
  `057c3563...8ec6f`。metadata SHAは`b2fae395...84b1`。
- support bundleは26 files / uncompressed 978,989 bytes。Stage D sourceを重複格納せず、親exp512と同じ
  6 datasets / 11 kernels / 25 dependency mappingsと`project.yml`だけをbootstrapする。
- metadataはprivate、T4、internet off、run-on-push、competition source 1。外部submitは含まない。
- `2026-08-05 08:20:59 UTC`に`kentookumura/exp514-shared-likpf-stage-d-visible`
  version 1をpushした。remote id_noは`129770672`、statusは`RUNNING`。
- remote/localは56/56 cell sourceが完全一致し、source-list SHAは`8a9eda2a...ee4e6`。

## 2026-08-05 Stage C不要化とStage D runtime見積もり契約

- ユーザーの`Stage Dのvisible testに対する実行時間から見積もることとしてください。つまりStage Cは不要です。`
  を実験契約の変更指示として記録した。
- Stage C 200-well shadowは実装・package・実行しない。Stage C用source / Notebookは作成しておらず、
  既存作業は実装可否の監査だけだった。状態はPASSではなく`not_required_by_user_override`とする。
- Stage Dはsubmission-readyな同一コードをcurrent visible testで実行し、工程別秒数、visible well/row数、
  peak RSSを記録する。Stage D package/runはこの時点では未承認である。
- 4-way well並列工程はlower=`工程秒×200/4`、upper=`工程秒×200/visible wells`、逐次per-well工程は
  `工程秒×200/visible wells`とし、固定overhead / I/Oは1回だけ加える。
- estimated upperが`32,400秒`以下なら`estimated_pass_not_hidden_runtime_guarantee`、超過なら
  `estimated_fail`とする。visible 3 wellsとhidden 200 wellsでは長さ・欠損率・負荷が異なり得るため、
  見積もりの不確実性はhighと記録する。
- この方法はhidden 200-well runtimeの実測、9時間完走保証、precision/determinismのpositive evidenceではない。
- ユーザー指示によりStage Bの能動監視は停止したままとし、kernelのcancelや再pushは行っていない。

## 2026-08-05 実装記録

- ユーザーの`exp514を実装してください`を、設計済み範囲の実装承認として記録した。
- 実装開始前に設計時SHAを再計算し、次がすべて一致した。
  - exp512 compact inference: `169828797...1b6240f`
  - exp512 config: `951a665b...e8d15`
  - exp073 replay source: `4af212a8...a647e`
  - exp413 config: `d12e6d74...74036a7`
- `experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/prepare_exp514_shared_likpf_candidate.py`を追加した。親exp512 source SHAが一致しない場合は生成を停止する。
- exp073 replay sourceとAST一致する`stable_seed`、`_interp1`、`_pf_lik_allseeds`を別名でself-contained化した。
  x1.0 GR sigma、500 particles、128 seeds、scale 3/5/8/12、temperature-5 branch summaryを固定した。
- producerはraw test wellをSP45より先にwell別threadで処理し、raw `preds[128,n_eval]` / `liks[128]`を
  all-scale aggregateとbranch summary作成後に`del`する。process-wide bankにはaggregate、exp413 float32 frame、
  branch、SHA、ledgerだけを保持する。
- SP45 adapterはknown prefix exact + suffix aggregateのfull-length配列を読み、旧
  `run_pf_lik_ensemble_scales`とlast-known fallbackを実行経路から外した。
- exp413 adapterは`likpf_scale_5`とaudit用arithmetic `likpf_mean`だけを渡し、後段
  `replay_source.build_likpf`を外した。exp413内部でsemantic `likpf_mean`をscale5へ置換する親契約は維持した。
- ledgerはeligible wellごとにproducer/core/SP45/exp413 `1 / 1 / 1 / 1`、legacy SP45 / exp413 duplicate /
  fallback `0 / 0 / 0`以外をfail-closeする。
- learned x1.3 likelihood-PF、Gold masked-prefix PF、`pf_ancc`、`pf_z`、Beam、selector/hold/hedge、
  保存model、final 0.50/0.50式、model-package disabledは親のまま保持した。
- full inference候補:
  `exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py`、
  7,920行 / 53 cells、source SHA `8b1616dd...71634`、Notebook SHA `30a41bb3...e383f`。
- Stage A専用候補:
  `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py`、
  746行 / 7 cells、source SHA `89129ad8...b3873`、Notebook SHA `9916807b...1fa9`。
  raw属性だけで32 wellをstratified固定し、thread 1/4 × 2 runsのSHA/ledgerを出す。suffix truth、
  full exp512 inference、submissionは読まない。
- 親compactは7,236行 / 8 numbered sections、exp514 full候補は7,920行 / 9 numbered sections。
  親の全章を維持し、共有producer/adapter/Stage A章を追加した。正規Notebookはplaceholderのまま変更していない。

## 静的検証

実行済み:

```bash
uv run python experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/prepare_exp514_shared_likpf_candidate.py
uv run python -m py_compile experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/prepare_exp514_shared_likpf_candidate.py \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py
.venv/bin/ruff check --select F821 experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/prepare_exp514_shared_likpf_candidate.py \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py
.venv/bin/pytest -q experiments/exp514_exp413_likpf_seed_bank_reuse_on_exp512/tests/test_exp514_contract.py
make validate-template
make validate-exp EXP=exp514_exp413_likpf_seed_bank_reuse_on_exp512
.venv/bin/pytest
make update-summary
```

- dedicated contract testは7件PASSした。source SHA、exp073 kernel/interp/stable-seed AST一致、
  synthetic bankのthread parity、raw bank非保持、consumer各1回、duplicate/fallback除去、
  target-free Stage A、承認境界、source hashを確認した。
- ローカル`.venv`にNumbaがないため、実Numba kernelはローカル実行していない。kernel correctnessはAST一致、
  adapter/thread/ledger behaviorはdeterministic test coreで確認した。実kernelのthread 1/4・2-runは
  Kaggle Stage Aの未実行gateであり、PASS扱いしていない。
- Kaggle package/run、output取得、CV/LB、submissionは実行していない。
- repository全体の`.venv/bin/pytest`は`1,861 passed / 8 skipped / 4 failed`だった。
  4件は今回触れていないexp293の既存contract SHA 2件と、exp296の完了後status/run flag期待2件であり、
  exp514の失敗はない。exp514 dedicated 7件とstrict validationは再実行してPASSした。
