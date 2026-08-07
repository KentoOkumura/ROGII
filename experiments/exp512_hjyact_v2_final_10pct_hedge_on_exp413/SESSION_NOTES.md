# exp512_hjyact_v2_final_10pct_hedge_on_exp413 セッションノート

## 現在の状態

- Route: `ensemble`
- 実装: 完了（別名compact self-contained inference候補）
- 最終weight: exp413 `0.50` / hjyact-v2 final `0.50`
- 正規Notebook採用: 未承認
- Kaggle package/push/run: 承認済み・version 1--6実行済み
- 現在の判定: version 6 COMPLETE、current-test submit-check PASS
- 提出: 未承認・未実行
- CV / LB: なし

`10pct_hedge`というディレクトリ名は初期設計の履歴識別子として残した。2026-08-05のユーザー指示を
最終契約として、コード、設定、テスト、文書はすべて`0.50 / 0.50`へ変更した。

## 2026-08-05 実装記録

- `kaggle-review-exp`に従い、実装前にsteeringとsource/parent契約を監査した。
- `kaggle-platform`で公開kernelの現在のsourceとinput inventoryを取得した。
- source identity:
  - kernel: `hjyact/ultimate-pf-config-strategy-a-reproducible-score`
  - id/version/run: `128161011 / 2 / 337064157`
  - pull Notebook SHA: `4b4879a6d427422c127a300e09dc763b71ea5e7878eb3639941c75753a23933c`
  - 全code cell連結SHA: `ee93ce4c80c6490cbf2f9cfe518e8e3b54516c212aa813c4a045a64b4c126088`
  - visible final submission SHA: `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`
- 初期設計にあった`cced2c1a...`はlegacy design snapshot IDとして残し、実際にpullしたraw Notebookとcode-cell
  SHAを別フィールドへ明示した。
- active code cell 37個を抽出し、診断/plot/CV-only cellを除いた。sourceのPython 3.12型f-string 1箇所は
  ローカル/Kaggle互換構文へquoteだけ修正した。
- learned trajectoryのprecomputed visible CSV探索と推論時training fallbackを削除し、SHA固定の3保存modelと
  dynamic test featureだけを許可した。
- exp413 runtime source SHA `0eea5b11d6852d0c2170914e993d6aba1204c02f2de00c3a809b299c028ef1dd`
  をgeneratorで検証して関数本体を埋め込んだ。

## candidate reuse契約

- hjyact learned-replayから次の共有面を1回生成し、同一process内でexp413へ渡す。
  - raw well/typewell alignment
  - learned 7-beam bank
  - multiscale NCC bank
  - formation/dense geometry bank
  - deterministic GR/geometry feature block
- exp413 adapterは共有frameをdeep copyし、exp413固有の`pf_ancc`、`pf_z`、stable-seed likelihood-PFだけを再生成する。
- hjyact learned likelihood-PFはGR sigma `x1.3`, seed base 0、exp413は`x1.0`, well別stable seedを保持する。
- trackerはwell/nodeごとのdefinition/input/parameter/seed/dtype/order/content SHA、generation=1、exp413 hit=1を
  `candidate_reuse_manifest.json`へ保存し、不一致時はfail-closeする。

## 実行量preflight

- active scientific variant: 1
- final blend: 1（weight fit/gridなし）
- LightGBM train config: 0
- new booster: 0
- parent/control retraining: 0
- source Ridge: 1 config × 5 folds = 5 runtime fits
- exp413 saved model files: 40 parent selector + 20 signed selector + 15 TVT = 75
- hjyact saved model files: 5 trainer wrappers + 3 learned + 5 model-package = 13
- saved model files total: 88
- contained estimators: exp413 75 + trainer folds 25 + learned 3 + model-package 5 = 108
- accelerator contract: GPU / internet off

## 2026-08-05 Kaggle実行承認

- ユーザーの「実行してください」を、candidateのKaggle GPU/internet-off推論、`submission.csv`生成、
  完了監視、必要なoutput取得と技術検証までの承認として記録した。
- competition submit、正規Notebook採用、weight/profile/threshold変更への承認には拡張しない。
- push前実行量を再確認した: scientific variant 1、final blend 1、LightGBM train config 0、
  new booster 0、parent/control retraining 0、saved model files 88、runtime Ridge 1 config × 5 folds。
- 正規`*_inference.ipynb`はplaceholderのまま維持し、compact candidateから実行専用
  `*_current_test_inference.ipynb`を生成してcanonical slug
  `kentookumura/exp512-hjyact-v2-equal-blend-inference`へpushする。
- package preflightはGPU `Gpu`、internet off、run-on-push、7 datasets、11 kernel sources、competition inputでPASS。
  execution Notebook SHAは`31e7ec39...9aac803`、bootstrap済みpackage SHAは`cdc8a4cc...794fdf`。
- push前の同一kernel pullは403だったため、既存private kernelは存在しないものとして初回pushへ進む。
- 初回pushはKaggle `SaveKernel 400`で実行前に停止した。response本文は
  `The kernel source must be less than 1 megabytes in size.`で、slug/title/inputではなく1 MiB source制限が原因。
- 科学コードは変えず、全`src/` bundleを外し、exp413 runtimeが実際にimportする
  `src/__init__.py`、`candidate_selector_pipeline.py`、`signed_residual_meta.py`だけを明示bootstrapする
  `--no-src` packageへ縮小して、同じcanonical kernel idへ再pushする。
- 縮小packageは909 KiB、SHA`b6f1cec5...71f38ed`となり、同じslugへのversion 1 pushが成功した。
- Kaggle id_noは`129733543`。pull metadataでprivate、GPU `Gpu`、internet off、7 datasets、11 kernels、
  competition inputを確認し、push直後statusは`RUNNING`。
- version 1は129.020秒で`ERROR`。全support/input SHA確認後、Ridge fitと予測生成前に
  `FormationPlaneKNN`へ空well listが入り`KeyError: wid`で停止した。
- 原因はsource version 2の旧固定mount
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction`が現在のKaggle mountに存在しないこと。
  旧pathと`/kaggle/input/rogii-wellbore-geology-prediction`をtrain/test/sample内容で一意解決するだけの
  path resolverを追加した。profile、seed、model、特徴、weightは変更していない。
- 修正後candidateは6,878行/SHA`a2c2fd7d...f92cc6a5`、v2 packageは911 KiB/SHA
  `7e7ec785...8865791`。静的検証と専用pytest 6件を再PASSして同じkernelへversion 2をpushする。
- version 2は135.781秒で`ERROR`。競技データrootの一意解決と全input SHA監査はPASSしたが、
  Ridge特徴量tableだけが旧dataset mount
  `/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts`を固定参照し、読込前に停止した。
  Ridge fit、保存model推論、最終予測はまだ0回である。
- 既にSHA監査済みの`HJYACT_INPUT_AUDIT["roots"]["ridge"]`をRidge rootへ渡すだけのpath修正を行う。
  model、profile、seed、特徴量定義、0.50/0.50 weightは変更せず、同じkernelのversion 3として再実行する。
- version 3 packageは932,691 bytes、SHA`9de6d51a...350ed157`でKaggle 1 MiB制限内。
  構文、Ruff F821、Jupytext round-trip、専用pytest 6件、strict experiment validationを再PASSした。
- canonical kernel `kentookumura/exp512-hjyact-v2-equal-blend-inference`へversion 3をpushし、
  `RUNNING`を確認した。
- version 3は1,194.469秒で`ERROR`。Ridge 5 fold、hjyact完成版、exp413保存model推論、
  14,151行の0.50/0.50 `submission.csv`生成まで完了したが、最後のexp413 exact content parityで
  fail-closeした。hjyact visible submission SHA parityは先にPASSしている。
- exp413 observed decompressed SHAは`3a9bbd1f...be68d87`、期待は`875a1334...584dc4`。
  v3診断用にparity parquet、exp413 prediction、inference metrics、submissionだけを`/tmp`へ取得した。
  外部submitは行っていない。
- referenceとの比較では12 candidate値と、K16以外の全native confidenceが全14,151行で一致した。
  train-only K16の`np.linalg.lstsq`係数がGPU環境で最大`7.06484e-9`ずれ、
  `geometry_gr_delta`だけ3,300行/最大`2.38419e-7 ft`差、exp413 finalは2,203行/
  最大`0.0165 ft`、RMSE`0.000753012 ft`差となった。
- 信頼済みexp413 referenceのtrain-only K16係数を保存パラメータとして固定する。runtime fitは継続し、
  pinned値との差が`1e-7`を超えればfail-closeする。候補、モデル、weight、hidden routingは変えない。
- version 4 packageは936,542 bytes、SHA`61291760...13d8a67`で1 MiB制限内。
  Jupytext round-trip、構文、Ruff F821、専用pytest 7件、strict experiment validationをPASSした。
- 同じcanonical kernelへversion 4をpushし、`RUNNING`を確認した。
- version 4は1,229.395秒で`ERROR`。runtime-fit/pinned K16係数差は`7.06484e-9 < 1e-7`で
  audit PASSしたが、exp413 prediction SHAはversion 3と同じ`3a9bbd1f...be68d87`でparity FAILした。
  係数固定だけでは、K16 GR posteriorの`posterior @ grid`自体がGPU workerのBLAS縮約順に依存する。
- raw data全773 train / 3 test wellsを使うK16限定ローカル診断では、Haswell OpenBLASの元`matmul`が
  referenceのcandidateと`geometry_gr_delta`を14,151/14,151行bitwise再現した。固定加算順の
  sum/einsum/fsum/sequential/lanes/pairwiseは最大`2.38419e-7 ft`の差を残した。
- hjyact本体のGPU数値経路を変えないため、K16だけをfresh Python subprocessへ隔離し、NumPy import前に
  `OPENBLAS_CORETYPE=Haswell`を固定する。childはarchitectureを`Haswell`へfail-closeし、train-only
  runtime fit audit、dynamic raw-test prediction、行/finite/content SHAを実行してparentへparquetを返す。
  static visible prediction sidecarは使わない。
- version 5 packageは947,252 bytes、SHA`67a1acae...ca86b04e`で1 MiB制限内。
  child code単体構文、Jupytext、py_compile、Ruff F821、pytest 7件、strict validationをPASSした。
- 同じcanonical kernelへversion 5をpushし、`RUNNING`を確認した。
- version 5は1,187.181秒で`ERROR`。K16 childはfresh Python processで`Haswell` architectureを確認し、
  runtime-fit/pinned係数差`7.064842932891224e-9 < 1e-7`をPASSした。Ridge 5 folds、hjyact final、
  exp413 saved-model推論、0.50/0.50 prediction生成まで完了し、hjyact visible parityもPASSした。
- それでもexp413 decompressed content SHAはversion 3/4と同じ
  `3a9bbd1f...be68d87`で、期待`875a1334...584dc4`と一致しなかった。CPU architectureの固定だけでは
  K16 posterior decodeのplatform依存BLAS縮約差を消せなかった。NumPy/OpenBLAS version差は可能性があるが、
  exp413 reference実行時のversion記録がないため原因確定とは扱わない。
- version 3の必要最小限の診断outputを用いた提出形式チェックはPASSした。14,151行/2列、header/row数、
  ID順、重複0、NaN/Inf 0をsample submissionと照合し、診断CSV SHAは
  `b960c2b16f01e4224850a5c644a04b792a471b3a0def08018a8d184fea713e23`だった。
  ただしexp413 component exact parityがFAILしているため、総合submit readinessは`FAIL`であり外部提出していない。
- 次の妥当な選択肢は、(1) exact SHA契約を維持して環境再現を続ける、または
  (2) exp413差（2,203行、最大0.0165 ft、RMSE 0.000753012 ft）を数値許容して現在のdynamic predictionを
  採用する、の2つである。後者は科学契約変更、前者は追加の約20分/Kaggle GPU再実行を伴うため、
  どちらもユーザー確認前には進めない。static visible sidecarは引き続き禁止する。

## 2026-08-05 numerical tolerance承認とversion 6 preflight

- ユーザーは選択肢1を明示選択し、exp413 reference差の許容上限をmax absolute `0.02 ft`、
  RMSE `0.001 ft`として進めることを承認した。これはcompetition submitや正規Notebook採用の承認ではない。
- gateを無条件に緩めず、v3--v5で同一再現し、ローカルreference比較でmax `0.0165 ft`、
  RMSE `0.0007530119954096194 ft`と監査済みのcontent SHA `3a9bbd1f...be68d87`だけをnumerical witnessとして
  許可する。exact reference SHA `875a1334...584dc4`も引き続きPASSする。未知の第三SHAはfail-closeする。
- static exp413 prediction sidecarは追加せず、dynamic sampleからのexp413生成を維持する。
- version 6もscientific variant 1、final blend 1、LightGBM train config 0、新規booster 0、
  parent/control再学習0、runtime Ridge 1 config × 5 folds、保存model 88ファイル/108推定器で不変。
- candidate sourceは7,120行、SHA`66ed4f78...3c18f804c`。候補Notebook SHAは
  `08ca7417...e095037`、実行Notebook SHAは`c0154c38...12ca97d3`。正規Notebookは未変更。
- 構文、Ruff F821、Jupytext round-trip、専用pytest 7件、strict validationをPASSした。
- `--no-src`と明示32 support filesで再packageし、952,694 bytes / SHA
  `1ff97aba...0a4cf13`でKaggle 1 MiB制限内。embedded configはnumerical max `0.02 ft`、RMSE
  `0.001 ft`、submission authorization false、GPU/internet offを確認した。
- pre-push pullでcanonical kernel id_no `129733543`、private、GPU、internet off、同一11 kernel / 7 dataset
  inputsを確認した。同じslugへversion 6としてpushする。
- canonical kernelへversion 6をpushし、直後status `KernelWorkerStatus.RUNNING`を確認した。
- version 6はscientific output生成を1,571.153秒、Notebook後処理を含むlog終端を1,587.335秒で
  `KernelWorkerStatus.COMPLETE`。14,151行の`submission.csv`を生成し、external submissionはfalse。
- hjyact final SHAは`b192d3f3...b9ded4a`でsource exact parity PASS。exp413 content SHAは
  `3a9bbd1f...be68d87`で監査済みwitnessと一致し、reference差max `0.0165 < 0.02 ft`、
  RMSE `0.000753012 < 0.001 ft`でnumerical gate PASS。K16 runtime-fit/pinned差も`7.06484e-9 < 1e-7`。
- candidate reuse manifestは5 nodes × 3 wells = 15 recordsすべてgeneration=1 / cache hit=1、
  consumer hit 3、fallback false。共有frame生成は7.78478秒。
- CSV再読込後の0.50/0.50 formula max errorは`1.8189894e-12 ft < 1e-9`。header/row数、ID順、
  重複0、NaN/Inf 0をsample submissionへ照合し、`kaggle-submit-check`はFAIL/WARNなしでPASSした。
- submission SHAは`b960c2b1...a713e23`でversion 3診断runとbyte-identical。正式な同一v6 package rerunは
  1/2のため、hidden-well stochastic determinismとdeterministic anchorは未証明のままにする。
- 必要最小限のsubmission/components/metrics/reuse/reproducibility出力だけを`/tmp/exp512-v6-output`へ取得した。
  外部submit、正規Notebook採用、full output archive取得は行っていない。

## 2026-08-05 version 7 speed-run preflight

- ユーザーは、SP45のtest-well loopを4並列、exp413 exact/self-GR HMM・route PF・K16をwell単位4並列にし、
  model-package correctionを無効化したうえで、対象を一部wellへ縮めず全well inferenceをrunするよう明示した。
- scientific variant 1、final blend 1、LightGBM train config 0、学習fold 0、新規booster 0、
  parent/control再学習0。source Ridgeだけは従来どおり1 config × 5 runtime fits。
- model-package 5 modelを入力・load・推論から外したため、保存model読込はexp413 75 + hjyact 8 = 83ファイル、
  trainer wrapper内部を含む推定器は103。model-package dataset sourceもKaggle metadataから削除した。
- SP45はlocal Generatorのseed `0..127`をwell内で従来順に保持する。exp413 PFはstable per-well seed、
  HMM/K16は決定論的入力を保持し、全経路をtask完了順ではなくtest-well入力順で親側が再結合する。
- candidate sourceは7,236行、SHA`16982879...1b6240f`。候補Notebook SHAは`c02af8a5...fc60cc`、
  execution Notebook SHAは`246b5414...fae69f`。正規Notebookは未変更。
- 構文、Ruff F821、Jupytext round-trip、専用pytest 8件、strict validationをPASSした。
- canonical kernelのpre-push pullでid_no `129733543`、private、GPU、internet offを再確認した。
- `--no-src` packageは960,584 bytes / SHA`76f03a8f...32988a`で1 MiB未満。run-on-push true、
  competition input 1、dataset input 6、kernel input 11、model-package dataset 0を確認した。
- competition submitと正規Notebook採用は引き続き未承認であり、このrunでは行わない。
- canonical kernel version 7のpushに成功し、直後statusは`KernelWorkerStatus.RUNNING`。
- version 7は`KernelWorkerStatus.COMPLETE`。visible 3 wellの科学処理終端は`1,197.667秒`で、version 6の
  `1,571.153秒`から`373.486秒 / 23.771%`短縮した。ただしこの比率を9時間提出制限の余裕とは解釈しない。
- SP45はrequested 4 / effective 3 workers、3 wells / 14,151 rowsを`303.658秒`。version 6のsequential
  `218.912秒`より`38.712%`遅く、visible 3 wellではThreadingBackendのCPU競合/overheadが勝った。
  したがって全体短縮をSP45並列化の効果とは解釈しない。
- exp413全体は`483.784→297.388秒`（`38.529%`短縮）。route PFはeffective 3 / nogilで
  `51.124→40.961秒`（`19.879%`短縮）、exact/self-GR HMMは`55.449 / 57.535秒`、K16はeffective 3で
  `61.354→46.116秒`（`24.836%`短縮）。K16 Haswell/pinned-kappa監査も`7.06484e-9 < 1e-7`でPASS。
- model-package correctionはlog・metrics・model manifestの3箇所でfalse/skipped、入力model 0を確認した。
  version 6では最終weight 0だった経路を省いたため、hjyact final exact SHAは不変。
- hjyact SHA `b192d3f3...b9ded4a` exact PASS、exp413 SHA `3a9bbd1f...be68d87` witness PASS、
  formula / exp413 CSV boundary max errorはいずれも`0.0 ft`。reuse 15/15 recordsはgeneration=1 / cache hit=1、fallback 0。
- final submissionは14,151行、2列、sample ID順exact、重複0、NaN/Inf 0。submit-checkはFAIL/WARNなしでPASS。
  SHA `b960c2b1...a713e23`はversion 6とbyte-identical。external submissionはfalse。
- 必要最小限のsubmission/components/metrics/reuse/reproducibility出力だけを`/tmp/exp512-v7-output`へ取得した。
  full output archive、正規Notebook採用、competition submitは行っていない。
- 200 well評価はwell比例工程ごとに外挿した。SP45 `4.2–5.6時間`、learned `0.9–1.0時間`、Gold
  `2.2–2.7時間`、model-package `0時間`、exp413 `4.1–5.5時間`、固定費/I/O `0.1–0.2時間以上`で、
  合計`約11.5–15.0時間`。現行v7は9時間制限に対してFAIL / 要追加高速化と訂正する。

## 再現性と検証

- 候補source: 7,120行、SHA
  `66ed4f78e0c3525ab7f7f52d99f2b2a1cb100e36223c6465bd49c2b3c18f804c`
- 候補Notebook: 51 cells、SHA
  `08ca74175cd85defd683b41a0e8763f3ef36dfa8b381b66114125df53e095037`
- 静的検証: Jupytext変換/round-trip、`py_compile`、Ruff F821、専用pytest 7件、`validate-exp`がPASS。
- visible既知SHAはdynamic sample ID-order SHAが一致した後のpost-hoc assertionだけに使う。
- deterministic anchor: まだ不可。v3/v6 submissionはbyte-identicalだが、同一v6 package rerunは1/2であり、
  hidden-well stochastic pathの決定性も未証明。
- honest OOFなし。条件付きRMSE三角上限は
  `0.50 * 7.201 + 0.50 * 6.568 = 6.8845`だが、同じPublic rowsで両scoreが再現される場合だけ成立する。

## 実行コマンド

実行済み:

```bash
.venv/bin/python scripts/prepare_exp512_hjyact_v2_candidate.py /tmp/exp512-hjyact-v2-source/ultimate-pf-config-strategy-a-reproducible-score.ipynb
.venv/bin/python -m py_compile experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py
.venv/bin/ruff check --select F821 experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py
.venv/bin/pytest -q experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/test_exp512_contract.py
make validate-exp EXP=exp512_hjyact_v2_final_10pct_hedge_on_exp413
```

未実行または未完了:

- 正規Notebook採用
- 同一条件2-run reproducibility gate
- Codexによるcompetition submit

## 2026-08-05 code submission scoring完了

- ユーザーからexp512のPublic LBが`6.541`であると明示された。
- Kaggle CLIのsubmission一覧でref `55255459`、submitted
  `2026-08-05 02:08:11.450000 UTC`、status `COMPLETE`、Public LB `6.541`、Private LB空欄を確認した。
  exp512への帰属はユーザー確認を正とする。submission一覧はdescriptionとscript versionを返していないため、
  refだけからkernel versionを再推定しない。
- Codexはcompetition submitを実行していない。`competition_submission_approved: false`はagent actionの履歴として
  保持し、別途`external user submission observed`として記録した。
- Public LBはexp413 / exp510の`7.201`から`-0.660`、source公開値`6.568`から`-0.027`。
  新しい全体・アンサンブルPublic LB基準とする。条件付き三角上限`6.8845`との差は`-0.3435`。
- 数値score付き`COMPLETE`は、実際に提出されたhidden rerunがplatform上限内で完走した直接証拠である。
  一方、監視開始時点ですでに完了し後続提出も存在したため、正確なscoring elapsedは復元不能。Kaggle submissions
  APIもhidden well数・工程別runtimeを返さない。したがって事前の200-well工程別外挿`11.5–15.0時間`は
  保守的planning auditとして残し、visible 3 wellだけからhidden runtimeを逆算しない。
- score完了はformal同一package 2-run、hidden-well stochastic determinism、honest OOF、Private LBを証明しない。
  weight/profile変更、追加rerun、再提出は行っていない。

## 2026-08-05 v6構成latest-version rerun preflight

- ユーザーは速度最適化v7ではなく、v6構成をもう一度実行してcanonical kernelの最新versionにするよう明示した。
  この承認はKaggle package push/runまでで、competition submitと正規Notebook採用は含まない。
- `kaggle kernels pull .../6`は403だったが、v7 push直前に同kernelから取得済みの
  `/tmp/exp512-v7-prepush.HuDLU8/exp512-hjyact-v2-equal-blend-inference.ipynb`が残っていた。
  これは52 cells / 42 code cells、code-cell SHA `9b5a4cae...bdabee6`、pulled Notebook SHA
  `8823e6ca...58a09b1`。
- embedded support ZIPは32 files。candidate SHA `66ed4f78...c18f804c`がv6記録と一致し、embedded configは
  `numerical_tolerance_contract_approved_v6_preflight`、model-package enabled。manifestはmodel-package 5 files、
  metadataは`pilkwang/rogii-model-package`を含む7 datasets、GPU / internet off、canonical id/title exactである。
- v6 sourceにはv7の`SP45_WELL_N_JOBS` / `EXP413_WELL_N_JOBS`定数とmodel-package-disable guardが存在しない。
  したがってv7 sourceを逆編集した近似復元ではなく、Kaggleに存在したv6 sourceの再pushである。
- push前実行量: scientific variant 1、final blend 1、LightGBM train config 0、new booster 0、parent/control
  retraining 0、runtime Ridge 1 config × 5 folds、保存model 88 files / 108 estimators。
- expected visible signaturesはhjyact `b192d3f3...b9ded4a`、exp413 witness `3a9bbd1f...be68d87`、
  final submission `b960c2b1...a713e23`。不一致時はfail-closeし、自動再pushしない。
- exact v6 pull directoryを変更せず`kaggle kernels push`し、canonical kernel version 8のpushに成功した。
  competition submitは行っていない。
- push直後のlatest source/metadataを再pullした。pulled Notebook SHAはv6 sourceと同じ
  `8823e6ca...58a09b1`、code-cell SHAも`9b5a4cae...bdabee6`。7 datasets、model-package dataset、
  GPU `Gpu`、internet off、canonical id/titleの一致を再確認した。
- version 8は`KernelWorkerStatus.COMPLETE`。scientific output生成は`1,193.477秒`、Notebook変換までの
  total logは`1,205.751秒`。visible 3 well実測の内訳はSP45逐次well loop `175.581秒`、
  visible-prefix `112.3986秒`、exp413 `360.328秒`だった。これら3 well実測だけをhidden時間へ外挿しない。
- hjyact component `b192d3f3...b9ded4a`、exp413 component content `3a9bbd1f...be68d87`、
  final submission `b960c2b1...a713e23`はv6とbyte-identical。固定0.50/0.50 formula errorは`0.0 ft`、
  current-test submit-checkもPASSした。これによりsame-v6 visible component/final output gateは2/2 PASS。
- candidate reuseは15 recordsすべてgeneration=1 / cache hit=1、fallback 0。model-package correctionはv6どおり
  5 modelsを実行したが、p95差`26.700659 ft > 25 ft`のguardで無効化され、最終weightは0だった。
- v6/v8のcandidate reuse manifest比較では、15 records中3 wellの`deterministic_gr_geometry_feature_block`だけ
  content SHAが一致しなかった。他のrecord契約と最終2 component / submissionは一致したため、visible出力の
  再現gateはPASSとするが、全中間生成物のbyte再現やhidden-well stochastic determinismまでは主張しない。
- latest versionはversion 8のexact v6 contractになった。v7の4並列/model-package無効化は履歴として残るが、
  latest sourceには適用されていない。v6工程別200 well概算`約14–18時間`は9時間planning gateをFAILのまま残す。
  user submission ref `55255459`のCOMPLETEは別のplatform完走証拠だが、APIからscript version・hidden well数・
  正確なruntimeを取得できないためversion 8の9時間保証には流用しない。
- 必要最小限のsubmission/components/metrics/reuse/reproducibility/model-package reportだけを
  `/tmp/exp512-v8-output.PHU5Kw`へ取得した。full output archiveとcompetition submitは実行していない。
