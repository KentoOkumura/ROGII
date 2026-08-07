# exp270 exact HMM posterior mode candidate audit セッションノート

## 目的

exp209 raw exact HMM の posterior mean を control とし、同一 posterior/score から marginal MAP、global Viterbi、joint top-5 path を target-free に生成して mode candidate の direct 品質と oracle headroom を監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: version 4の2 shardとSHA固定aggregateが完了。technical PASS、direct mode候補negativeでbranch closed
- CV / LB: posterior mean direct RMSE 11.938287 / 対象外
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## ユーザー確認済み契約

- 親は exp223 ではなく exp209。
- joint top-K は 5。
- TVT grid-index sequence が同一なら rate path が違っても重複排除。
- block oracle は 128 / 256 / 512 行。

## 実行量

Kaggle train push 前の固定値は次のとおり。

- active HMM variants: 1
- LightGBM configs: 0
- folds: 0
- total boosters: 0
- HMM well-runs: 773
- outer workers: 1
- Numba threads: 4
- GPU: false
- parent/control GPU retraining: false
- inference / submission: false / false

posterior mean は mode readout に必要な forward-backward の同じ CPU pass から再生成する。保存済み exp209 cache は parity control として読み、decompressed SHA `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5` を必須とする。

## 実装内容

- exp209 と同じ target-free HMM input preparation と forward-backward を self-contained notebook に展開。
- 各 state の上位 5 部分 path を保持する exact joint-state DP を追加。
- position/rate/rank の backpointerを 75 通りの遷移 code に畳み、`uint8` 1 byte に圧縮。
- global top-5 を復元後、TVT position-index 列の SHA256 で重複排除。unique 5 未満を許容し backfill なし。
- posterior mean/MAP、mode mass/gap、path score/log posterior、edge/switch/curvature、pairwise path distance を保存。
- direct、distance、hidden-like、by-well、focus well、unique-best、row/block/well oracle を実装。
- oracle prediction、selector、blend、full posterior、rate path、submission は保存しない。
- schema、decoder manifest、raw/decompressed gzip SHA、prediction array SHA、raw input manifest SHA を保存。
- inference notebook は fail-closed で停止する。

## ローカル確認ログ

実データを使う notebook 実行はしていない。実行済みの静的確認は次のとおり。

```bash
make new-steering EXP=exp270_exact_hmm_posterior_mode_candidate_audit
make new-exp EXP=exp270_exact_hmm_posterior_mode_candidate_audit
.venv/bin/python -m py_compile experiments/exp270_exact_hmm_posterior_mode_candidate_audit/exp270_exact_hmm_posterior_mode_candidate_audit_train.py
.venv/bin/ruff check experiments/exp270_exact_hmm_posterior_mode_candidate_audit/exp270_exact_hmm_posterior_mode_candidate_audit_train.py tests/test_exp270_exact_hmm_posterior_mode_candidate_audit.py
.venv/bin/pytest -q tests/test_exp270_exact_hmm_posterior_mode_candidate_audit.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp270_exact_hmm_posterior_mode_candidate_audit/exp270_exact_hmm_posterior_mode_candidate_audit_train.py
make validate-exp EXP=exp270_exact_hmm_posterior_mode_candidate_audit
make prepare-kaggle-notebooks EXP=exp270_exact_hmm_posterior_mode_candidate_audit EXTRA_ARGS="--strict"
```

- targeted tests: 7 passed
- ruff: passed
- py_compile: passed
- strict experiment validation: passed
- Kaggle package preparation: passed
- small-trellis exhaustive parity: exact joint top-5 score/path passed
- synthetic exp209 forward-backward parity: posterior/log-likelihood exact一致
- synthetic well integration smoke: 5 evaluation rows、posterior mean / top1 finite、joint top-5からunique TVT path 3本へdedup、backpointer 285,360 bytesで完走

## Kaggle 実行ログ

### 2026-07-17 version 1

- push時刻: 2026-07-17 13:14 UTC / 22:14 JST
- kernel: `kentookumura/exp270-exact-hmm-posterior-mode-audit-train`
- id_no: `127594551`
- version: 1
- accelerator: CPU (`enable_gpu=false`, `machine_shape=None`)
- internet: false
- run_on_push: true
- input kernels: exp209 id_no `126193687`、exp115 id_no `124519917`
- 実行量: HMM 1 variant / 773 well-runs / LightGBM 0 config / 0 fold / 0 booster
- URL: https://www.kaggle.com/code/kentookumura/exp270-exact-hmm-posterior-mode-audit-train

初回に directory 名全体を使った53文字slug `exp270-exact-hmm-posterior-mode-candidate-audit-train` を試したが、Kaggle `SaveKernel 400` で作成されなかった。pullが403で未作成を確認し、同じexp270のまま科学内容を変えず、43文字のcanonical slug `exp270-exact-hmm-posterior-mode-audit-train` と同一titleへ揃えて再prepare/pushした。version 1 pushとpull metadata確認は成功した。

version 1 は13秒でERROR。competition data が `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train` にmountされた一方、resolverが旧direct pathと1階層globしか探索せず、`raw_wells=0` でfail-closeした。HMM計算は開始しておらず、生成物・数値結果はない。

### 2026-07-17 version 2

- push時刻: 2026-07-17 13:18 UTC / 22:18 JST
- kernel / id_no: version 1と同じ `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` / `127594551`
- 変更: Kaggle competition mount pathとrecursive fallbackをtrain data resolverへ追加
- targeted tests: 7 passed（competition mount resolver testを追加）
- 科学条件・実行量・入力kernel・CPU metadata: version 1から変更なし
- status: 20 / 773 wells完了時点でversion 3へ置換

competition mount resolver修正後、`raw_wells=773` と先頭wellのexact joint top-5生成を確認した。20 wellまでの実測はおおむね35.6--72.8秒/wellで、全3,783,989 evaluation rowsではKaggle実行上限への余裕が小さかった。notebookは全well完了後にのみ成果物を保存するため、version 2の途中生成物は保存対象外である。

### 2026-07-17 version 3

- kernel / id_no: version 1--2と同じ `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` / `127594551`
- 変更: forward-backward / exact top-Kのposition遷移を、rateごとにinner `prange`を41回起動する形から、rate軸を1回の`prange`で処理する同値なNumba schedulingへ変更
- 科学条件・候補定義・top-5 dedup・入力・実行量・CPU metadata: version 2から変更なし
- synthetic top-K benchmark: 256 time x 573 grid x 41 rate、4 threadsで1.668秒から1.078秒（約35%短縮）
- Numba有効 targeted tests: 7 passed（small-trellis exhaustive top-5 score/path、exp209 forward-backward exact parityを含む）
- status: Kaggle time limitで`CANCEL_ACKNOWLEDGED`

通常logsは1,202 entries、最終時刻27,467.399秒（7時間37分47秒）だった。全773 wellの開始logはあり、772番目`ff8bb73a`は42.552秒で完了、773番目`ffefef30`を開始した直後で終了した。次well開始logをもって直前wellの関数returnを確認できるため、772 / 773 wellsはmemory上で生成完了している。ただし生成物保存は全well loop、concat、parity、metrics集計の後に行う設計だったため、`kaggle kernels files`は空で、candidate、metrics、SHA、summaryはいずれも残っていない。したがってpartial resultを数値評価には使わない。

version 3の高速化自体はversion 2より有効だったが、単一notebook full runではKaggle time limit内に集計・保存の余白を確保できなかった。科学仮説の棄却ではなく、execution/checkpoint設計の失敗として扱う。

## 再現性メモ

- seed policy: `no_rng_exact_hmm_decoder`
- stochastic components: なし
- CPU/GPU runtime: CPU only、outer workers 1、Numba threads 4
- raw input: horizontal/typewell file SHA と aggregate manifest SHA を保存予定
- feature schema/content: schema SHA、candidate gzip raw/decompressed SHA を保存予定
- decoder manifest: HMM / decoder / candidate bank config の canonical JSON SHA を保存予定
- prediction: row index + float32 candidate matrix の content SHA を保存予定
- model / submission: 学習 model と submission は存在しない
- deterministic anchor: false。Numba parallel floating arithmetic の微小差を tolerance と content SHA で監査する
- Kaggle kernel id / version: `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` version 3 / id_no `127594551`

## 次のアクション

1. shard 0 / 1をKaggle CPUで実行し、各artifactのcoverageとSHAを確認する。
2. 確認したraw/decompressed/sidecar SHAをconfigへ固定し、aggregate notebookをprepare/pushする。
3. aggregate完了後にexp209 mean parity、coverage、finite、unique path count、direct/oracle/hidden-like/worst/focus readoutを記録する。

## 2026-07-18 deterministic 2-shard recovery

- ユーザー承認に基づき、単一runの科学内容を変えず、well idだけの
  `sha256("exp270::well_shard::<well>")[:8]` little-endian modulo 2で2分割する。
- shard 0は363 wells / 1,792,363 rows、shard 1は410 wells / 1,991,626 rows。
  合計は773 wells / 3,783,989 rowsで、HMM well-runsはversion 3と同じ773のまま。
- 実行前契約: active HMM variant 1、LightGBM config / fold / booster 0 / 0 / 0、
  outer worker 1、Numba threads 4、GPU / inference / submissionなし。
- exp209 cacheはdecompressed SHA固定のposterior-mean parity controlとして読むだけで、
  controlや親モデルの再学習は行わない。
- `train_variant0/1`は候補gzip、path診断、pairwise距離、well/input manifest、summary、
  各SHAをshard単位で保存する。canonical `train`は両shardのrows/wells/stable partitionと
  raw/decompressed/sidecar SHAをfail-closedで確認してから集約する。
- target-free分割、集約前SHA固定、3 notebookのmode固定を含むtargeted testsは
  通常環境10件およびNumba JIT環境10件でPASS。Jupytext round-trip、py_compile、Ruff、
  strict experiment validationもPASSした。

### shard package preflight

- shard 0: `kentookumura/exp270-exact-hmm-mode-audit-shard0`
- shard 1: `kentookumura/exp270-exact-hmm-mode-audit-shard1`
- 両packageともprivate、CPU、internet false、run-on-push true、competition sourceと
  `kentookumura/exp209-joint-exact-parity-train`だけを入力に持つ。
- package内config SHAとnotebook bootstrap manifestのconfig SHAは両shardとも
  `8942117b7216a1dd91d1149bc26e21f80539129cbe488aea9bc4f01702218b33`で一致した。
- notebook内`RUN_KIND_OVERRIDE`はshard 0 / 1へ固定され、package内
  `kaggle_push_approved=true`、合計773 HMM well-runs、0 config / 0 fold / 0 boosterを確認した。
- `kaggle kernels list --mine`で上記2 slugが未作成であり、既存versionを上書きしないことを確認した。

### Kaggle CPU shard version 1 push

- shard 0: version 1、id_no `127732571`
- shard 1: version 1、id_no `127732575`
- 両pushは成功し、remote pull metadataでもprivate、CPU、internet false、competition source、
  exp209 kernel sourceのみであることを再確認した。
- push直後のstatusは両方`KernelWorkerStatus.RUNNING`。通常logsはまだ空だったため、
  空logだけで失敗とは判断しない。ユーザー方針どおり継続監視は行わず、完了連絡後に監査する。

### Kaggle CPU shard version 1 failure audit

- 2026-07-19の再確認時点で両shardは`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`、
  `failureMessage=null`、保存fileなしだった。Python tracebackは記録されていない。
- shard 0の最終stdoutは14,860.655秒で363 / 363 well開始時、shard 1は410 / 410 wellの
  HMM生成を17,734.822秒で完了した後の集計中に停止した。
- version 1 pushではKaggle CLIの`kernels push --timeout`を指定していなかった。
  CLI 2.2.3は明示値がある場合だけsession timeoutをSaveKernel requestへ設定するため、
  約5時間で停止した観測と合わせ、Notebook UIの12時間上限ではなくpush時のsession timeout不足を
  実行失敗の原因と判断する。科学仮説、候補、coverageのnegative resultとしては扱わない。

### Kaggle CPU shard version 2 rerun contract

- ユーザー承認に基づき、version 1と同じ2 packageをbyte-levelで変更せず再pushする。
  package config SHAは両方
  `8942117b7216a1dd91d1149bc26e21f80539129cbe488aea9bc4f01702218b33`。
- 唯一の実行条件変更はCLIへ`--timeout 43200`を明示し、12時間のsession timeoutを
  SaveKernel requestへ渡すことである。コード、入力、shard割当、科学条件は変更しない。
- 実行量: active HMM variant 1、shard 0は363 wells / 1,792,363 rows、shard 1は
  410 wells / 1,991,626 rows、合計773 HMM well-runs。LightGBM config / fold / boosterは
  0 / 0 / 0、outer worker 1、Numba threads 4、CPU、GPU / inference / submissionなし。
- exp209 cacheはposterior-mean parity controlとして読むだけで、親/controlの再学習は行わない。

### Kaggle CPU shard version 2 push

- push時刻: 2026-07-19 00:01 UTC / 09:01 JST
- shard 0 / shard 1とも`kaggle kernels push --timeout 43200`でversion 2のpushに成功した。
- kernel / id_no: `kentookumura/exp270-exact-hmm-mode-audit-shard0` / `127732571`、
  `kentookumura/exp270-exact-hmm-mode-audit-shard1` / `127732575`。
- push直後のstatusは両方`KernelWorkerStatus.RUNNING`。ユーザー方針どおり継続監視は行わず、
  完了連絡後にfile、coverage、SHA、parityを監査する。

### Kaggle CPU shard version 2 failure audit

- 2026-07-19 12:08 UTC / 21:08 JSTに両kernelを再監査した。statusは両方
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`、`kaggle kernels files`は両方空だった。
- shard 0の通常logは590 entries、最終時刻17,086.616秒（4時間44分47秒）。
  `363/363 well=fef8af96`の開始が最後で、Python traceback、OOM message、timeout messageはない。
  最終wellはprogress間隔の都合で完了metaを表示しないため、最終well中か直後の後段処理かは
  logだけでは確定できない。再現位置と無例外停止からresource kill、特にmemory pressureが最有力。
- shard 1は410 / 410 wellのHMM生成を11,455.374秒（3時間10分55秒）で完了した。
  その後stdoutがなく、43,200秒で`nbclient.exceptions.CellTimeoutError`になった。
  HMM終了後の無出力処理だけで31,744.626秒（8時間49分5秒）を費やした。
- shard 1のtracebackが`after 43200 seconds`を明示しているため、version 2の
  `--timeout 43200`は正しく反映された。今回の失敗はtimeout指定漏れではない。
- 現実装は全per-well DataFrameを`frames`へ保持し、全HMM後に単一cell内で`pd.concat`、
  full sort、exp209 gzip読込、数百万object idのset / `np.setdiff1d` / index整列、gzip保存、
  raw/decompressed SHAを連続実行する。stage logがなく停止箇所は一意に確定できないが、
  2 shardでもこの後段はmemory / runtime boundedではない。
- 数値生成物はなく、posterior parity、candidate RMSE、oracle headroomは未評価のまま。
- 次案は科学条件を変えず、(A) 4以上のshardへ細分化、または(B) chunk逐次保存・逐次parityで
  全well DataFrame保持を廃止すること。再実行コストと実装範囲が異なるため、採用案はユーザー確認後に決める。

### Kaggle CPU shard version 3 implementation / rerun contract

- 2026-07-19、ユーザーは推奨したmemory-bounded 2 shard方式の実装と再実行を承認した。
- shard割当、親exp209、HMM、decoder、candidate定義、truth後付け、2 shardの363 / 410 wells、
  合計773 HMM well-runsは固定する。LightGBM config / fold / boosterは0 / 0 / 0、CPU、
  outer worker 1、Numba threads 4、親/control再学習、GPU、inference、submissionなし。
- version 2との差はexecution実装だけ。全well DataFrameの累積保持と一括concatを廃止し、
  well単位candidate gzip書込、prediction SHA用binary stream、100,000-row線形parity、
  mtime 0 / compresslevel 1、stage elapsed / current RSS / peak RSS logを導入する。
- aggregateも全object ID sortを線形parityへ置き換え、candidate gzipとprediction SHAを
  fixed chunkで処理する。数値候補と評価定義は変更しない。

### Kaggle CPU shard version 3 implementation / package preflight

- canonical / shard0 / shard1の3 self-contained Jupytext sourceへmemory-bounded実装を反映した。
  shard sourceは`scripts/prepare_exp270_shard_sources.py`でcanonical sourceをmodeだけ置換して生成し、
  mode以外がbyte-levelで一致することを確認した。
- shard generationはper-well candidate frameをsingle deterministic gzip streamへ直ちに書き、
  float32 candidate matrix / int64 row indexだけを一時binaryへ書く。全well frame concatはない。
- exp209 parityは100,000-row candidate/control chunkをordered id/wellで線形照合する。
  aggregateも同じ線形parity、100,000-row deterministic gzip write、chunked prediction SHAへ変更した。
- array-bundle SHAのin-memory / frame-chunk / binary-part一致、mtime 0 gzip raw SHA再現、
  filtered controlとの線形parity、全well frame非保持を含むtargeted tests 13件を追加・更新した。
  通常Numba環境13件、`NUMBA_DISABLE_JIT=1`環境13件ともPASS。
- canonical / shard sourcesのpy_compile、Ruff、3 notebookのJupytext round-trip、strict experiment
  validationはPASSした。ローカルfull HMM runは行っていない。
- 両package config SHA:
  `921e210716fca476dca2d3afb10b9277ed52c08119dd4134d2219b4766cb00b2`。
- package notebook SHA: shard 0
  `a216654ea956b66e40180eb7720f287a155c32f08804a8195b7e6c098ae867be`、shard 1
  `eaeefd4a7632e2b37ec52cab098d1bba8a2cd980d9bf5e2e7b6d71810afa9978`。
- 両packageはprivate、CPU、internet false、run-on-push true、competition inputとexp209 kernel
  sourceのみ。実行量はHMM variant 1 / 合計773 wells、LightGBM config / fold / booster 0 / 0 / 0、
  outer worker 1、Numba threads 4、親/control再学習、GPU、inference、submissionなし。
- remote pullで同じcanonical kernelの存在とid_no `127732571` / `127732575`を確認した。
  version 3は両方`kaggle kernels push --timeout 43200`で実行する。

### Kaggle CPU shard version 3 push

- push時刻: 2026-07-19 12:31 UTC / 21:31 JST。
- shard 0 / shard 1とも同じcanonical kernelへversion 3のpushに成功した。
- commandは両方`kaggle kernels push --timeout 43200`。12時間session timeoutを明示した。
- kernel / id_no: `kentookumura/exp270-exact-hmm-mode-audit-shard0` / `127732571`、
  `kentookumura/exp270-exact-hmm-mode-audit-shard1` / `127732575`。
- push直後のstatusは両方`KernelWorkerStatus.RUNNING`。継続監視は行わず、完了連絡後に
  stage/RSS log、file、coverage、parity、prediction/raw/decompressed/sidecar SHAを監査する。

### Kaggle CPU shard version 3 failure audit

- 2026-07-20に再監査し、両kernelとも`KernelWorkerStatus.ERROR`、保存fileなしを確認した。
- shard 0は363 wells / 1,792,363 rowsのHMMとstream writeを15,488.061秒
  （4時間18分8秒）で完了した。parity開始時のcurrent / peak RSSは416.621 / 1,541.902 MB。
- shard 1は410 wells / 1,991,626 rowsを16,434.264秒（4時間33分54秒）で完了した。
  parity開始時のcurrent / peak RSSは422.645 / 1,553.941 MB。
- 両方ともHMM完了の約18秒後、exp209 posterior-mean parityで明示的にfail-closeした。
  shard 0はmax / mean absolute difference 0.000972656 / 0.000323569 ft、shard 1は
  0.000972574 / 0.000323439 ftで、固定許容値0.00001 ftを超えた。
- traceback経路はcandidate gzip CSVとexp209 controlの線形比較であり、prediction SHA用の
  一時binaryはparityへ使っていない。resource kill、OOM、12時間timeoutではない。
- exp209 `exact_hmm_smoother._numeric_frame`は保存直前に`hmm_mean_tvt`を含む全numeric列を
  float32へcastする。一方、exp270は同じforward-backwardのposterior meanをfloat64のまま
  CSVへ保存していた。観測差の上限と平均は高TVT値のfloat32量子化幅に一致し、synthetic
  forward-backward kernel exact parityも維持されている。保存dtype契約の実装漏れと分類する。
- したがってversion 3はmemory-bounded executionを実証したが、mode candidateの数値評価には
  到達していない。partial candidate、parity、SHA、summaryはいずれも採用しない。

### Kaggle CPU shard version 4 fix / rerun contract

- `posterior_mean`だけをexp209保存済みcontrolと同じfloat32へ正規化してからcandidate gzipへ
  書く。marginal MAP / global Viterbi / top-K等の新規mode候補はfloat64のまま維持する。
- parity許容値0.00001 ft、HMM input / grid / transition / emission / calibration / prior、decoder、
  TVT-sequence dedup、truth境界、shard割当は変更しない。許容値緩和や候補の丸め救済は行わない。
- 実行量はactive HMM variant 1、shard 0 / 1は363 / 410 wells、合計773 HMM well-runs、
  LightGBM config / fold / booster 0 / 0 / 0、outer worker 1、Numba threads 4、CPU、
  親/control再学習、GPU、inference、submissionなし。
- 高TVT値でexp209 float32保存契約を固定する回帰testを追加し、targeted tests 14件を
  通常Numba環境と`NUMBA_DISABLE_JIT=1`環境の両方でPASSした。
- version 4は同じcanonical shard kernel / id_no `127732571` / `127732575`へ、
  `kaggle kernels push --timeout 43200`で再実行する。

### Kaggle CPU shard version 4 package preflight

- canonical sourceからmodeだけを置換してshard 0 / 1 sourceを再生成し、3 sourceのpy_compile、
  Ruff、Jupytext round-trip、strict experiment validationをPASSした。
- package config SHAは両shardとも
  `e90041bace91b49e97e2868da019ef831c39b660f7ca5f5beb35a51f16bf9342`で、notebook bootstrap内
  config SHAともbyte一致した。
- package notebook SHAはshard 0
  `71a51c32cd1d02a7f71e61dc48a0fcef7d0f0d5fc23cddfb3dea200a5d5f66cd`、shard 1
  `13787a808db5637973fa0c6f26ad9dc2094bb7e28d325d43f75a31a545f9361c`。
- metadataはprivate、CPU、internet false、run-on-push true、competition inputとexp209 kernel
  sourceだけである。各notebookの`RUN_KIND_OVERRIDE`、保存dtype修正、合計773 HMM well-runs、
  0 boosterをpackage本体で再確認した。

### Kaggle CPU shard version 4 push

- push時刻: 2026-07-20 00:19 UTC / 09:19 JST。
- shard 0 / shard 1とも同じcanonical kernelへversion 4のpushに成功した。
- commandは両方`kaggle kernels push --timeout 43200`。12時間session timeoutを明示した。
- kernel / id_no: `kentookumura/exp270-exact-hmm-mode-audit-shard0` / `127732571`、
  `kentookumura/exp270-exact-hmm-mode-audit-shard1` / `127732575`。
- push直後のstatusは両方`KernelWorkerStatus.RUNNING`。継続監視は行わず、完了連絡後に
  stage/RSS log、file、coverage、parity、prediction/raw/decompressed/sidecar SHAを監査する。

### Kaggle CPU shard version 4 completion audit

- 2026-07-20に両kernelの`KernelWorkerStatus.COMPLETE`を確認し、通常logsとKaggle outputを取得した。
- shard 0は363 wells / 1,792,363 rows、runtime 16,236.225秒、peak RSS 1,547.344 MB。
  shard 1は410 wells / 1,991,626 rows、runtime 18,007.399秒、peak RSS 1,553.738 MB。
  合計は固定契約どおり773 wells / 3,783,989 rows。
- exp209 posterior-mean parityは両shardともmax / mean absolute difference 0.0 / 0.0 ft、
  tolerance 0.00001 ft、全行PASS。version 3の保存dtype不一致は解消した。
- candidate gzipはshard 0 / 1で63,498,217 / 70,562,316 bytes。取得ファイルからraw gzip、
  decompressed content、candidate schema、decoder manifest、input manifest、path diagnostics、
  pairwise path distance、well manifest、pre-self summary SHAを再計算し、全項目summaryと一致した。
- candidate CSVを100,000-row chunkで再走査し、ID=`<well>_<row_idx>`、well内strict row順、
  fixed candidate finite、stable SHA256 shard割当を全行確認した。prediction array-bundle SHAも再計算し、
  shard 0 `214e060e13f17bc000083694426cb61b1762afb5f69314ebe06cf1d53da9933c`、
  shard 1 `f129a0007e87cb97e147f5ad68fce6e7229825f99475d6dc83419d4052d1cc4f`でsummaryと一致した。
- decoder scientific mapping SHAは両shard
  `c5b9be8b68eae97cb657d04ae63d63ee5d58b2d7f9d75bbb49f45e47c9aa8837`。
- unique TVT path数はshard 0 / 1で平均4.181818 / 4.141463、最小1、5本未満89 / 103 wells。
  no-backfill契約どおりで、候補品質の判定はaggregate readoutへ持ち越す。
- `kaggle kernels files`のsize列は候補gzipを約1 KBと誤表示したため根拠に使わず、取得outputの
  実byte数とSHAを正とした。候補、summary、sidecarはすべて実体を取得できた。

### SHA-fixed aggregate execution contract

- shard 0 / 1のcandidate raw/decompressed、path diagnostics、pairwise path distance、well manifest
  SHAを`config.yaml`へ固定した。aggregateはSHA不一致、coverage不一致、重複ID、stable shard不一致を
  すべてfail-closeする。
- aggregateが実行するHMM variant / LightGBM config / fold / boosterは0 / 0 / 0 / 0。
  保存済み2 shardとexp209 parity control、exp115 hidden-like assignmentだけを読み、CPUで
  direct/oracle/hidden-like/by-well/focus readoutと最終候補・metrics・manifestを生成する。
- 親/control再学習、GPU、inference、submissionなし。候補定義、truth境界、parity tolerance、
  oracle非保存契約は変更しない。

### SHA-fixed aggregate package preflight

- 既存canonical aggregate kernel `kentookumura/exp270-exact-hmm-posterior-mode-audit-train`
  / id_no `127594551`をremote pullで確認し、同じkernelへversion追加する。
- package config SHAは
  `5a8ea2a7e2d19eafac532b0b925ab9668b8aa3ec1756fa15bd1f2592ec36823f`で、notebook bootstrap内
  configとbyte一致。package notebook SHAは
  `51b7c8804cb9d778b1201650d0ae5d35f807674420ab21d3249806d4fccc4933`。
- metadataはprivate、CPU、internet false、run-on-push true。competition input、exp209 control、
  exp115 hidden-like assignment、監査済みshard 0 / 1だけを入力に持つ。
- notebook modeは`aggregate`固定、2 shardの全必須SHAは64文字で設定済み。実行量は
  HMM variant / LightGBM config / fold / booster 0 / 0 / 0 / 0。

### SHA-fixed aggregate version 4 push

- push時刻: 2026-07-20 11:31 UTC / 20:31 JST。
- canonical kernel `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` / id_no
  `127594551`へversion 4を`kaggle kernels push --timeout 43200`でpushした。
- push直後のstatusは`KernelWorkerStatus.RUNNING`。aggregateは保存済み2 shardの結合と評価のみで、
  HMM / LightGBM / fold / boosterは0 / 0 / 0 / 0。継続監視は行わず、完了連絡後にlogs、metrics、
  coverage、direct/oracle readout、最終SHAを監査する。

### SHA-fixed aggregate version 4 completion audit

- 2026-07-20にcanonical aggregate kernel
  `kentookumura/exp270-exact-hmm-posterior-mode-audit-train` / id_no `127594551`のversion 4が
  `KernelWorkerStatus.COMPLETE`となったことを確認し、logsとKaggle outputを取得した。
- aggregateは3,783,989 rows / 773 wellsを156.241秒、peak RSS 3,097.277 MBで結合・評価した。
  shard 0 / 1読込後のunionは1,792,363 + 1,991,626 rows、363 + 410 wellsで、重複や欠落はない。
- exp209 posterior-mean parityはmax / mean absolute difference 0.0 / 0.0 ft、tolerance
  0.00001 ft、全3,783,989 rowsでPASSした。
- candidate gzipをchunk再走査し、ID=`<well>_<row_idx>`、well/row strict order、posterior mean / marginal MAP /
  top-1 finite、top-2からtop-5の欠損数603,167 / 821,989 / 909,299 / 970,358、
  oracle/selector/blend列不在を確認した。欠損数はunique pathが5未満のno-backfill契約と一致する。
- prediction array-bundle SHAをcandidate float32 matrix + int64 row indexから再計算し、
  `d42a492bc4065ac88b481d9dba1e24b4c3b4ab331c93fce73a433d592aa16f19`でsummaryと一致した。
- candidate gzipは134,101,285 bytes、raw SHA
  `bef798c79e902faf93bf8ef5e75c6f868722fc8c4bca61c6dfd10778cfb4d520`、decompressed SHA
  `986276e2495b23d5de2425542efd0433c59f1a398c073b3053218ba9007a5ecf`で一致した。
- decoder manifest / candidate schema / input manifest SHAは
  `863ccb7ec073e2b624fb59d43ed91a1604cf6a7d17e7a345ab9ccec23a9b9e52` /
  `8c75d1fd4e71e81496f6384ea49787dba1dff16329bf0048dc8c65019798006d` /
  `6ee0a89c40bd505631ce431d7f4f8452576da00fca266af31565a3a333de3a41`。
  summary記録の13 artifact SHAを取得実ファイルから再計算し、すべて一致した。
- downloaded `metrics.json` SHAは
  `1332971e26c5967dc19f57a908d0a888939426fda0f72492604f30b17272bdc5`、summary SHAは
  `5a467f283204809e67851df291c8c1b979d8a767b60b6588cd35ea7a54cb06fa`。
- `kaggle kernels files`のsize列は引き続き誤表示だったため根拠にせず、取得outputの実byte数とSHAを正とした。

### Scientific readout and decision

- direct RMSEはposterior mean `11.9382872349`が最良。marginal MAPは`12.5924792323`
  （`+0.6541919974 ft`）、global Viterbiは`15.5516646970`（`+3.6133774621 ft`）だった。
  top-2 / 3 / 4 / 5もcoverage `0.840600 / 0.782772 / 0.759698 / 0.743562`で、RMSE
  `15.302589 / 15.157698 / 15.140188 / 14.674071`とすべてposterior meanを悪化させた。
- marginal MAPは206 / 773 wellsを改善したがwell差中央値は`+0.210160 ft`、worst regressionは
  `+13.196777 ft`。global Viterbiは205 / 773 wells改善、中央値`+3.467042 ft`、worst well
  `86454a6f`で`+55.179043 ft`だった。
- verification-like spatial / typewell-purgedでもposterior mean `12.564491 / 12.367244`が最良で、
  marginal MAP `13.406237 / 13.217184`、global Viterbi `16.989852 / 16.699334`は悪化した。
- all-7-mode oracleはrow / block128 / block256 / block512 / wellで
  `7.516850 / 7.567530 / 7.608996 / 7.685922 / 8.536362`。ただしmean / MAP / Viterbiの
  3候補だけでも`7.517189 / 7.567871 / 7.609328 / 7.686242 / 8.536605`で、top-2からtop-5の
  追加oracle価値は最大`0.000342 ft`だった。paths-only oracleも約`15.549`で弱い。
- unique TVT path数は平均4.160414、最小1、5本未満192 wells。row unique-bestはposterior mean
  37.5834%、marginal MAP 33.8878%、global Viterbi 4.0647%、well unique-bestは410 / 166 / 61 wells。
  これらはtrue TVTを使う事後診断で、target-free selector evidenceとは扱わない。
- focus well `11d0f5ac`はposterior mean `21.160939`、marginal MAP `21.218470`、global Viterbi
  `20.793050`、row / block128 oracle `20.238791 / 20.267534`。局所改善はglobal悪化を覆さない。
- oracle prediction、selector、blend、inference、submissionは生成していない。technical contractは
  PASSしたため、実装失敗ではなくdirect mode-candidate仮説のnegative resultとしてbranchを閉じる。
- oracleだけを根拠にexp270固有の救済backlogは追加しない。既存高優先実験の順位は維持し、
  candidate追加、top-K拡張、selector学習、raw-test inference、submissionへ進まない。

### Final record validation

- root `metrics.json`のJSON parse、root configとのstatus一致、prediction SHA、summary row、
  `KAGGLE_DIRECTION.md`実装待ち表からの削除を再確認した。
- `make validate-exp EXP=exp270_exact_hmm_posterior_mode_candidate_audit`は
  `experiment validation passed (strict)`、`make validate-template`はPASS。
- `review_exp_docs.py exp270 --root .`はcore evidence categoryが全項目存在すると判定した。
- `make update-summary`で296 experimentsを再生成し、exp270 status / CV / parentを
  `completed_train_side_direct_negative_oracle_diagnostic_only_no_inference_no_submit` /
  `11.938287234887435` / `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`へ反映した。
