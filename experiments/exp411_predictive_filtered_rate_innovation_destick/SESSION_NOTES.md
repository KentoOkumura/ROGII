# exp411_predictive_filtered_rate_innovation_destick セッションノート

## 目的

exp408で主因と確定したforward rate-prior hysteresisに対し、predictive→filtered rate
innovationだけで発火する方向付きde-stickを、単一変更として設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 Version 5完了・`stage0_fail_closed`・branch close
- 優先度: P2
- CV / LB: まだなし
- inference / submission: 無効
- 実装承認: あり（ユーザー指示 `exp411を実装してください`）
- Kaggle実行承認: Stage 0のみあり（ユーザー指示 `実行してください`、2026-07-27）

## 2026-07-26 設計確定

ユーザーの「第一案と第二案のバックログ、実験ディレクトリ、steeringを作成して
設計を確定。実装はまだ」という依頼により、exp411を第一案として採番した。

### 根拠

- exp408 exclusive forward cause SSE: `59.3978%`
- filtered zero-directed rate under-response SSE: `70.3580%`
- transition displacement errorとoffsetのSSE加重符号一致: `90.2246%`
- exp338 global / well-level `sig_r=0.004`化: `+2.124061 ft`、0/5 folds
- exp268 initial-rate window best gain: `0.042706 ft`
- exp370 GR-change / ESS reset trigger: 退化

### 固定した変更

- `u=(mu_filtered-mu_predictive)/0.005`
- two-sided CUSUM drift allowance: `0.01`
- trigger threshold: `1.0 rate cell`
- activation: 次の32 transitions
- stay mass transfer: `10%`
- refractory: 128 rows
- edge: outward neighborがないsource stateだけno-op
- active中の重複trigger / direction flip: 不可

### 実行量

Stage 0予定:

- treatment variants: 1
- target wells / HMM well-runs: `32 / 32`
- parent control rerun: 0
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

Stage 1予定:

- treatment variants: 1
- target wells / HMM well-runs: `773 / 773`
- parent control rerun: 0
- reporting folds: 5
- model / booster / PF / Beam / GPU: 0

Stage 0は実装済み・実行未承認、Stage 1は未実装・未承認。

## 2026-07-26 Stage 0実装候補

### fixed32 manifest

`build_stage0_manifest.py`で、exp408 persistent scope、SHA固定されたexp226 fold identity、
rawのtarget-freeなprefix / suffix row数とsuffix raw-GR欠損率だけを読んで固定した。
truth、error、episode列はmanifest生成に使っていない。

- rows: 32
- role: persistent 16 / control 16
- fold counts: `8 / 6 / 6 / 6 / 6`
- exact quartile match: 9 / 16 controls
- maximum quartile Manhattan distance: 2
- manifest SHA256:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- exp408 persistent scope SHA256:
  `ce245abce24dae98d37b6e0a2adf73fa57a29e0e53864bee983aa916238ea51e`
- exp226 decompressed SHA256:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`

### HMMとschedule

- exp209と同じposition grid/kernel、41 rate states、`sig_r=0.002`、`sig_p=0.02`、
  `mom=0.998`、Gaussian GR emission、prior、posterior-mean readoutを持ち込んだ。
- 各rowのemission前predictive rate meanとemission後filtered rate meanから
  `u=(mu_filt-mu_pred)/0.005`を計算する。
- CUSUMは全suffix rowで更新し、active / refractory中は再triggerだけを禁止する。
- trigger row自身はparent transitionのまま、次のrowから32 transitionsだけ
  stay massの10%を方向隣接stateへ移す。
- rate-grid外向きedge sourceはno-op。
- forwardで確定した`active_direction[t]`をbackwardの同じtransition indexで再利用する。
- future directionは、物理interval rate `(dTVT+dZ)/dMD`のpast 32-row medianと
  future 32-row medianの差として固定した。

### notebook構成

- train候補:
  `exp411_predictive_filtered_rate_innovation_destick_compact_selfcontained_train.py/.ipynb`
- inference候補:
  `exp411_predictive_filtered_rate_innovation_destick_compact_selfcontained_inference.py/.ipynb`
- train候補は9章・2,227行。比較元exp408 compact trainは10章・2,483行であり、
  exp411固有のmanifest、de-stick kernel、truth-late gate、生成物保存まで同等の
  self-contained粒度を保つ。
- 同一exp helper importと`__file__`依存はない。
- 正規Notebookは既存placeholderを上書きせず、採用判断待ち。

### 静的検証

- Jupytext train / inference `--test`: PASS
- `py_compile`: PASS
- Ruff F821: PASS
- 専用pytest: `13 passed`
- exp408回帰を含むpytest: `21 passed`
- `task validate-exp`はtask runner未導入のため実行不可。規定どおり
  `make validate-exp EXP=exp411_predictive_filtered_rate_innovation_destick`へfallbackし、
  strict experiment validation PASS
- 独立したexp209 `_hmm2_fb`との小trellis untreated parity:
  posterior `atol=2e-7`、log-likelihood `<=2e-6`でPASS
- 同一kernelのno-trigger treatment path:
  posterior mean / log-likelihood差`<=1e-10`でPASS

### 実行コスト契約

- active treatment variants: 1
- Stage 0 HMM well-runs: 32
- saved parent control HMM rerun: 0
- LightGBM configs / trained folds / boosters / models: `0 / 0 / 0 / 0`
- PF / Beam / GPU runs: `0 / 0 / 0`
- `design.stage_0_execution_approved=false`
- `execution.kaggle_execution_authorized=false`

## 2026-07-27 Stage 0実行承認

ユーザーの `実行してください` を、直前に提示した正規Notebook採用とStage 0 Kaggle CPU
実行の承認として記録した。承認範囲はStage 0だけであり、Stage 1、inference実行、
submissionは含まない。

push前の実行量を再確認した。

- active treatment variants: 1
- Stage 0 HMM well-runs: 32
- saved parent control HMM rerun: 0
- LightGBM configs / trained folds / boosters / models: `0 / 0 / 0 / 0`
- PF / Beam / GPU runs: `0 / 0 / 0`
- runtime: Kaggle private CPU、internet off
- initial canonical candidate:
  `kentookumura/exp411-predictive-filtered-rate-innovation-destick-train`
- initial title:
  `exp411 predictive filtered rate innovation destick train`

credential checkerはAPI token未設定を報告したが、Kaggle CLI用OAuth credentialと
legacy credentialは利用可能であることを確認した。token実値は表示していない。

明示承認に基づきJupytext percent sourceから正規Notebookを生成し、既存placeholderを
置き換えた。候補ファイルは比較・復旧用に維持した。

- canonical train SHA256:
  `471d978cd2adc1198500490e0a2f8fbf83e4316897a14fad5aca2d5c85d7ff6a`
- canonical inference SHA256:
  `b41bfef949de7a71dd2fbcf1d59b9a8afe3703ea772dbd6001eadcc27789304b`

push前検証:

- Jupytext train / inference `--test`: PASS
- `py_compile`: PASS
- Ruff F821（source / contract test）: PASS
- exp408回帰込みpytest: `21 passed`
- strict experiment validation: PASS
- strict Kaggle package preparation: PASS
- initial 56-character-name packaged notebook SHA256:
  `04fd594dd8440b8b38285fa686d12744b5b0b780b6f831e9b4491a500ad9b172`
- metadata: private / CPU / internet off / run-on-push
- canonical `id`末尾slugとtitle由来slug: 一致
- parent kernel source:
  `kentookumura/exp209-joint-exact-parity-train`
- bootstrap support manifestでfixed32 manifest、metadata、persistent episode、
  loose configのbyte数とSHAを固定し、strict照合を通した。

### 初回pushのmetadata 400と復旧

- 初回canonical候補はslug / titleとも56文字で、Kaggle `SaveKernel` の詳細なし400を
  返した。
- 同じidのmetadata pullは403で、kernelが未作成であることを確認した。
- 親source `kentookumura/exp209-joint-exact-parity-train` はmetadata pullに成功した。
- 既存exp405で確認済みのKaggle kernel名50文字制約に合わせ、科学契約、Notebook、
  configの実行量、runtimeを変えず、`predictive`だけを`pred`へ縮めた50文字の
  `exp411-pred-filtered-rate-innovation-destick-train`へid / titleを同時にそろえて
  再packageする。
- 50文字id / titleでstrict experiment validationとstrict package validationを
  再度通した。
- final packaged notebook SHA256:
  `00a4d6dd243619e0d7b5228e18771db81a7c81a59227af063356f374d38ad1f5`
- final packaged config SHA256:
  `34bea6e7f284087535bfbd21df08e86ccc90e61eb32a330c03350345e70b8736`

### Kaggle Stage 0 Version 1

- push result: `Kernel version 1 successfully pushed`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp411-pred-filtered-rate-innovation-destick-train`
- Kaggle kernel id_no: `128773391`
- pull後metadata:
  private / CPU / TPU無効 / internet無効 / `machine_shape: None`
- pull後source:
  competition + `kentookumura/exp209-joint-exact-parity-train`
- 最終状態: ERROR
- 原因:
  exp209保存予測の実ファイルdecompressed SHAはexp209記録と一致する64桁
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
  だったが、exp411 configへの転記時に`8b`が欠けた62桁値を入れていた。
- 影響:
  support bootstrap、実行量、親source解決、実行承認確認まではPASSし、HMM実行前の
  SHA fail-closedで停止した。Stage 0 scientific resultは生成されていない。
- 修正:
  configの2箇所をexp209記録と同じ64桁SHAへ直し、64桁literal一致のcontract testを
  追加した。科学契約、fixed32、実行量、CPU設定は変更しない。

Version 2 push前再検証:

- exp408回帰込みpytest: `22 passed`
- Ruff F821: PASS
- strict experiment validation: PASS
- strict Kaggle package validation: PASS
- packaged notebook SHA256:
  `4102ade5cd893678f273aaa15e0de62eaca1f6f3685b91525750e3f3acae8f13`
- packaged config SHA256:
  `a11af294365e8dfe723e94b2ec57a8f48f3d5f2b2d4fc97ddee765a02a505da5`

### Kaggle Stage 0 Version 2

- push result: `Kernel version 2 successfully pushed`
- kernel id / URLはVersion 1と同じ。
- scientific contract / fixed32 / execution counts / CPU metadata: Version 1から不変
- 最終状態: ERROR
- 原因:
  exp209保存予測cacheは`META_COLUMNS = [id, well, target]`で、独立した`row_idx`列を
  持たない。exp411 loaderが存在しない`row_idx`を`usecols`へ要求して停止した。
- 影響:
  正しい親SHAの検証はPASSしたが、親cache読込時点で停止した。HMM実行とStage 0
  scientific resultは生成されていない。
- 修正:
  `target`は引き続き読まず、exp209生成コードの`id = f"{well}_{int(row_idx)}"`契約に
  従って、exact well prefixを検証したうえでid suffixからrow indexを復元する。
  well名にunderscoreがあっても末尾splitへ依存しない。正常・prefix不一致・非数値suffix
  のcontract testを追加する。

Version 3 push前再検証:

- canonical train Notebook SHA256:
  `0992ec0f306ca190fe3cb783c591541c106d91623811babd8918348a7f65af7d`
- exp408回帰込みpytest: `23 passed`
- Jupytext / py_compile / Ruff F821: PASS
- strict experiment validation: PASS
- strict Kaggle package validation: PASS
- packaged notebook SHA256:
  `791b371018637f7273af059a18c8734821db5f550f65e0d57b034e807fb15f09`
- packaged config SHA256:
  `31e94a2137b7347d6da781aa3828e414d38279cd7fc46034f51ff6409c4ea76c`

### Kaggle Stage 0 Version 3

- push result: `Kernel version 3 successfully pushed`
- kernel id / scientific contract / fixed32 / execution counts / CPU metadata: 不変
- 最終状態: ERROR
- 原因:
  competition raw horizontal CSVは
  `MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input`で`id`列を持たない。
  target-free読込で`TVT`を除外した後、HMM input preparationが存在しない`id`を参照した。
- 影響:
  parent cacheのSHA、schema、row index復元はPASSし、最初のfixed32 wellのHMM input
  preparationで停止した。HMM処理とStage 0 scientific resultは生成されていない。
- 修正:
  raw DataFrameのRangeIndexをexp209と同じrow indexとして使い、
  `f"{well}_{row_idx}"`を明示生成して親cache idと照合する。rawに`id`を要求しない。
  実raw headerを確認し、id列なしの最小HMM input preparationとunderscoreを含むwell名の
  cache id生成をcontract testへ追加する。

Version 4前の追加監査で、全32予測freeze後のtruth-late loaderにも同じraw `id`誤認が
残っていたため、実行前に修正した。truthは従来どおりfreeze後だけ読み、RangeIndexから
cache idを再構成してfrozen idと照合する。truth-late順序を変えていない。well metric
appendはsource上で1回だけであることもcontract testで固定した。

Version 4 push前再検証:

- canonical train Notebook SHA256:
  `f19bd0430903cf88ceb2505604b733b89f18ef10b614e0a019b76b1390dd9b86`
- exp408回帰込みpytest: `25 passed`
- Jupytext / py_compile / Ruff F821: PASS
- strict experiment validation: PASS
- strict Kaggle package validation: PASS
- packaged notebook SHA256:
  `4702ba9eb8034f98914341a30fc41606059b44a38afa7b4328d625826a6179a8`
- packaged config SHA256:
  `31e94a2137b7347d6da781aa3828e414d38279cd7fc46034f51ff6409c4ea76c`

### Kaggle Stage 0 Version 4

- push result: `Kernel version 4 successfully pushed`
- kernel id / scientific contract / fixed32 / execution counts / CPU metadata: 不変
- 最終状態: ERROR
- 進捗:
  32/32 HMM well-runsは完走した。HMM elapsedは`1009.643139 s`、peak RSSは
  `1.020450592 GiB`。各wellでtrigger / active rowsが生成され、HMM本体の例外はない。
- 原因:
  activation scheduleのgzip書込後に、pandas既定float parserでreadbackすると
  浮動小数の末尾桁が変わり、logical SHAが不一致になった。
- 診断:
  Version 4 outputを`/tmp`へ取得して確認した。schedule decompressed SHAは
  `fd27809c4b07183ec63d8d57f6154032884314e6a7bca9996fc021a7c14a34a7`。
  既定parserの再serialize SHAは不一致だったが、`float_precision="round_trip"`では
  decompressed bytesと17,343,274 bytesすべて一致し、同じSHAになった。
- 影響:
  truth / episode読込とStage 0 gate評価の前に停止したため、scientific resultはまだ
  確定していない。Version 4でHMM 32 well-runsを実行済み。
- 修正:
  deterministic gzip readbackだけを`float_precision="round_trip"`へ変更し、
  Version 4の実値を含むfloat round-trip SHA contract testを追加する。
- Version 5のper-run契約:
  1 treatment / 32 HMM well-runs / parent rerun 0 / LightGBM・fold・booster・model
  `0 / 0 / 0 / 0` / PF・Beam・GPU `0 / 0 / 0`。
  retryを含む実績累計HMM well-runsはVersion 5完走時に64となる。

Version 5 push前再検証:

- canonical train Notebook SHA256:
  `7519b50702db30086c8adc903760d96a8d5ea6b6458a0b8fb3eaf1987f057a3d`
- exp408回帰込みpytest: `26 passed`
- Jupytext / py_compile / Ruff F821: PASS
- strict experiment validation: PASS
- strict Kaggle package validation: PASS
- packaged notebook SHA256:
  `460df29ba707f6c528ec49dc36856a7761562adc00a09fdccf3ccb4a4722940c`
- packaged config SHA256:
  `31e94a2137b7347d6da781aa3828e414d38279cd7fc46034f51ff6409c4ea76c`

初回Version 5 pushはKaggleの`Maximum batch CPU session count of 5 reached`で
拒否され、Version 5は作成されなかった。占有中の5枠はexp413が1件、exp402 fold0–3が
4件で、すべてRUNNING。他実験は停止せず、1枠解放後に同じ検証済みpackageをpushする。
CPU枠待ちstatusをlocal configへ反映後、loose / bootstrap一致を保つためstrict packageを
再生成した。

- final waiting package notebook SHA256:
  `94ebd2536cfc77b8848e95fdf702078c1ef3185e9b178a2a8b05ae509d4be578`
- final waiting package config SHA256:
  `ac6082f3af033497cf165bdd8562d0b7515a1fb7f3e2f5c60f13ae17e58bd3aa`

### Kaggle Stage 0 Version 5再開

- 再開時status:
  exp413、exp402 fold0/2/3はCOMPLETE、exp402 fold1だけRUNNINGで4枠解放済み。
- 再確認:
  OAuth credential利用可能、26 tests PASS、strict experiment validation PASS、
  package SHA不変、canonical id_no `128773391`、private CPU / internet off、
  parent source不変。
- push result: `Kernel version 5 successfully pushed`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp411-pred-filtered-rate-innovation-destick-train`
- per-run contract:
  1 treatment / 32 HMM well-runs / parent rerun 0 / LightGBM・fold・booster・model
  `0 / 0 / 0 / 0` / PF・Beam・GPU `0 / 0 / 0`
- 最終状態: `COMPLETE`

### Kaggle Stage 0 Version 5結果（2026-07-28回収）

- Kaggle status: `KernelWorkerStatus.COMPLETE`
- kernel id_no: `128773391`
- 実行: 1 treatment / 32 HMM wells、parent rerun / LightGBM / fold / booster /
  model / PF / Beam / GPUはすべて0
- 32 / 32 wells完走
- elapsed: `1,133.132777 s`
- peak RSS: `1.020561 GB`
- full runtime projection: `27,372.238645 s`
- retryを含む累計HMM well-runs: Version 4の32 + Version 5の32 = 64

technical gateは13 / 13 PASSした。finite coverage `1.0`、maximum normalization
error `2.797762e-14`、no-trigger parent parityとzero-active saved-parent parityは
最大差`0.0 ft`、schedule readback SHA、truth-late順序、fixed32 role / fold、
runtime / RSS guardもすべてPASSした。

mechanism gateは2 / 6 PASSで`stage0_fail_closed`:

- future-rate direction agreement:
  `0.2253968254 < 0.60`、FAIL
- per-fold agreement:
  `0.281879 / 0.201681 / 0.260417 / 0.236486 / 0.135593`、
  strict PASS `0 / 5 < 4 / 5`、FAIL
- pre-onset trigger coverage:
  `1.0 >= 0.50`、PASS
- eligible lead-time episodes:
  `25 >= 8`、PASS
- control active-row fraction:
  `0.136118774 > 0.10`、FAIL
- persistent minus control active-well fraction:
  persistent `16 / 16`、control `16 / 16`、差`0.0 < 0.20`、FAIL

fixed32 RMSEはparent `9.968802828 ft`、treatment `9.972554480 ft`、
delta `+0.003751652 ft`。これは事前契約どおりdiagnosticでpromotion gateではない。

実ファイル確認が必要なSHA / readoutを監査するため、Kaggle outputを
`/tmp/kaggle-output/exp411_predictive_filtered_rate_innovation_destick/train_v5`
へ取得した。主要照合結果:

- activation schedule: 156,088 rows、raw
  `853a0920...804a`、decompressed / logical
  `fd27809c...34a7`
- predictions: 156,088 rows、raw `c95746ea...42a1`、
  decompressed / logical `5f6e469c...8c0d`
- well metrics: 32 rows、`4a8e9842...5e91`
- trigger truth-late readout: 633 rows、`a0981988...0d48`
- episode lead readout: 25 rows、`3d2a6b8b...c963`
- summary: 6,805 bytes、`5c952e7e...42b3`
- input manifest: 1,387 bytes、`12d6bc0e...a1de`

schedule / predictionのgzip raw SHA、decompressed SHA、readback logical SHA、
非圧縮readout SHA、CSV行数はすべてVersion 5ログと一致した。

結論としてtriggerは早いが、future rate方向と一致せず、controlにも同程度に発火する。
technical failureではなく固定した科学仮説のnegative resultとして信頼できる。
`promotion_eligible: false`のためStage 1を実行しない。inference / submissionも
未実行のままbranchを閉じる。

最終記録検証:

- Kaggle output `metrics.json`とlocal `metrics.json`のresult / gate / artifact
  evidence完全一致: PASS
- exp408→exp411 / exp411→exp408の両pytest収集順: 各`26 passed`
- exp408 testのNumba stubへ`ModuleSpec`を追加し、科学コードを変えず収集順依存を解消
- exp411 / exp412 / exp420 strict experiment validation: PASS
- `make update-summary`: 416 experiments更新

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- RNGなし。well / row sort、CUSUM更新、pass順を固定する。
- gzipはdecompressed content SHAを主証拠にする。
- sample、scientific contract、trigger schedule、prediction、metrics SHAを保存する。
- deterministic submission anchorではなく、submissionは生成しない。
- 実装後のKaggle push前にloose / bootstrap configとNotebook bodyを照合する。

## 未実施

- Stage 1 full 773-well HMM
- inference / submission

## 次のアクション

exp411は追加実行しない。exp412はcausal trigger / future evidence不足という先行条件を
満たしたが、設計確定・未実装のままであり、実装と各Kaggle実行は別指示を待つ。
同じscheduleとdirection / control gateを使うexp420は、現行契約のままStage 0を
runせず、実装参照として保持する。
