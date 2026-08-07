# exp498_geometry_mean_reversion_tail_regime_physics_readout セッションノート

## 目的

exp490の平均改善とwell-tail悪化の二極化を、予測再生成なしに観測可能な物理regimeで
説明できるか監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle version 2完了、technical PASS / physics-regime FAIL、terminal close
- CV / LB: なし
- 実装承認: 2026-08-01の追加依頼で承認
- Kaggle package / run承認: 2026-08-01の「実行してください」で承認
- inference / submission: 範囲外

## 2026-08-01 設計記録

- 最新番号exp497の次としてexp498を採番した。
- ユーザー指定名を`exp498_geometry_mean_reversion_tail_regime_physics_readout`とした。
- KAGGLE_DIRECTIONの既存P2 backlogを具体化し、親をexp490、routeを`pf_beam`に固定した。
- exp490 merge v1 prediction / by-well / episode / K16 segment / well manifestと
  scientific contractのSHAを固定した。
- rawはexp490 shard input manifestとSHA一致するhorizontal / typewellだけを許可し、
  horizontal suffix truth `TVT`はusecols段階で禁止した。
- truth-late 2 phase、7物理量の絶対bucket、単一primary regime
  `weak_gr_geometry_conflict`を固定した。
- physics支持はcoverage、harm rate ratio、fold一貫性、mean delta、catastrophic captureの
  6項目all-ANDとした。
- primary FAIL後のbucket統合、interaction探索、閾値緩和、same-OOF救済は禁止した。

## 実行量契約

現在は全項目0。別承認時の予定は次のとおり。

| 項目 | 予定 |
| --- | ---: |
| readout | 1 |
| target-free well aggregation | 773 |
| truth-late fold readout | 5 |
| HMM well-run | 0 |
| new prediction | 0 |
| model config / trained fold / booster | 0 / 0 / 0 |
| PF / Beam / GPU | 0 / 0 / 0 |

親controlやexp490を再実行しない。CPU saved-artifact diagnosticだけである。

## 再現性メモ

- `docs/06_reproducibility.md`を確認した。
- RNGなし、well / row / segment / bucketをstable sortする。
- exp490 gzip predictionは展開後content SHAを主証拠にする。
- feature schema / logical content / contract / summary SHAを予定生成物へ記録する。
- model / prediction / submission SHAは対象外で、count 0をmanifestへ記録する。
- Kaggle private CPU / internet offでのpackage / runは2026-08-01に承認された。
- rerun一致までdeterministic diagnosticとは呼ばない。

## notebook状態

正規train notebookはJupytext percent形式のcompact self-contained sourceから生成する。
正規inference notebookは範囲外のためMarkdown 1 cell、code cell 0のplaceholderを維持する。
Kaggle packageは実行承認反映後に作成する。

## 2026-08-01 実装記録

- 固定済み設計を変えず、`exp498_geometry_mean_reversion_tail_regime_physics_readout_train.py`
  にsource解決、SHA検証、Phase A feature freeze、Phase B truth-late readout、固定all-AND、
  planned生成物保存をself-contained実装した。
- raw file SHAの正本がexp490 shard `input_manifest`ではなく`decoder_manifest`内にあることを
  実生成物で確認した。4 decoder manifest自体のSHAをconfigへ固定し、重複なし773 wellの
  `horizontal_sha256` / `typewell_sha256`へ結合する実装に補正した。
- Phase Aはpredictionの12 safe columnsだけを250,000行chunkで読み、well境界をまたぐ
  chunkを連結してwell単位に集約する。horizontalは`usecols=[TVT_input, GR]`で`TVT`を
  読まず、typewellは`TVT, GR`だけを読む。
- exp490と同じknown-prefix GR residual std + `[10,60]` clipを使い、visible-prefix TVT上の
  typewell GR p95-p05をsigmaで割ったinformation ratioを追加した。
- feature CSV / logical SHA / schema / contract SHAを保存してledgerをfreezeした後だけ、
  prediction fold、saved by-well outcome、episode outcomeを読む。
- 合成fixtureでsafe-column、raw manifest 773 well、bucket端点、primary式、truth-late遮断、
  sparse-regime fail-close、all-AND PASS例、chunk跨ぎ集約、canonical notebook構成を検証する
  empty bucket保持とsaved-outcome schemaを検証する13 testsを追加した。
- 実装時点のplanned executionはreadout 1 / well aggregation 773 / folds 5、HMM / prediction /
  model / booster / PF / Beam / GPU 0。親control再実行はない。
- exp490 compact trainは9章 / 2,399行、exp498 trainは同じ役割を満たす9章 / 1,457行。
  exp498はdecoder/HMM生成を持たないreadout専用なので短いが、source/SHA、chunked input、
  raw物理量、freeze、truth-late、gate、生成物、guarded orchestrationの章をすべて持つ。
- canonical train notebookは21 cells（Markdown 11 / code 10）、output 0へ変換した。
- 固定入力のread-only SHA解決はmerge 5生成物、decoder manifest 4件、raw SHA contract
  773 wells、scientific contract `6398bbac...9a35`をPASSした。
- `task` / system `python`は環境に存在しなかったため、規定fallbackの
  `make validate-exp` / `.venv/bin/python`を使用した。strict experiment validation、
  template validation、文書reviewのcore evidenceをPASSした。

## 設計検証

- strict experiment validation: PASS。
- config YAML / metrics JSON parse: PASS。
- execution guard: readout承認後だけrunし、inference / submissionはfalseを維持した。
- notebook guard: train 21 cells（Markdown 11 / code 10、output 0）、inference code 0。
- experiment summary更新: exp498とexp490からのlineageを記録済み。
- experiment docs review: core evidence categoriesあり。

## 2026-08-01 実行承認

- ユーザーの「実行してください」をKaggle private CPU package / scientific readoutの承認として
  記録した。inference / submissionは引き続き範囲外である。
- 実行量を再確認した。scientific readout 1、target-free well aggregation 773、truth-late
  fold readout 5で、HMM well-run / new prediction / model config / trained fold / booster /
  PF / Beam / GPUはすべて0である。
- 親controlとexp490は再実行しない。
- Kaggleの50文字slug制約に合わせ、canonical kernel IDを意味を保った48文字の
  `exp498-geometry-mean-reversion-tail-regime-train`とした。
- package前testで、未実行時には未到達だった実行ガードが存在しない
  `execution.kaggle_run_approved`を参照することを検出した。承認の正本である
  `implementation.kaggle_run_approved`へ修正し、承認済み契約testへ更新した。

## 実行前アクション記録

事前検証後、Kaggle private CPU / internet off / private / run-on-push packageを監査し、
canonical slugへ1回pushする。完了後に6生成物とscientific gateを回収する。

## 2026-08-01 Kaggle package監査

- package: `experiments/exp498_geometry_mean_reversion_tail_regime_physics_readout/kaggle/train`
- kernel: `kentookumura/exp498-geometry-mean-reversion-tail-regime-train`
- metadata: private / CPU / TPU off / internet off / run-on-push。
- source: competition 1件、exp490 merge + shard0..3のkernel 5件。
- packaged notebook: 22 cells（bootstrap 1 + canonical 21）、code 11、output 0。
- packaged configはlocal configとbyte一致し、readout 1 / well aggregation 773 /
  fold readout 5、HMM / prediction / model / trained fold / booster / PF / Beam / GPU 0。
- config SHA256: `2dd331b73052ee48f81bdc2f7f6ecdb4790e20c82d7ff8aeb0e5fb335682db86`
- canonical notebook SHA256:
  `ae8970e1bb6d10de7843025beb4648667266c39408a74c91de852fe4a4f4a5cb`
- packaged notebook SHA256:
  `0a4cf5554930466cf7c0ae68a1d33a68feb62b0e5808c6a1af5e35165fa06407`
- kernel metadata SHA256:
  `cce7f6976b7167e08a9bf8f61afc736d06fe288e66c233054c84a24873c5d01f`

## 2026-08-01 Kaggle version 1 技術FAIL

- push result: version 1、Kaggle `id_no=129328553`、private CPU / internet off。
- status: `KernelWorkerStatus.ERROR`、約92秒で終了。
- 入力SHA確認とPhase A feature freeze後、truth-late joinで`KeyError: 'fold_x'`となった。
- 原因はmergeで`suffixes=("_manifest", "_outcome")`を明示したのに、fold parity確認だけ
  pandas既定の`fold_x` / `fold_y`を参照していた実装不整合である。
- `fold_manifest` / `fold_outcome`へ修正し、fold一致と不一致fail-closeの回帰testを追加する。
- scientific gateへ到達せず、resultは未判定。固定bucket、primary式、gate、入力、実行量は
  変更しないため、修正版を同じslugのversion 2として実行する。
- 修正後は回帰testを含む14 tests、Ruff F821、py_compile、Jupytext round-trip、
  strict experiment validationをPASSした。
- version 2 packageはprivate / CPU / internet off / run-on-pushと5 kernel inputsを維持し、
  packaged codeに`fold_manifest` / `fold_outcome`修正が入り、`fold_x`がないことを確認した。
- version 2 canonical notebook SHA256:
  `b346c53de6e019114a3bd520797fd4030d4db64a3e55f0d7cef9f8a8ad233365`
- version 2 packaged notebook SHA256:
  `c12488ac26e4f0a89fad5ad086906747df575091eede50639f8c0ded91d909cd`
- version 2 train source SHA256:
  `09851281a2485c895c370d8f8940fc90c323b9dd77b9607443df4d3b9fbeaa41`

## 2026-08-01 Kaggle version 2 結果

- status: `KernelWorkerStatus.COMPLETE`。
- kernel: `kentookumura/exp498-geometry-mean-reversion-tail-regime-train`
- Kaggle id_no / version: `129328553 / 2`。
- runtime: private CPU / internet off、`76.685304 sec`、peak RSS `0.360542 GiB`。
- actual execution: readout 1 / rows 3,783,989 / well aggregation 773 / folds 5。
  HMM / prediction / model / trained fold / booster / PF / Beam / GPU / inference /
  submissionはすべて0。
- technical checksは全PASS。fixed input SHA、row / well identity、finite feature、bucket、
  horizontal suffix truth read 0、pre-freeze outcome read 0を確認した。
- primary `weak_gr_geometry_conflict`は0 / 773 wells。weak observationは359 wells、
  geometry disagreement `>=10 ft`は0 wells（最大`5.337991 ft`）、early abs offset
  `>=5 ft`は1 well（最大`5.037076 ft`）で、3条件同時成立は0だった。
- coverage / fold support、pooled/fold harm方向、pooled/fold mean delta方向、bounded-coverage
  catastrophic captureの6 checksはすべてFAIL。51 catastrophic wellsのcaptureは0。
- complement 773 wellsではharmful 211（27.2962%）、mean / median delta RMSE
  `-0.769496 / -0.057105 ft`。persistent 638 episodesのSSE reductionは
  `41.409965%`で、exp490のpooled improvementを再現したがprimary FAILを救済しない。
- decision: `terminate_mean_reversion_tail_regime_cause_tracking`。geometry / early-offset
  thresholdを緩和せず、secondary bucket / interaction / same-OOF rescueを行わない。
  exp490 terminal closeを維持し、復元力を弱める後続式、inference、submissionは作らない。
- 完了後の`execution.run_readout`、`kaggle_push_approved`、`run_on_push`はfalseへ戻し、
  未承認rerunをfail-closeした。version 2 package metadataは実行証跡として保持する。

## version 2 生成物

- output: `kaggle/output/train_v2`
- input manifest SHA: `450e2cfba697cbe700338b3b5e430bed04bf01e72c659ec0ab81121ab4c93ae2`
- feature contract file / logical SHA:
  `ccdc3d5d5233546a350385f6f5f8b5d1ac351488bb3c62acd50af20df37d55c1` /
  `92d1e78a197a9726640ea049891a6081e30784b98f3faa0b8ab113af8eb2416c`
- feature content SHA: `c1d31113e7247ada9be0d6fd1e183808f7fbc04af256612481363ee900e0f5ad`
- by-fold / bucket SHA:
  `eaa7cea3b0a6cd0164495ddf85573f7e28c219787b7ded9da78f2fa0394357fb` /
  `c1a0e5ebb69ac95dfd4af3fda6e421caeb99d02b03971aefffb4b1c91039a406`
- summary / Kaggle metrics SHA:
  `485964967acf3a6cc913eaeba6ebaa458384d37090bb0f71d1a16e1f298c1df7` /
  `74a02ecbf176ed68b76f906c6069fd133857ce4aa6a219309d790124d4b3fee9`
