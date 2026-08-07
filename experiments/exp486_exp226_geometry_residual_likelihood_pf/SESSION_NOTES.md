# exp486_exp226_geometry_residual_likelihood_pf セッションノート

## 目的

exp279 absolute geometry unaryとexp281 residual-offset stateのPF移植設計を確定する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage1_all_variants_gate_failed_terminal_close`
- Priority: P2・高リスク
- compact self-contained Stage 0 / Stage 1・専用test: 実装済み
- 正規train Notebook採用・Kaggle package / push / fixed32 Stage 0:
  version 1 COMPLETE / fail-closed
- Stage 1: canonical version 4 COMPLETE / technical PASS / scientific gate FAIL
- inference・submission: 未承認
- variant CV: absolute `9.726938029` / residual `11.139812021`
- LB: なし

## 固定variant

1. `absolute_geometry_unary_sigma20_lambda050`
2. `slow_residual_offset_state`

exp419のrelative geometry proposalとは非重複。二variantは独立評価し、
same-OOF winner selectionはしない。

## 実行契約

- Stage 0: 64 PF wells、8,192 seed-well、4,096,000 particle starts。
- Stage 1: 1,546 PF wells、197,888 seed-well、98,944,000 particle starts。
- control PF / HMM / Beam / model / booster / GPU rerun 0。
- fixed32実run 1。Stage 0再実行なし。

## Leakage / 再現性

exp226 prediction-time allowlistは`well_id,row_idx,suffix_offset,tvt_geop`。
stable seeds、両variant freeze後のtruth attach、input/geometry/prediction SHAを固定する。

## 2026-07-30 compact self-contained実装

ユーザー依頼:

```text
exp486を実装してください
```

承認範囲は、凍結済みsteeringの「別承認後に2 variantをJupytext percent形式で
実装し、allowlist / toy parity / truth-late / seed / SHA testを作る」までと
解釈した。既存の正規train/inference Notebookは上書きしていない。

作成:

- `exp486_exp226_geometry_residual_likelihood_pf_compact_selfcontained_train.py`
- `exp486_exp226_geometry_residual_likelihood_pf_compact_selfcontained_train.ipynb`
- `exp486_exp226_geometry_residual_likelihood_pf_compact_selfcontained_inference.py`
- `exp486_exp226_geometry_residual_likelihood_pf_compact_selfcontained_inference.ipynb`
- `tests/test_exp486_exp226_geometry_residual_likelihood_pf.py`

実装内容:

- exp226 OOFは`usecols`で
  `well_id,row_idx,suffix_offset,tvt_geop`だけを読み、
  `fold,tvt_pred,gr_delta,tvt_true,error,abs_error`をprediction前に読まない。
- absolute variantはexp404の`(U=TVT+Z, U-rate)`遷移、初期化、Gaussian GR
  x1.0、ESS/resampling/rougheningを維持し、
  `0.50*(-0.5*min(((TVT-tvt_geop)/20)^2,600))`だけをlog weightへ加える。
- residual variantは各粒子を`(offset, offset_rate)`として持ち、
  `TVT=tvt_geop+offset`、rate momentum `0.998`、rate noise `0.002`、
  offset noise `0.005`、初期center
  `last_known_tvt-tvt_geop[first]`を実装した。
- 両variantは同じ`sha256("likpf::train::<well_id>")`base seedを使い、
  variant名をseed labelへ含めない。各well内で独立実行し、予測を混合しない。
- absolute geometry residual / log factor / ESS / resamplingと、
  residual offset / offset-rate / geometry delta / support / ESSを
  target-free ledgerとして保存する。
- fixed32の両variant prediction、mechanism ledger、runtime ledger、
  scientific/input contract、全content SHAをfreezeした後だけ、
  suffix truth、保存exp404 control、role、foldをattachする。
- fixed32 truth-late RMSEは記述値だけとし、CV、variant winner selection、
  Stage 1選択には使わない。
- inference候補はtest-side exp226 geometry再生成が未承認のため、
  明示的に例外を送出するfail-closed guardだけである。

実行契約:

- active scientific variants: `2`
- Stage 0: `2 × 32 = 64` PF well-runs、`8,192` seed-well、
  `4,096,000` particle starts
- Stage 1: 別承認時だけ`2 × 773 = 1,546` PF well-runs、`197,888`
  seed-well、`98,944,000` particle starts
- control PF / LightGBM config / trained fold / booster / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- `execution.selected_stage=null`、run flagはすべてfalse

構成比較:

- 直接のcompact parent exp404は11章・2,174行。
- exp486 train候補は11章で、runtime/config、fixed32/geometry入力、
  exp404前処理、2 PF kernel、parity/state contract、freeze、
  truth-late readout、gate、guarded orchestrationをNotebook上に展開した。
- 同一exp helper import、`__file__`、`Path(__file__)`は0。

検証コマンド:

```bash
PYTHONPYCACHEPREFIX=/tmp/exp486-pycache \
NUMBA_CACHE_DIR=/tmp/exp486-numba-cache \
.venv/bin/pytest -q \
  tests/test_exp486_exp226_geometry_residual_likelihood_pf.py

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp486_exp226_geometry_residual_likelihood_pf/\
exp486_exp226_geometry_residual_likelihood_pf_compact_selfcontained_train.py

.venv/bin/python -m py_compile <compact train/inference sources>
.venv/bin/ruff check <compact sources and dedicated test> --select F821,F401,E9
```

初回専用test結果は`14 passed`。absolute `lambda=0`がexp404 kernelと
bitwise一致し、residual no-noise toy transition、allowlist、fixed32 SHA、
common seed、truth-late ledger、inference guardを確認した。

最終検証:

- dedicated tests: `14 passed`
- repository full test: `make test` exit `0`、収集nodeid `1,983`
- train / inference Jupytext変換と`--test`: PASS
- train / inference / settings `py_compile`: PASS
- Ruff `F821,F401,E9`: PASS
- Ruff format check: PASS
- template / strict config / strict exp validation: PASS / PASS / PASS
- `make update-summary`: PASS、454 experimentsを再生成
- compact trainは11章・23 cells・2,339 source lines。親exp404の
  11章・2,174 linesに対して、2 variant kernelとgeometry契約をNotebook上へ
  追加しており、薄いorchestrationではない。

## 2026-07-30 Stage 0実行承認

ユーザー依頼:

```text
実行してください
```

承認範囲は、compact候補の正規train Notebook採用、canonical Kaggle package
作成・push、fixed32 Stage 0実行までとする。Stage 1、raw-test inference、
submissionは含めない。

push前実行量:

- active scientific variants: 2
- PF well-runs: `2 × 32 = 64`
- seed-well trajectories: `64 × 128 = 8,192`
- particle starts: `8,192 × 500 = 4,096,000`
- reporting folds: score用学習0、truth-late表示5
- saved control PF rerun: 0
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`

入力はexp226 OOF Notebook source 1件と保存exp404 control Dataset 1件を使う。
親control、exp226、HMM、Beam、modelは再実行しない。

正規Notebook / package preflight:

- 正規train Notebook: 23 cells、SHA
  `1e2929daf687b281adaf2cf9f27d1ea2e1b0c2d6c3e6baa17730181273d0814f`
- initial canonical kernel:
  `kentookumura/exp486-exp226-geometry-residual-likelihood-pf-train`
- title slugとkernel id slug: 一致
- private / CPU / internet off / run-on-push: PASS
- competition source: `rogii-wellbore-geology-prediction`
- Notebook source: `kentookumura/exp226-k16-kappa-repro-train`
- Dataset source: `kentookumura/exp404-v1-frozen-predictions`
- package config SHA:
  `7ea3bd4da83d44923574cb5ad49b2a56f9bed997dc6bd1d629cf0972bef10243`
- compact source SHA:
  `2a035791ec25eb2b6039e7af1b76b9bed1718b35d272634baa5c2c56d79fc9ba`
- push Notebook SHA:
  `27d501e4daa657a64d8315655928efaed3ac1f91a579db2ca9dbc7a29fbd1ed4`
- metadata SHA:
  `6972de4dd2e59c1b4685535622ee6b831c33ff6c011b59332f4f0c70d75eecc7`
- bootstrap ZIP内のconfig、settings、compact source、fixed32 manifestは
  正ファイルとbyte一致。`__file__`は0。

初回push:

```bash
make push-kaggle-train EXP=exp486_exp226_geometry_residual_likelihood_pf
```

- Kaggle `SaveKernel`は詳細なし400を返した。
- id/title slug自体は一致していたが、slugが51文字だった。
- read-only確認では対象kernel pullは403で、Notebookは作成されていない。
- exp226 Notebook sourceはid_no `126463591`としてpullでき、exp404 Dataset
  sourceにも必要なpaired predictionが存在した。
- Kaggle slug境界を避け、意味を維持した43文字のcanonical名
  `kentookumura/exp486-exp226-geometry-residual-likpf-train` /
  `exp486 exp226 geometry residual likpf train`へ一度だけ短縮してpackageを
  再生成する。以後このidを固定する。

再生成package:

- config SHA:
  `f9078d658f866c3d85fa99eea6ec118fdc4cd80b8d6069ad2744713d86634b12`
- push Notebook SHA:
  `21faa5677d54dd8b58acb157181362478789dfa820febe84909a7b1709d14035`
- metadata SHA:
  `1d2778ef8a09eac178fc09605389f59107d75a95ae586a13191d1693eda0d054`
- id/title slug一致、private / CPU / internet off / run-on-push、
  Notebook/Dataset/competition sourcesを再確認した。
- bootstrap ZIP内のconfig、settings、compact source、fixed32 manifestは
  再び正ファイルとbyte一致した。

Kaggle push / pull確認:

- canonical kernel v1 push: 成功
- kernel id:
  `kentookumura/exp486-exp226-geometry-residual-likpf-train`
- Kaggle kernel id_no: `129170320`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp486-exp226-geometry-residual-likpf-train`
- pull後もid/title slug一致、private / CPU / internet off、
  Notebook/Dataset/competition sources一致を確認した。
- Kaggle pull metadata SHA:
  `8784a6486a407b844ddfcd9f7858d3b03ca8d399087a1d2dedbdf7c2397de13a`
- Kaggle pull Notebook SHA:
  `d81d7bc70f04c45412e97e31173e7929941073cbd5d842933a6fb3d48cd617c3`
- 監視開始時点のstatusは`KernelWorkerStatus.RUNNING`。実行中ログはまだ
  空だが、同一v1を継続監視し、空ログを理由に再pushしない。

### fixed32 Stage 0完了結果

- terminal status: `COMPLETE`
- completed at:
  `2026-07-30T12:52:05.140987+00:00`
- experiment status: `stage0_fail_closed`
- stage:
  `stage0_fixed32_technical_mechanism_preflight_not_cv`
- Stage 1 eligible: false
- next action:
  `close_failed_contract_without_parameter_or_gate_rescue`

実行量:

- 2 scientific variants
- 32 wells / 156,088 rows
- 64 candidate PF well-runs
- 8,192 seed-well trajectories
- 4,096,000 particle starts
- saved control PF / LightGBM config / trained fold / booster / HMM / Beam /
  GPU run: `0 / 0 / 0 / 0 / 0 / 0 / 0`

technical gate:

- `technical_all_pass=false`
- PASS:
  common stable seed、execution count、finite mechanism ledger、
  finite prediction、geometry allowlist/coverage、forbidden read 0、
  pre-freeze truth/control/role-fold read 0、variant式/state、RSS
- FAIL:
  `runtime_projection=false`
- fixed32 target-free wall:
  `1,029.996204853 sec`
- 64 variant-wellの合計時間:
  `7,487.545464754 sec`
- 事前固定式によるfull投影:
  `180,871.020132966 sec`
- 上限:
  `30,600 sec`
- 超過:
  `150,271.020132966 sec`（上限比`491.081765%`）
- peak RSS gate measurement:
  `1.239162445 GiB`

参考としてwall timeを単純に`1,546 / 64`倍した値は
`24,880.845823 sec`だが、事前固定gateはvariant-well合計時間を使う。
終了後にgate式を差し替えず、original gateのFAILを保持する。

mechanism gate:

- `mechanism_all_pass=false`
- absolute geometry factor active、geometry residual std非負、
  両variant ESS正、residual state非退化はPASS
- `residual_support_fraction_bounded=false`
- support min / max:
  `0.9999999999999988 / 1.0000000000000011`
- `support > 1`: 54,924 / 156,088 rows
- `support < 0`: 0 rows

support FAILは正規化weight和の浮動小数overshootで、最大約`1.1e-15`である。
ただし独立したruntime gateも大幅FAILしているため、事後tolerance追加や再runで
Stage 0を救済しない。

fixed32記述RMSE（CVではない）:

- absolute geometry unary:
  `9.183489453268399`
- slow residual-offset state:
  `10.399506240490478`
- saved exp404 control:
  `9.616740808061033`
- exp226 `tvt_geop` reference:
  `9.267204778193300`

absolute unaryはsaved control比`+0.433251354793 ft`、exp226 reference比
`+0.083715324925 ft`だが、fixed32はCVではなくtechnical gate FAILである。
同一scopeのwinner selectionは禁止しているため、Stage 1へ昇格させない。
residual stateはsaved control比`-0.782765432429 ft`だった。

truth-late / freeze:

- before both freezes:
  truth / control / role-fold / forbidden geometry reads =
  `0 / 0 / 0 / 0`
- geometry safe rows:
  `156,088`
- frozen variant-wells:
  `64 / 64`
- after freeze:
  truth / control / role-fold =
  `156,088 / 156,088 / 32`

artifact検証:

- Kaggle outputを`/tmp/exp486-v1-output`へ取得した。
- scientific contract SHA:
  `62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e`
- prediction logical / decompressed / raw SHA:
  `10451d62e5921fd5624d93b5c0025ac2e575fdbc7e42c8d9bd24b7ba6f736821` /
  `1bf351e9e57e08e84b4d1a9d719d2f1dfbef2a99d7e95b585b8c97039804a058` /
  `4a572b8f4cbc4a9a0c2f3bdbc955136fc14aef0005c47a4a20b1dd8ca04293ab`
- absolute ledger decompressed / raw SHA:
  `7bf5d5a11db73045a3967ddf27b2d983afa904242355dae59d98b2a1f8c62310` /
  `f8b55b83c39abec09764b4aca5dd1e81fb47b938803d5c5c130ab8421c84e20b`
- residual ledger decompressed / raw SHA:
  `177cf13a28633f5b6ca0672f5e8759c571edb7e7499d5adb692289b05e05e38d` /
  `8f914049cd61344e082c1546de724da4cdff1644153805565bdb467ebb69d2cf`
- gate / runtime / summary SHA:
  `5908d54aa2d60ef06d8688a014ba3cbca4e896198b5718a6bcc05316f2137e2e` /
  `d3e9aafc700cbc20f55a063ea1968c2735eab33bc3ab05ef42309a0dd4712a7a` /
  `fe706d83875fba69eca545a1af9b441fe038989c60e3eb7bb9a2cc6104695878`
- terminal log SHA:
  `d86bb06c6b6862866709692f757bbe61748b7cc5fc36e5a7189e3b2038c2d5eb`
- 取得ファイルのraw SHAはKaggle summary記載値と一致した。

結果記録後の検証:

- exp486専用test:
  `14 passed`
- Jupytext roundtrip:
  PASS
- `py_compile`:
  PASS
- ruff `F821/F401/E9`:
  PASS
- `metrics.json` parse:
  PASS
- `make validate-exp EXP=exp486_exp226_geometry_residual_likelihood_pf`:
  strict PASS
- `make update-summary`:
  PASS、456 experimentsを再生成
- `make test`:
  exp486外のcollection error 5件で停止。repository rootにあるexp410用
  `config.yaml`をexp297/301/333/336/349の旧path resolverが先に読む既存問題で、
  exp486専用14件は独立にPASSした。他実験とroot loose fileは変更しない。

## 次のアクション

branchを閉じる。fixed32をCVとして解釈せず、runtime/support gate緩和、
favorable rerun、parameter/noise/grid変更、二variant併用、same-OOF winner、
Stage 1、inference、submissionへ進まない。

## 2026-07-30 Stage 1実行承認・実装

ユーザー依頼:

```text
実行時間は許容するのでStage 1に進んでください
```

この明示承認により、直前のbranch閉鎖判断を履歴として残したまま、全773 wellsの
train-side Stage 1を再開する。Stage 0 original gateは
`runtime_projection=false`、`residual_support_fraction_bounded=false`のまま
変更しない。後者はsupport min/max
`0.9999999999999988 / 1.0000000000000011`という正規化weight和の丸め誤差なので、
Stage 1 technical readbackだけ`[-1e-12, 1+1e-12]`を許容する。

Stage 1実行契約:

- scientific variants: 2
- wells / score rows: 773 / 3,783,989
- candidate PF well-runs: `2 × 773 = 1,546`
- seed-well trajectories: `1,546 × 128 = 197,888`
- particle starts: `197,888 × 500 = 98,944,000`
- reporting folds: 5、学習fold: 0
- saved control PF / HMM / Beam / LightGBM config / booster / GPU rerun:
  `0 / 0 / 0 / 0 / 0 / 0`
- exp226 geometryはprediction前にallowlist
  `well_id,row_idx,suffix_offset,tvt_geop`だけを読む
- truth、保存exp404、保存exp209、reporting fold、hidden-like roleは、
  2 variants ×773 wellsのprediction / mechanism ledger / content SHA
  freeze後だけattachする
- absolute / residualは保存exp404 controlに対して独立判定し、
  fixed32値または同じOOFからwinnerを選ばない
- raw-test inference / submissionは引き続き未承認

実装:

- all-well raw identityとexp226 geometry coverageを固定した。
- 2 variant prediction、absolute/residual mechanism ledger、well auditを
  deterministic gzip / SHA付きでfreezeする。
- 保存exp404 / exp209、5 fold、hidden-like spatial/typewell-purgedをtruth-lateで
  attachし、overall/fold/raw missing/high missing/1000+/hidden-like/by-wellと
  fixed HMM-PF 50:50 guardをvariant別に出力する。
- Stage 0 original FAIL保持、runtime exception、support numerical tolerance、
  control parity、実行量、SHA readbackを共通technical gateにした。
- `cv`は単一値にせずvariant別に報告し、eligible variant listだけを出す。

push前検証:

- exp486専用test: `16 passed`
- Jupytext train / inference roundtrip: PASS
- `py_compile`: PASS
- ruff `F821/F401/E9`: PASS
- Stage 0 control再学習・再実行: 0
- package config SHA:
  `f3b46831c6d76a2b107bc335ce4d0159268f77dbbe3c0eaf054eb991597f12f9`
- compact source SHA:
  `71f82913ec31a48f51d374511b2ca5cb2623e968f03e7434605006d64613a20c`
- canonical source Notebook SHA:
  `783c4c8e843a3c0f431b94354e29e3ba724eecb4481a8838604376bf9f027565`
- push Notebook SHA:
  `fe1b958849cb98b863981e88158eab9a903f3d90041e1b24d65c79d0c72c2c92`
- metadata SHA:
  `bce988a3ac9581050f6cf910fb4809148f0b018e02f6492a9dd2732dab33f1c3`
- metadataはprivate / CPU / internet off / run-on-push、competition 1、
  saved exp404 Dataset 1、exp226 / exp209 / exp115 Notebook source 3、
  id/title slug一致を確認した。
- package内config / settings / compact sourceは正ファイルとbyte一致した。

canonical kernel
`kentookumura/exp486-exp226-geometry-residual-likpf-train`の同一slugを使い、
version 2としてpushする。結果・artifact SHA・最終gateは完了後に追記する。

Stage 1 push:

- version 2 push: 成功
- pushed at: `2026-07-30T13:31:53Z`
- kernel id_no: `129170320`（version 1と同一）
- initial status: `KernelWorkerStatus.RUNNING`
- pull metadata SHA:
  `852ed6eab6006d7def7b835aff5cbe713df563c96c80229252bb4db1c2152fbe`
- pulled Notebook SHA:
  `1e5c047e875ff97659851d6709116ce8201a2fae4f584dd2896802a18fcbb06d`
- pull後もprivate / CPU / internet off、competition / Dataset / 3 Notebook
  sources一致を確認した。
- 実行開始直後のログは空。空ログを理由に再pushせず、同一version 2を監視する。

### Stage 1 version 2 ERRORとtarget-free resume

- terminal status: `ERROR`
- failed at: `2026-07-30T17:10:58Z`
- elapsed to error: `13,311.053096 sec`
- failure cell: truth-late saved exp209 HMM integrity check
- 原因:
  configの期待SHAが
  `8e2f42367b7b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
  という62文字になっていた。正しい既存固定値と実ファイルSHAは
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`。
- 科学parameter、入力source、PF計算自体のfailureではない。

ERROR前に次を完了・freezeしていた。

- 773 wells / 3,783,989 rows
- 2 variants / 1,546 candidate PF well-runs
- 197,888 seed-well / 98,944,000 particle starts
- truth/control/role-fold read before freeze: `0 / 0 / 0`
- prediction logical SHA:
  `70a5ac662c9c58fe54d050f1350ed08e912ecb4edc6362e98e3c3663cd704ea8`
- prediction raw / decompressed SHA:
  `0fe0cdda02c49eaa80ab668cb8e68e5b3e02b98f46a5105be6818497f2b65de3` /
  `05f692238c53711172f5e4e430eb46766cd26f2e3dac92472cb211b5639153e6`
- absolute ledger raw / decompressed SHA:
  `3f7381f1265d9b5bc9f0b9a68d4a3a088a620bfc04d674faca0d608390ea7b96` /
  `ef76d89fef9529d11501a5c17999e95e22d45006c0fd1e6b80e71d674b6c5a80`
- residual ledger raw / decompressed SHA:
  `9aaa61988642bc41e9d4b16162980e7ba6b4cfeab3612716af9d6580b5fdfcf0` /
  `e41dc48abbc849c5faa6d564ddd45faefdb100747af9c57911c258eba07a59b7`

Kaggle ERROR outputを`/tmp/exp486-v2-error-output.PZgsXG`へ取得し、freeze
manifestと実ファイルのraw/decompressed SHAを一致確認した。回収artifactを
private Dataset
`kentookumura/exp486-v2-stage1-frozen-targetfree`（datasetId `11428952`）
として作成した。Kaggle側metadataはprivateを確認した。

version 3 resume方針:

- scientific contract SHAはversion 2と同じ
  `62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e`。
- Datasetのprediction / 2 mechanism ledgers / audit / freeze manifestを
  raw/decompressed/logical SHAで再検証する。
- version 3 current PF well-runs / particle starts: `0 / 0`。
- source version 2の1,546 PF well-runs / 98,944,000 particle startsを
  experiment全体の実行量として保持する。
- truth、保存exp404/exp209、fold、hidden-like roleをrefreeze後にattachし、
  同じ事前固定gateを評価する。
- inference / submissionは未承認のまま。

resume loader、展開済みKaggle CSV対応、streaming logical SHAを実装し、
専用test `17 passed`、Jupytext、構文、ruff、strict exp validationをPASSした。
ローカル全artifact同時readbackはworkspace memory上限でexit 137となったため、
モデル/PFをlocal実行せず、Kaggle high-memory readoutを正とする。

version 3 push前契約:

- current scientific variants: 2（source freezeの同じ二variant）
- current PF well-runs / seed-well / particle starts:
  `0 / 0 / 0`
- aggregate Stage 1 source execution:
  `1,546 / 197,888 / 98,944,000`
- control PF / HMM / Beam / LightGBM config / trained fold / booster / GPU
  rerun: `0 / 0 / 0 / 0 / 0 / 0 / 0`
- reporting folds: 5、学習fold: 0
- package config SHA:
  `947afd8321ada9adb46ec200019c50b6f182713631c06c1a0ff31830cb019604`
- compact source SHA:
  `fa6c8ace353eacade7818f138aeb849c7fb06cf209cff2736996b08e7c376c07`
- canonical source Notebook SHA:
  `2b11bfa58fd2fb9c0fd56ff2046bbc688065340b4b5bade56368c40c72e6bee7`
- push Notebook SHA:
  `272f5097b243eab7eb64d09b6632eca2721b64f79d14d13e42b106c3e149072e`
- metadata SHA:
  `b95668245d8432bf0acf31b2fe0e076f42b25fbe179bffc5c2558a70f3a5012b`
- metadataはprivate / CPU / internet off / run-on-push、competition 1、
  Dataset 2（保存exp404、freeze済みexp486）、Notebook source 3、
  canonical id/title一致を確認した。

version 3 push:

- push: 成功
- pushed at: `2026-07-30T23:00:08Z`
- initial status: `KernelWorkerStatus.RUNNING`
- kernel id_no: `129170320`（versions 1/2と同一）
- pull metadata SHA:
  `55cec43d83caae78c992edcb410d608218e69adcf7748f3dd0cd42155827a70f`
- pulled Notebook SHA:
  `027095b83d77bf61304cc091dcd53d2de09ecd2c62944468b909ee628b80a447`
- pull後もprivate / CPU / internet off、Dataset 2 / Notebook 3 /
  competition source一致を確認した。

version 3 terminal:

- status: `ERROR`
- failed at: `2026-07-30T23:03:18Z`
- current PF well-runs / particle starts: `0 / 0`
- failure:
  `frozen resume logical SHA mismatch: absolute_ledger`
- Datasetのraw/decompressed SHA、prediction subset logical SHA、coverage readまでは
  PASSした。KaggleがCSV.gzを展開した全列ledgerをpandasで読み、
  floatを再serializeしたbyte表現だけが元CSVと一致しなかった。
- absolute / residual ledgerは全列CSVで、元freeze時のlogical SHAと
  decompressed payload SHAが同一である。version 4では展開payload SHAを
  logical integrityの正とし、読込後schema / row / well / finite値を別checkする。
- 科学parameter、prediction、mechanism値、gateは変更しない。

version 4:

- package config SHA:
  `1b1cf538681b0127449db590bb282c530708ef1ad7b9bb56396d42ab4cceedfe`
- compact source SHA:
  `7f5b9b896f77d8c9ba81e905fb34e0291f03b99865b4a5c128a2b8c1c9d23247`
- push Notebook SHA:
  `2d54ea06d516d7ab7db15a0e42a27358a361337197c02e21226425bb2b9efddd`
- metadata SHA:
  `b95668245d8432bf0acf31b2fe0e076f42b25fbe179bffc5c2558a70f3a5012b`
- push: 成功
- pushed at: `2026-07-30T23:08:59Z`
- initial status: `KernelWorkerStatus.RUNNING`
- current PF well-runs / particle starts: `0 / 0`

### Stage 1 version 4 COMPLETE・terminal close

- terminal status: `COMPLETE`
- result generated at: `2026-07-30T23:17:29.082818+00:00`
- current version PF well-runs / particle starts: `0 / 0`
- source version 2を含むStage 1実行量:
  - scientific variants: 2
  - candidate PF well-runs: 1,546
  - seed-well trajectories: 197,888
  - particle starts: 98,944,000
  - control PF / HMM / Beam / LightGBM config / trained fold / booster /
    GPU rerun: `0 / 0 / 0 / 0 / 0 / 0 / 0`
- rows / wells / reporting folds: `3,783,989 / 773 / 5`
- combined Stage 1 elapsed: `13,769.492127 sec`
- version 4 restored-freeze / large-artifact完了:
  `82.271970 / 458.439031 sec`
- peak RSS: `4.505966 GiB`

technical gateは全項目PASSした。

- target-free freeze前のtruth / control / role-fold read:
  `0 / 0 / 0`
- frozen variant-wells: `1,546 / 1,546`
- saved control parity max absolute difference:
  `4.231708317e-10`
- fixed HMM-PF control parity max absolute difference:
  `1.691996303e-7`
- residual support min / max:
  `0.1653684743404633 / 1.000000000000003`
- Stage 0 original FAILを保持し、Stage 1 readbackの`1e-12` toleranceだけを適用

独立科学gate:

| variant | RMSE | control比 gain | 改善fold | by-well p95 | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| absolute geometry unary | 9.726938029 | +1.187584044 | 4/5 | +10.069321492 | +44.021977054 |
| slow residual-offset state | 11.139812021 | -0.225289948 | 2/5 | +4.795182565 | +32.921501347 |
| saved exp404 | 10.914522073 | - | - | - | - |

absoluteはpooledと全事前scopeを改善し、固定HMM-PF 50:50も
`10.084909849 → 8.871021642`へ改善した。しかしby-well p95上限0と
worst上限`+0.25 ft`を大幅にFAILした。residualは固定50:50では
`+0.252270349 ft`改善したが、pooled、3 folds、raw observed、high missing、
1000+、hidden-like 2面、well tailをFAILした。

このためeligible variantsは`[]`、最終statusは
`stage1_all_variants_gate_failed_terminal_close`。事前契約どおり、
same-OOF winner、sigma/lambda/noise/particle/seed/temperature/gate変更、
blend/selector rescue、raw-test inference、submissionを行わない。

主要SHA:

- scientific contract:
  `62dcb499c0c9c9320091fa28663771493847dd6f46f03737015d1373dddc5f8e`
- prediction logical:
  `70a5ac662c9c58fe54d050f1350ed08e912ecb4edc6362e98e3c3663cd704ea8`
- primary / by-well / blend metrics:
  `326aa169d85e003d7e81e494381ba943c0442e8c5c015886e55ebf7c4020988c` /
  `63aa32898dd34a725fe68c16f4569a21060967444c4389c3ed4f833d1e022a72` /
  `ce4541211b6a422e5f5c1ce52bcc9833bb69f9401e0c26c28004edd23f39490b`
- gate / runtime / summary:
  `ec5e450ae953267fdbfc37170df5899b629e01d5f0c087d9bb027f6592dabb46` /
  `d11ef8af62b67ca7f140df15ca9eab48c49f27854dcfb25ad4d54e421dfb0e99` /
  `b637038f913411714c672914cfa9532190d97df92a9f0f77cd83c283b4e20f29`
- retrieved terminal log:
  `6bc378e5aaa39dad1e1a6a66bb8fd8b1123300e6a60e20b7a1bdc7d931faa452`

最終ローカル検証:

- exp486専用test: `18 passed`
- Jupytext roundtrip: PASS
- `py_compile`: PASS
- ruff `F821/F401/E9`: PASS
- `validate_experiment.py`: strict PASS
- `validate_project.py --strict`: PASS
- `make update-summary`: PASS、461 experiments
- Kaggle API terminal再確認:
  `KernelWorkerStatus.COMPLETE`
