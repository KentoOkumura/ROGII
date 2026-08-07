# exp485_stratified_initial_rate_bank_pf セッションノート

## 目的

exp268 initial-rate bankをPFの単一equal-strata priorへ移植する設計を確定する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage1_gate_failed_terminal_close`
- Priority: P3
- Stage 0実装・契約test・正規train Notebook・Kaggle package: 完了
- Kaggle fixed32 Stage 0 version 1: COMPLETE / fail-closed
- Stage 1: canonical kernel version 3 COMPLETE / scientific gate FAIL
- inference・submission: 未承認
- CV: `11.092618091` / LB: なし

## 固定差分

- rate windows: `30/32/64/128/256`。
- 500 particles: 各100、`particle_index % 5`でinterleave。
- within-center spread: `0.01`。
- duplicate centerは削除せず、component labelは診断専用。

## 実行契約

- Stage 0: 32 PF wells、4,096 seed-well、2,048,000 particle starts。
- Stage 1: 773 PF wells、98,944 seed-well、49,472,000 particle starts。
- control PF / HMM / Beam / model / booster / GPU rerun 0。
- push対象は1 scientific variant、LightGBM config 0、trained fold 0、
  booster 0、control PF / HMM / Beam / GPU rerun 0。
- fixed32 Stage 0の実run 1回。再実行なし。

## 実装

- `exp485_stratified_initial_rate_bank_pf_compact_selfcontained_train.py`
  をJupytext percent形式で作成し、正規train Notebookへ採用した。
- `particle_index % 5`のexact 5×100 allocationを実装した。
- rate bankは`30/32/64/128/256`行、minimum valid steps 3、
  fallback 0.0、duplicate保持に固定した。
- component labelは診断専用で、resampling時にparent labelをcopyする。
- row 0/32/128/512/finalでfiltered posterior massとpost-resample countを保存し、
  first extinction row、first-resample ESSを記録する。
- Stage 0は`MD/Z/GR/TVT_input`とType Wellだけを読み、truth/controlは読まない。
- outputはprediction、rate bank、component ancestry、well/runtime audit、
  scientific/input/freeze SHA、fail-closed gate、metricsを保存する。

## 検証ログ

- `.venv/bin/python -m py_compile ...`: PASS
- `.venv/bin/ruff check ...`: PASS
- `.venv/bin/pytest -q .../test_exp485_contract.py`: `7 passed`
- Jupytext `--to ipynb --test`: PASS
- `make validate-exp EXP=exp485_stratified_initial_rate_bank_pf`: strict PASS
- `make prepare-kaggle-notebooks ... --strict`: PASS
- `make test`: exp485外のcollection error 5件で停止。repo rootに残る
  exp410用`config.yaml`をexp297/301/333/336/349の旧path resolverが先に読み、
  各scientific contractと不一致になる既存問題である。exp485対象testは独立に
  7/7 PASSしており、この作業ではroot loose fileや他実験を変更しない。
- package metadata:
  - id: `kentookumura/exp485-stratified-initial-rate-bank-pf-train`
  - title: `exp485 stratified initial rate bank pf train`
  - CPU / internet off
- 親compactは2,174行・11章、exp485 compactは1,869行・11章。
  exp485はStage 0 target-free scopeに限定しつつ、入力、rate bank、PF、
  parity、freeze、gate、生成物の各章を正規Notebook上に展開している。
- duplicated-center sentinelはexp404 parent kernelとprediction / log-likelihood
  の双方でbitwise equalityをPASSした。

## Kaggle Stage 0実行

- 2026-07-30:
  - credential preflight: OAuth / legacy CLI credential PASS。
  - canonical kernelの事前`pull`は403、`kernels list --search`は`Not found`。
    既存versionなしとして同じcanonical idへ初回pushする。
  - packageを`run_on_push=true`で再生成した。
  - bootstrap内configは`run_stage_0=true`、push approval true、
    fixed32 asset同梱、variant/config/fold/booster/control PF
    `1/0/0/0/0`を確認した。
  - compact source SHA256:
    `765b0d95abd8796b98b8e56b8dfb4969a469da4740a1fd04df966e90bb07b333`
  - canonical train Notebook SHA256:
    `80fddd016fd05c0a7cbfca316e6756643b8dcd00b3b3180f31b8d45c26472b50`
  - push package Notebook SHA256:
    `d199c1895a42229d63e5f6dd06425cc4a3f33c1b0374c6b26992ae1a38c07cf6`
  - kernel metadata SHA256:
    `dae4074a7c54407119589a24cdbedfbee664dda8e8d05cb27c67ad18eaa8f302`
  - canonical kernel version 1 push: 成功。
  - Kaggle `id_no`: `129169067`。
  - pulled metadata: private / CPU / internet off / competition source一致。
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp485-stratified-initial-rate-bank-pf-train`
  - push直後は実行中。同じkernel idを監視し、別slugへの再pushなしで
    `COMPLETE`を確認した。

### 完了結果

- terminal status: `COMPLETE`
- generated at: `2026-07-30T12:40:53.929312+00:00`
- status: `stage0_fail_closed`
- all gates: FAIL（14 checks中13 PASS）
- 唯一のFAIL:
  - `runtime_projection_within_limit=false`
  - candidate `1,278.942056894 sec`
  - full projection `30,894.444061853 sec`
  - fixed limit `30,600 sec`
  - excess `294.444061853 sec`（`0.962%`）
- resource:
  - total `1,290.874269962 sec`
  - peak RSS `0.364715576 GiB`
- execution:
  - 32 wells / 156,088 rows
  - 4,096 seed-well / 2,048,000 particle starts
  - variant/config/fold/booster/control PF/HMM/Beam/GPU:
    `1/0/0/0/0/0/0/0`
- technical/mechanism PASS:
  - 5×100 allocation、interleave、rate式、fallback、duplicate保持
  - finite prediction coverage、posterior mass normalization
  - particle count conservation、stable seed、execution count
  - duplicated-center exp404 prediction/log-likelihood bitwise parity
  - non-global-degeneracy、RSS、truth-late read 0
- mechanism readout:
  - multiple unique centers `25/32` wells
  - one unique center `7/32` wells
  - fallback centers `0`
  - component extinct seed fraction max `0.921875`
- content SHA:
  - scientific contract:
    `c3cc258bafb9489d4ce02f06e9cc4a63f00805230a86982c1ae8140cae8ee86e`
  - prediction logical / decompressed:
    `a9ed7c26b9e69723ddffe7cb144a91391a984f2f0849d1f89c5519c2f403c198` /
    `39941f00e25927611e84c10af0057f85cf81813020f45ed5351138e74e464c9c`
  - rate bank logical / decompressed:
    `432b9af8ca05e06ccc6fb69603c44addd328031d8d18dad766ca40adccc5a953` /
    `369cf494018ec648a9bed8cabff21f2870d3bd3d3b3254164ac656acfafc9903`
  - component ancestry logical / decompressed:
    `96c89c4f09b4434da0f0e52f39829457f2d37a2801113ea0c1e5a79304b4fec6` /
    `e9cca2189cf8513ba5b1cb116572f36d6ed94a1c852cc4de0ea89f791ea23538`
  - terminal log:
    `2b3643b9a22e172b4fe85024a3e2e79c9c1566838c49c0839a284a5103219d07`
- output archive: 未取得。logsにgate、runtime、count、SHAが揃っており、
  Stage 1入力や提出物の実ファイル確認は不要なため。

## Runtime例外とStage 1承認

- ユーザーは`30,894.444 sec`程度のfull runtimeを許容範囲と明示した。
- 元の`30,600 sec` gate FAILは監査履歴として保持し、PASSへ変更しない。
- ユーザーは全773 wellsのStage 1実行を別途明示承認した。
- 実行契約:
  - scientific variant `1`
  - candidate PF well-runs `773`
  - seed-well trajectories `98,944`
  - particle starts `49,472,000`
  - LightGBM config / trained fold / booster `0/0/0`
  - control PF / HMM / Beam / GPU rerun `0/0/0/0`
- Stage 1 self-contained実装は、candidate prediction / rate bank /
  component ancestry / SHAを全773 wellsでfreezeした後だけtruth、保存exp404
  control、exp209 HMM、reporting fold、hidden-like roleを結合する。
- 専用contract testはStage 1承認境界、runtime例外、promotion gate、
  prediction SHA readbackを追加して`10 passed`。
- inferenceとsubmissionは未承認のまま無効。

## Version 2実行前の次アクション（履歴）

canonical Kaggle kernel version 2へStage 1 packageをpushし、同じkernel idを
完了まで監視する。

## Stage 1 pre-push

- prepared at: `2026-07-30 13:06:36 UTC`
- canonical kernel version 1を`kaggle kernels pull -m`で確認し、
  id_no `129169067`の同じkernel idへversion追加する。
- package metadata:
  - private / CPU / internet off / run-on-push true
  - competition source: `rogii-wellbore-geology-prediction`
  - dataset source: `kentookumura/exp404-v1-frozen-predictions`
  - kernel sources: exp209 / exp226 / exp115
- bootstrap内config:
  - status `stage1_approved_pending_kaggle`
  - `run_stage_0=false`, `run_stage_1=true`
  - Stage 1 approval / runtime exception `true/true`
  - variant/config/fold/booster/control PF `1/0/0/0/0`
  - Stage 1 `773` PF well-runs / `98,944` seed-well /
    `49,472,000` particle starts
- pre-push validation:
  - contract tests `10 passed`
  - py_compile / ruff F821 / Jupytext roundtrip: PASS
  - strict experiment validation: PASS
- SHA256:
  - Stage 1 scientific contract:
    `599d39931c9e5f820469b531d1ed64b383f381918cdb0055700aa0beb7dd4233`
  - compact source:
    `d99ffe1a4e10a5a90c3e8f1da940d5180b9855595af34ea91d915443ae166f7b`
  - canonical train Notebook:
    `a864273487de00738875b5fab5b62bd3b6697bc044a1f3888b01a0f77ee73c93`
  - push package Notebook:
    `2de39fc609fe3ddafd1c0948fef79f41492e19dc182469685900af3175cc070d`
  - kernel metadata:
    `dcc9ae3fea7fec952d52dddccfc18a2600fe3a973921779a62ad85713f896c30`
  - executed config:
    `290c513a1f5ee32b38bdf5c2f2d3c7514c309f5136e4f5c840e6ebdd3ce667b7`

## Kaggle Stage 1実行

- pushed at: `2026-07-30 13:07:35 UTC`
- canonical kernel version 2 push: 成功
- id / id_no:
  `kentookumura/exp485-stratified-initial-rate-bank-pf-train` / `129169067`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp485-stratified-initial-rate-bank-pf-train`
- push直後status: `RUNNING`
- 同じkernel idを監視し、logs空や一時的なstatus API errorを理由に
  別slugへの再pushは行わない。

### Version 2停止結果

- terminal status: `ERROR`
- terminal time: `2026-07-30 15:52 UTC`
- runtime to error: `9,825.627 sec`
- failure:
  `ValueError: exp485 saved exp209 HMM decompressed SHA mismatch`
- 失敗位置は全773 wellsのtarget-free prediction / rate bank /
  component ancestry / well auditをfreezeした後、truth-late readout中の
  保存exp209 HMM integrity checkである。
- truth/control/fold/hidden-likeのfreeze前readはすべて0。errorまでに
  truthと保存exp404 controlはfreeze後の揮発メモリへ読み込まれたが、
  target-free凍結成果物へ混入していない。
- target-free実行量:
  - 773 wells / 3,783,989 rows
  - 98,944 seed-well / 49,472,000 particle starts
  - status `ok` 773/773、finite 100%、unique ID 3,783,989
- v2 output archiveは、Stage 1の同一評価を再開する実ファイル確認が必要なため
  例外的に取得した。

### Version 2凍結成果物と再開

- private Dataset:
  `kentookumura/exp485-stage1-v2-frozen-target-free`
- DatasetはKaggle側でgzip CSVを展開するため、loaderは`.csv.gz`と`.csv`の
  両方を受け付ける。ただしgzip raw SHAまたは展開後content SHA、論理SHA、
  行数、well数、実行回数をすべて照合する。
- prediction:
  - gzip raw SHA:
    `265962880a0eaa172880f874840e988dfdf9aa5386a5d0cbe12630c2f70decbd`
  - decompressed SHA:
    `7cb11d339d92ca0ae3fef2de243e9754b83f2ec74211707b898f7ff7a2e77750`
  - logical SHA:
    `246e7473289bc19743fc3957b319b95a8b72543fb1f5748e0f34f390e980ea46`
- rate bank decompressed SHA:
  `c1d9d4b9acdafc2d02115de0d848e22cdb9d67586ff816debf297f1fa29efc38`
- component ancestry decompressed SHA:
  `d81b1b93ff0c47947ee27a9cc18e24fe8ed0686347fd4394f8e6a46fec018f80`
- well audit raw SHA:
  `22e529880ead9c56cd9742d4710c4ec172db15e5a66846920f113e830453b371`
- exp209 HMMは現行の正本を再取得すると、decompressed SHAが既定値
  `8e2f4236...`と一致した。実際に使用する`id,hmm_mean_tvt`の
  3,783,989行をstorage非依存にhashし、
  `41957dff094daddc7c9f73ac52baf6bbc22332e8ec8534aedb73fdc0e84b649c`
  を正本として追加した。version 3はこの使用列SHAが不一致なら停止する。
- version 3は同一scientific contract
  `599d39931c9e5f820469b531d1ed64b383f381918cdb0055700aa0beb7dd4233`
  のtruth-late評価だけを再開する。PF rerunは0。

## Version 3実行前の次アクション（履歴）

canonical Kaggle kernel version 3へresume packageをpushし、同じkernel idを
完了まで監視する。inferenceとsubmissionは実行しない。

## Stage 1 resume pre-push

- prepared at: `2026-07-30 22:15:28 UTC`
- canonical kernel: version 2と同じ
  `kentookumura/exp485-stratified-initial-rate-bank-pf-train`
- private / CPU / internet off / run-on-push true
- source:
  - competition `rogii-wellbore-geology-prediction`
  - private Dataset `exp404-v1-frozen-predictions`
  - private Dataset `exp485-stage1-v2-frozen-target-free`
  - saved kernel outputs exp209 / exp226 / exp115
- execution:
  - scientific variant `1`
  - Stage 1 source execution `773` PF well-runs /
    `98,944` seed-well / `49,472,000` particle starts
  - resume kernel candidate PF rerun `0`
  - LightGBM config / trained fold / booster `0/0/0`
  - control PF / HMM / Beam / GPU rerun `0/0/0/0`
- validation:
  - contract tests `11 passed`
  - py_compile / ruff F821 / Jupytext roundtrip: PASS
  - strict experiment validation / strict package preparation: PASS
  - Kaggleから再取得した自動展開Datasetで全件resume loader: PASS
- SHA256:
  - scientific contract:
    `599d39931c9e5f820469b531d1ed64b383f381918cdb0055700aa0beb7dd4233`
  - compact source:
    `2d6fa3d8daa60ffab4302b06dd683866932ff875eab5a11e0cfe112beb4621ad`
  - canonical train Notebook:
    `22145f364b2e2029198e6f956435abb70b41d5679d2c1ce0696701220375140f`
  - push package Notebook:
    `7d8a3b76eae65b2dcc0e9a80de618289f69ee8d78c4a69e6eaba4e4476237ad8`
  - kernel metadata:
    `128f6654960a8337ed09945ad8524aa0a86eb9a11c97993fc2f69798ad919e0e`
  - executed config:
    `c9c0cd78871557a72d4655a642199cc01931339c5d5afa76a53385d19a3c6bd1`

## Kaggle Stage 1 resume実行

- pushed at: `2026-07-30 22:17:08 UTC`
- canonical kernel version 3 push: 成功
- id / id_no:
  `kentookumura/exp485-stratified-initial-rate-bank-pf-train` / `129169067`
- push直後status: `RUNNING`
- 同じkernel idを完了まで監視する。別slugへの再push、candidate PF再実行、
  inference、submissionは行わない。

### Version 3完了結果

- terminal status: `COMPLETE`
- generated at: `2026-07-30T22:22:59.250298+00:00`
- status: `stage1_gate_failed_terminal_close`
- candidate / saved exp404 control RMSE:
  `11.092618091 / 10.914522073`
- improvement: `-0.178096018 ft`
- positive folds: `1/5`
  - fold 0/1/2/3/4:
    `-0.024325 / -0.156209 / +0.012064 / -0.033681 / -0.577412 ft`
- scope:
  - raw GR observed: `-0.250496 ft` FAIL
  - raw GR missing: `-0.020822 ft` FAIL
  - high missing fraction: `+0.018240 ft` PASS
  - MD-since 1000+: `-0.207085 ft` FAIL
  - hidden-like spatial: `-0.108050 ft` FAIL
  - hidden-like typewell-purged: `-0.109059 ft` FAIL
- by-well delta p95 / worst regression:
  `+0.422389 / +33.053515 ft`、ともにFAIL
- fixed exp209 HMM+PF 50:50:
  `10.117590985` vs `10.084909849`、`+0.032681 ft`でFAIL
- technical gate: 19/19 PASS
- primary scientific gate / fixed-blend guard / promotion gate:
  `FAIL / FAIL / FAIL`
- runtime:
  - version 2 to error: `9,825.627377 sec`
  - version 3 Stage 1 function: `109.857446 sec`
  - aggregate: `9,935.484823 sec`
  - peak RSS: `3.470413 GiB`
  - actual aggregateは`30,600 sec`以内。元のStage 0 projection FAILは履歴保持。
- reproducibility:
  - scientific contract:
    `599d39931c9e5f820469b531d1ed64b383f381918cdb0055700aa0beb7dd4233`
  - prediction logical:
    `246e7473289bc19743fc3957b319b95a8b72543fb1f5748e0f34f390e980ea46`
  - promotion gate:
    `05644fd4a0fb4fe65ac44790ba75c32401e285807b0a54dad77eb6a90894559f`
  - terminal log:
    `a8a68a23ab123fa0e0fcff2cfcf494d444e8ef28b5a43daed6a825adf1f5d3fb`
- version 3 outputは正確なCV/scope/by-well/gate記録に実ファイルが必要なため
  取得した。大きなtruth-late artifactはGitへ保存しない。

## 最終判断

equal-strata initial-rate bankはhigh-missing scopeだけ小幅改善したが、pooled、
4/5 folds、raw observed、long-tail、hidden-like、by-well tail、固定blendで
悪化した。親tail30 modeの粒子を観測が十分なwellでも一律に減らす副作用が
headroomを上回ったと解釈する。事前登録どおりparameter/gate/blend/selectorの
同一OOF救済をせずbranchを閉じる。inferenceとsubmissionは実行しない。
