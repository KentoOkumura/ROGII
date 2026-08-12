# 設計

## アプローチ

exp458 v2で凍結済みの4 well成果物をKaggle Notebook inputから読み、論理SHAと
well identityを検証する。fixed32 manifestをwell列だけで読み、残り28 wellを
exp458の4-process/1-thread scaled engineで一度だけ計算する。全32 wellの
prediction、acceleration posterior、target-free diagnosticを連結してSHAを
freezeした後、truth-late readoutを実行する。

exp458のparent numerical parity FAILはそのまま保持する。exp489では
`user_approved_small_numerical_deviation` を明示し、exact-equivalentとは呼ばない。
Stage 0Bのscientific gateはexp444から変更しない。

## 実験範囲

- 対象実験: `exp489_acceleration_state_fixed32_mechanism_audit`
- Route: `pf_beam`
- 親実験: 実装親 `exp458_acceleration_state_exact_runtime_engine_audit`、
  scientific parent `exp444_acceleration_state_exact_hmm`、root `exp209`
- 変更する変数: 実行scopeをfixed4 runtime auditからfixed32 mechanism auditへ
  拡張する一点（4 well再利用 + 28 well新規計算）。
- 固定する変数: exp444 scientific contract、exp458 runtime engine、
  fixed32 manifest、全gate閾値、truth-late順序、CPU/worker/thread設定。

## 再現性設計

- seed policy: 乱数なし。well identityの順序は辞書順、成果物はstable sortする。
- stochastic 処理の有無: なし。deterministic exact-HMM近似実行engine。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: 外側4 process、各processのNumba/BLAS threadは1。
  完了順を捨てwell順に再整列する。
- CPU/GPU runtime と deterministic flags: Kaggle CPUのみ。GPU/AMPなし。
- train cache / test feature regeneration の SHA 記録方針: manifest、bootstrap
  assets、raw well input、prepared input、kernel、prediction、posterior、
  diagnosticのSHAを記録する。
- model manifest / prediction / submission SHA 記録方針: fitted modelとsubmissionは
  生成しないためnull。連結prediction/posterior/diagnosticのdecompressed
  logical SHAを主証拠にする。
- Kaggle package bootstrap 確認方針: notebook内config SHA、ローカルconfig SHA、
  exp458 v2とexp209 input、competition raw dataを実行前に解決・検証する。

## リスク

- リークリスク: fixed32 scopeはwell identityだけで確定し、全成果物freeze前の
  truth/role/fold/episode/cause/control読込をledgerで0件に固定する。
- CV/LB 不一致リスク: Stage 0Bは意図的に固定32 wellのpreflightでCVではない。
- ランタイム/メモリリスク: exp458実測から28 wellは約506秒、
  peak RSSは約13.1 GiBを想定する。
- 再現性リスク: exp458はexp444とbitwise exactではない。v2で観測した差を
  waiverに固定し、engine/source/config/SHAが変わればfailする。
