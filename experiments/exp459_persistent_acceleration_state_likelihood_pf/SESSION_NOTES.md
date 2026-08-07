# exp459_persistent_acceleration_state_likelihood_pf セッションノート

## 目的

likelihood-PFの状態を`(TVT, U-rate)`から
`(TVT, U-rate, persistent U-acceleration)`へ拡張するStage 0候補を実装する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage0_fail_closed`
- Priority: P3
- implementation / contract test / 正規train Notebook: 実装済み
- Kaggle package / push / fixed32 Stage 0実行: version 1 COMPLETE
- Stage 1 / inference / submission: Stage 0 mechanism FAILにより不適格・禁止
- CV / LB: なし

## 根拠と差分

- exp444は3値acceleration exact HMMの数値contractをPASSしたが、full換算
  `144,232.851 sec`でruntime FAILし、科学評価へ進んでいない。
- exp367は3 fixed signed-curvature pathsのreal-minus-circular識別とfold再現性を
  FAILしたため、曲率のGR識別力にはnegative contextがある。
- exp459はexp404/417の500 particles ×128 seeds likelihood-PFへ、
  exp444と同じ3値persistent accelerationだけを追加する。
- exp444 / exp367のprediction、path、score、triggerは使用しない。

## 固定実行契約

- scientific variant: 1
- acceleration values: `[-0.0005, 0, +0.0005]`
- transition: boundary-folded `0.08 / 0.84 / 0.08`
- Stage 0: 32 candidate PF well-runs、4,096 seed-well trajectories、
  2,048,000 particle starts
- Stage 1上限: 773 candidate PF well-runs、98,944 seed-well trajectories、
  49,472,000 particle starts、4 CPU shards
- 保存exp404 control PF rerun、LightGBM config、trained fold、booster、HMM、
  Beam、GPUはすべて0

## 再現性

- base PF stream: exp404 stable per-well / per-seed streamを維持
- acceleration stream: split / well / seed indexから別SHA256 streamを生成
- acceleration drawはbase PF streamを進めない
- zero-acceleration sentinelでexp404 bitwise parityを要求
- truth / error / fold / episode / hidden-like roleはcandidate predictionと
  acceleration ledger、runtime ledger、全SHAのfreeze後にだけ結合
- 初回runはdeterministic anchorにしない

## 作成ログ

- 2026-07-30:
  - `task new-steering ...`は`task`未導入のため実行不可。
  - 当初`exp455`でscaffoldを作成したが、並行更新で番号重複を検出したため
    現在の最大番号に続く`exp459`へ移動した。
  - steering、config、backlog、実験文書をdesign-onlyで確定。
  - PF実装、test、package、Kaggle run、inference、submissionは0。
  - 追加依頼`exp459を実装してください`を実装承認として記録。
  - `exp459_persistent_acceleration_state_likelihood_pf_compact_selfcontained_train.py`
    を作成し、persistent acceleration PF、独立Park-Miller acceleration stream、
    target-free freeze、truth-late readout、Stage 0 fail-closed gateを実装。
  - acceleration RNGは
    `SHA256(exp459::acceleration::<split>::<well>::<seed_index>)`からstable seedを
    作り、base NumPy RNGを進めない。
  - zero-acceleration時だけrate更新をexp404と同一式へ分岐し、prediction、
    log-likelihood、resampling count、minimum ESS、position clip countのbitwise
    parityを固定した。
  - direction metricを「row tのevidence-weighted filtered acceleration mean符号」と
    「one-step future true U-rate curvature符号」の一致とし、exact zeroとsuffix
    最終rowを除外することを実装・Notebookへ明記。
  - exp408 persistent episodeの固定区間でcandidate/control SSEを比較し、
    matched controlは保存exp404 `likpf_scale_5_x1p0`だけをtruth attachment後に読む。
  - compact self-contained train sourceを正規train Notebookへ採用。
  - 親compactとの比較:
    - exp404 compact train: 2,174行、主要章8個。
    - exp459 compact train: 2,707行、主要章11個、Notebook 23セル。
    - exp404 input preparation/PF contractに加えてacceleration transition、
      parity、freeze、truth-late mechanism、generated artifactsを展開しており、
      同一exp helper importだけの薄いNotebookではない。
  - 実行した検証:
    - `.venv/bin/python -m py_compile ...`
    - `.venv/bin/ruff check ... --select F821,F401,E9`
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`
    - `.venv/bin/pytest -q tests/test_exp459_persistent_acceleration_state_likelihood_pf.py`
  - 結果: `10 passed`。Jupytext roundtrip、py_compile、ruffもPASS。
  - Kaggle package、Stage 0/1実行、予測生成物、inference、submissionは0。
  - 追加依頼`実行してください`をcanonical Kaggle train package / pushと
    fixed32 Stage 0実行の承認として記録。
  - push前の実行量を再確認:
    - active scientific variant: 1
    - candidate PF well-runs: 32
    - seed-well trajectories: 4,096
    - particle starts: 2,048,000
    - zero-acceleration sentinel wells: 4
    - 保存exp404 control PF rerun: 0
    - LightGBM config / trained fold / booster: 0 / 0 / 0
    - HMM / Beam / GPU: 0 / 0 / 0
  - CPU-only、internet disabled、当初のcanonical kernel id
    `kentookumura/exp459-persistent-acceleration-state-likelihood-pf-train`
    でStage 0だけを実行する。Stage 1 / inference / submission flagはfalseのまま。
  - 2026-07-30 12:03 UTCの初回push:
    - 実験名全体の56文字slug
      `exp459-persistent-acceleration-state-likelihood-pf-train`は
      `SaveKernel 400 Bad Request`で実行開始前に拒否された。
    - 直前の同slug `pull -m`は403で、Kaggle側にkernelは作成されていない。
    - idとtitle由来slugは一致していたため、過去のexp437と同じslug長制約と判断。
    - scientific nameを維持し、既存表記`likpf`で48文字に短縮した
      `kentookumura/exp459-persistent-acceleration-state-likpf-train` /
      `exp459 persistent acceleration state likpf train`をcanonical実行slugとする。
    - self-contained sourceは`src` import / `sys.path`依存がなく、既定packageの
      不要な`src/`が1.2 MiBあったため、API request縮小として`--no-src`で
      再packageする。科学条件、入力、実行量、gateは変更しない。
  - 短縮canonical slug / `--no-src` packageをstrict validationし、
    package config byte parity、private CPU、internet off、run-on-pushを確認。
  - Kaggle version 1をpush:
    - kernel: `kentookumura/exp459-persistent-acceleration-state-likpf-train`
    - id_no: `129167965`
    - URL:
      `https://www.kaggle.com/code/kentookumura/exp459-persistent-acceleration-state-likpf-train`
    - push後pullでprivate、GPU/internet off、competition/dataset sourceを確認。
  - 2026-07-30 12:24 UTCに`KernelWorkerStatus.COMPLETE`を確認。
  - terminal logの完全なsummaryを取得し、log SHA
    `01049dd17616920cb0d39d73ef40b05eeed0f317d2488ae7dc2ae4f26a420525`
    を記録。全gate / runtime / content SHAがログに揃ったためoutput archiveは
    ダウンロードしていない。

## Kaggle Stage 0結果

- status: `stage0_fail_closed`
- technical all-pass: true
  - 4 zero-acceleration sentinel wellsのprediction / log-likelihood /
    resampling count / minimum ESS / position clip countはexp404とbitwise一致、
    最大絶対誤差はすべて0。
  - truth / control / role-fold / episodeのpre-freeze readはすべて0。
  - transition、update order、`-delta_Z` identity、RNG stream separation、
    execution count、finite coverage、runtime/RSSはPASS。
- mechanism all-pass: false
  - mean nonzero acceleration mass: `0.666245156` PASS
  - future curvature direction agreement: `0.501085875`、positive fold `0/5` FAIL
  - persistent episode candidate/control SSE:
    `11,085,386.426 / 9,931,447.639`、reduction `-11.619039%` FAIL
  - persistent改善well / fold: `7 / 3` FAIL
  - matched-control candidate/saved RMSE:
    `4.827296 / 4.392083 ft`、delta `+0.435213 ft` FAIL
  - matched-control by-well delta p95: `+1.785604 ft` FAIL
- runtime:
  - candidate `928.287 sec`
  - total `1,122.990 sec`
  - full projection `22,423.933 sec < 30,600 sec`
  - peak RSS `0.795540 GiB`
- primary SHA:
  - scientific contract:
    `4949c627aca356e83bbedf568aa59aa90eff142674e254159addbbb6a2f51ffe`
  - prediction decompressed:
    `c1464190c5949de817aa7a6e287ccc4dbca314db3f07948f6834b51cff9922ef`
  - acceleration ledger decompressed:
    `5c465f7ada33ec6583f7de0e08f84abbd827cb5fdcab694e1210021b60ba5206`
  - gate report:
    `14887feb2fcf563af173ed8c7b8abf27389e34c3b9e97ca805a2a2dacd1d2727`

## 次のアクション

`close_branch_without_parameter_or_gate_rescue`。Stage 1、inference、submission、
acceleration / transition / noise / particle / seed / temperature / emission /
gate / blend / selectorのsame-fixed32救済を行わない。exp444 / exp367も再分類せず、
既存の独立した非acceleration仮説を優先する。
