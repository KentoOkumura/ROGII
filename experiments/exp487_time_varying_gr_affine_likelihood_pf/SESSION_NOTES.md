# exp487_time_varying_gr_affine_likelihood_pf セッションノート

## 目的

exp345 causal / exp350 RTS affine scheduleをexp404 PF emissionへ移植し、
fixed32 Stage 0をKaggle CPUで実行する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage0_authorized_pending_push`
- Priority: P3・高リスク
- 実装: 完了
- canonical train採用、Kaggle package/push/Stage 0: 2026-07-30ユーザー承認済み
- Stage 1、inference、submission: 未承認

## 固定variant

1. `causal_ekf_affine_emission`
2. `bidirectional_rts_affine_emission`

base pathは保存exp209 mean/std。PF sigmaはaでscaleせずexp404 x1.0を維持する。
二variantは独立評価し、same-OOF winner selectionは行わない。

## 実行契約

- Stage 0: 64 PF wells、8,192 seed-well、4,096,000 particle starts。
- Stage 1: 1,546 PF wells、197,888 seed-well、98,944,000 particle starts。
- base HMM / control PF / Beam / model / booster / GPU rerun 0。
- 今回実行するのはStage 0の64 PF well-runsだけ。Stage 1は実行しない。

GPU学習は行わない。active scientific variant 2、LightGBM config 0、fold学習0、
booster 0、親HMM/control PF再実行0である。

## 再現性

base path、process noise、schedule/covariance、predictionをSHA freezeし、
その後だけtruth/fold/roleをattachする。PF seedはexp404 stable policy。

## 実装

- `exp487_time_varying_gr_affine_likelihood_pf_compact_selfcontained_train.py`
  - 保存exp209 mean/stdのSHA検証とrow identity alignment。
  - exp345と同じrobust prefix affine、outer-fold process-noise shrinkage、
    current-row posterior timing、missing raw GR skip、Joseph covariance。
  - exp350と同じidentity transition、`pinv(rcond=1e-12)`、terminal parity、
    covariance projectionを持つfixed-interval extended RTS。
  - exp404と同じposition/rate dynamics、500 particles、128 common seeds、
    ESS resampling、roughening、temperature 5。変更は
    `a_t * TypeWellGR(TVT_particle) + b_t`だけで、sigmaは再scaleしない。
  - causal/RTS schedule、PF ledger、predictionをdeterministic gzipとlogical SHAで
    freezeしてからtruth/control/fold/hidden-like roleをattachする。
  - fixed32 Stage 0 technical gateと全773 wells Stage 1 independent gateを実装。
- compact inference候補はcurrent-test exp209 pathとdeployment process-noiseが
  未実装であることを検証し、常にfail-closedとした。
- 2026-07-30の「実行してください」をcanonical train採用、Kaggle package/push、
  Stage 0実行の承認として記録した。canonical inferenceは変更しない。

## 親compactとの比較

- exp345 train 2,189行、exp350 train 2,785行、exp404 train 2,174行に対し、
  exp487 compact trainは3,361行。
- exp487はruntime/config、input/SHA、prefix/process noise、EKF、RTS、PF、
  target-free freeze、truth-late Stage 0、Stage 1 gate、orchestrationを14章で
  notebook上に展開しており、同一exp helperをimportする薄い構成ではない。

## 検証

- `task validate-exp ...`: `task`未導入のため実行不可。
- `make validate-exp EXP=exp487_time_varying_gr_affine_likelihood_pf`: PASS。
- `jupytext --to ipynb` train/inference: compact候補を生成。
- `jupytext --to ipynb --test` train/inference: PASS。
- `python -m py_compile` train/inference: PASS。
- `ruff check` train/inference/test: PASS。
- `pytest -q tests/test_exp487_time_varying_gr_affine_likelihood_pf.py`:
  15 tests PASS。exp350 forward EKF / RTSとのexact frame parityとexp404
  identity-affine PF bitwise parityを含む。
- dedicated + `test_kaggle_notebooks.py` + `test_scaffold.py`: 26 tests PASS。
- `make validate-template`: PASS。
- ローカルPF full runは実施しない。Kaggle Stage 0 package/push/runを開始する。

## 2026-07-30 Stage 0実行前記録

- 承認根拠: `user_message_2026_07_30_execute_exp487`
- active scientific variants: 2
- LightGBM configs / folds / boosters: 0 / 0 / 0
- Stage 0 candidate PF well-runs: 64
- seed-well trajectories / particle starts: 8,192 / 4,096,000
- parent base HMM / control PF / Beam / GPU rerun: 0 / 0 / 0 / 0
- planned kernel: `kentookumura/exp487-time-varying-gr-affine-likelihood-pf-train`
- planned title: `exp487 time varying gr affine likelihood pf train`
- runtime: Kaggle private CPU、8 workers、internet/GPU off
- strict package生成: PASS。metadataはprivate、CPU、internet off、
  `run_on_push=true`、canonical id/title slug一致。
- canonical train SHA256:
  `74806da8cc2c4a6087137052b938f19e4c1c0c5e7b9b47ea5b5f99982d267535`
- push package notebook SHA256:
  `d088176541d12dd665547b7d459659eecdc7dbb126b4e399da85d4775d4a8019`
- fixed32 manifest raw SHA256:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- 同時確認時点でexp484/485/486のCPU notebookがRUNNING。重複pushせず
  exp487のcanonical idへ初回versionを追加する。
- Stage 1、inference、submissionはfail-closedのまま。

## 次のアクション

canonical trainを生成・静的検証し、Kaggle packageをstrict生成して同一canonical
kernelへpushする。完了まで監視し、Stage 0 gateを記録する。

## Kaggle CPU枠待ち

- 初回push試行はKaggle APIの
  `Maximum batch CPU session count of 5 reached`で未受理。
- exp487 kernelは作成・実行されておらず、version重複はない。
- その時点の5件はexp483、exp484、exp485、exp486、exp490で、いずれも
  `KernelWorkerStatus.RUNNING`。
- 他実験を停止せず、1枠解放後に同じ検証済みpackageとcanonical idで再試行する。

## Kaggle Stage 0 version 1

- exp490が`2026-07-30 14:36:37 UTC`にCOMPLETEとなり、CPU枠が1つ解放。
- 同じ検証済みpackageをcanonical kernelへ再pushし、version 1作成に成功。
- kernel:
  `kentookumura/exp487-time-varying-gr-affine-likelihood-pf-train`
- id_no / initial status: `129180524` / `KernelWorkerStatus.RUNNING`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp487-time-varying-gr-affine-likelihood-pf-train`
- remote metadata:
  private、CPU、GPU/TPU/internet off、competition source 1、
  exp404 dataset source 1、exp209/exp226/exp115 kernel sources 3。
- remote pull notebook SHA256:
  `a77bc03474369c6cc4c07c86ea863ddc0cc882bf858bf5b7efdd4d48bf09b7ea`
- 同じversion 1を完了まで監視する。実行中の空logsやstatus API一時エラーを
  理由に再pushしない。

### version 1 ERRORと修正

- terminal status: `ERROR`。
- 科学処理前のraw identity guardで停止し、PF well-runは0。
- 原因: expected
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
  はexp485/486と同じtyped dataframe content SHA契約だが、exp487だけ
  CSV-text SHA helperを呼んでいた。
- local 773-well rawでexp485のtyped契約を再計算し、wells 773と期待SHAの
  exact一致を確認した。
- scientific variants、schedule、PF、seed、gate、input fileは変更しない。
  raw identity aggregationだけを同じtyped契約へ修正し、回帰testを追加する。
- 修正・再検証・strict再package後、同じcanonical kernelへversion 2をpushする。
- version 2 pre-push validation:
  - dedicated + notebook/scaffold tests: `27 passed`
  - Jupytext roundtrip / ruff / py_compile / strict exp validation: PASS
  - compact source SHA:
    `25a014bb71a8c9cdf411ca136a32965b6064afebf2c10d6135e40fe61ec3b970`
  - canonical notebook SHA:
    `d5bf6970dbc2b79a07ddd300bdd82754449624695eb964cfcf7e50e68c937051`
  - push package notebook SHA:
    `fb363dbfc4563658c755ee3983861482eb70a877e2d6b90fa1f15750e4fb0629`
  - executed config SHA:
    `3a69d3f1058397bfe54ac38559046b8e1b153d54b4d2603fe383918a0d1a0bb0`
  - kernel metadata SHA:
    `7d2bf783af0c11d5d6206b326e45dcb9464bba7095dba85f2f65b2821f59b018`

### version 4実行

- same canonical kernel version 4 push: 成功。
- id_no / initial status: `129180524` / `KernelWorkerStatus.RUNNING`
- remote pull notebook SHA256:
  `e5eba7079eff95a5328ad5268d28eefa7df5fe85f2c18ea730b3d4c7fb5ec66c`
- remote metadataでprivate、CPU、GPU/TPU/internet off、入力4件を再確認。
- version 4だけを完了まで監視する。

### version 4 ERRORと修正

- terminal status: `ERROR`。
- fixed32 32 wells×2 variants、8,192 seed-well trajectories、
  4,096,000 particle startsを完了し、全prediction/schedule/PF ledgerをfreezeした。
- truth-late saved exp404 control読込のlogical SHA guardで停止。
- raw gzip SHAとdecompressed SHAは一致しており、入力artifact自体は正しい。
- 原因: configのlogical SHAはexp404のfloat32 pre-serialization provenanceだが、
  exp487がCSV readback dataframeのtext SHAとして比較していた。
- exp484と同じくraw/decompressed artifact SHAを実行guardとし、logical SHAは
  pre-serialization provenanceとしてinput manifestへ記録する。
- control列契約、重複、finite、fixed32 ID coverageのguardは維持する。
- pre-serialization logical SHAがreadback SHAと異なってもraw/decompressed SHAが
  一致すれば通る専用回帰testを追加し、version 5へ進む。

### version 3実行

- same canonical kernel version 3 push: 成功。
- id_no / initial status: `129180524` / `KernelWorkerStatus.RUNNING`
- remote pull notebook SHA256:
  `d6c3b80a749afd7f13b37916e9a69e2480b99bd4a3e77484db804d0c84d52ed8`
- remote metadataでprivate、CPU、GPU/TPU/internet off、入力4件を再確認。
- version 3だけを完了まで監視する。

### version 3 ERRORと修正

- terminal status: `ERROR`。
- raw identity、773-well process noise、Numba warm-upを通過し、fixed32 PFを
  少なくとも15 wellsで開始した。
- 最初に完了したworkerのcausal schedule schema guardで停止。
- 原因: `decorate_schedule`が`id`、`well_id`、`suffix_offset`を先頭insertし、
  固定schemaが要求する`id`、`well_id`、`row_idx`、`suffix_offset`へ
  明示reindexしていなかった。列集合と値は同じ。
- decorate時にvariant別の列集合を検証して固定順へreindexする。
  schedule値、PF予測、科学契約、入力、seed、gateは変更しない。
- causal/RTS両方のexact column-order testを追加し、再検証後version 4へ進む。
- version 4 pre-push validation:
  - synthetic one-well decode → schedule/PF ledger freeze → SHA readback: PASS
  - isolated Numba warm-up compiled signatures: `1 / 1`
  - dedicated + notebook/scaffold tests: `28 passed`
  - Jupytext roundtrip / ruff / py_compile / strict exp validation: PASS
  - compact source SHA:
    `a83f64ad899909c5a777268e831ed837643585641d6284c80bc8ad89b00b04e1`
  - canonical notebook SHA:
    `3720e33d4da9a58f150bb4e80ed9ed3df52692425aa05c6c6868a95a11e56fde`
  - push package notebook SHA:
    `c9e4c67149ad2ee4fe9418b3fdae59890fe20ba2134439287602731a6848450b`
  - executed config SHA:
    `75651fad7d293fd0ec51ded3214ea7af738818a069d951e42665e658a23d1f2c`
  - kernel metadata SHA:
    `7d2bf783af0c11d5d6206b326e45dcb9464bba7095dba85f2f65b2821f59b018`

### version 2実行

- same canonical kernel version 2 push: 成功。
- id_no / initial status: `129180524` / `KernelWorkerStatus.RUNNING`
- remote pull notebook SHA256:
  `d77597f3bca6a2e533ead0a83f5423dd75a71f989d4a4e1ebbae32b1646e0c07`
- remote metadataでprivate、CPU、GPU/TPU/internet off、exp404 dataset、
  exp209/exp226/exp115 kernel inputsを再確認した。
- version 2だけを完了まで監視し、空logsを理由に再pushしない。

### version 2 ERRORと修正

- terminal status: `ERROR`。
- raw identity guardと773-well outer-fold process-noise生成は通過した。
- PF warm-upのNumba nopython compileで
  `Untyped global name '_interp1'`により停止し、candidate PF well-runは0。
- 原因: 親exp404では`_interp1`とPF kernelの両方を`@njit`しているが、
  exp487ではPF kernelだけをJIT対象にしていた。
- `_interp1`へ親と同じ`@njit(cache=True)`を付ける。PF式、乱数、入力、
  schedule、gateは変更しない。
- source contract testにdecorator sentinelを追加し、Numba有効環境でwarm-up
  compile smokeを行ってからversion 3を同じcanonical kernelへpushする。
- version 3 pre-push validation:
  - isolated Numba environment warm-up: PASS
  - `_pf_affine_allseeds` / `_interp1` compiled signatures: `1 / 1`
  - dedicated + notebook/scaffold tests: `27 passed`
  - Jupytext roundtrip / ruff / py_compile / strict exp validation: PASS
  - compact source SHA:
    `ba4a7d22241d40ecfff519b01d9e398601afadb52a52eda6509c5091bd8f599b`
  - canonical notebook SHA:
    `ccf6456032c7a735b10af6760f6a73fde7dbfe3cde6acaadbf2ed2ef437d68d4`
  - push package notebook SHA:
    `beaa83467ea853f10c3587f3d3a442974b8a03a174e4c5adbf47d1064be44826`
  - executed config SHA:
    `26f60670006d8c5e83a5d792b44533ea893349864bb4c1a8f811141b561a7e41`
  - kernel metadata SHA:
    `7d2bf783af0c11d5d6206b326e45dcb9464bba7095dba85f2f65b2821f59b018`

### version 5 pre-push validation

- scientific contractはversion 4から変更しない。saved exp404 controlの
  raw/decompressed artifact SHAを実行guardとし、float32 pre-serialization
  logical SHAはprovenanceとしてinput manifestへ記録する修正のみ。
- synthetic one-well decodeからprediction/schedule/PF ledger freeze、
  SHA readback、truth/control/fixed32 roleのlate attachまでPASS。
- isolated Numba environment warm-up: PASS。
- `_pf_affine_allseeds` / `_interp1` compiled signatures: `1 / 1`。
- dedicated + notebook/scaffold tests: `29 passed`。
- Jupytext roundtrip / ruff / py_compile / strict exp validation: PASS。
- compact source SHA:
  `e1c5b156348db5b2826bed4c7b0104b16b61d1a52455fa76e71e084d22b27216`
- canonical notebook SHA:
  `d214f6de24dddea0b77eadc73ffc33c23824fe652adfa0a931c52f800b083fdd`
- push package notebook SHA:
  `74541686560d47f8003e41e134d94c9f8b600d72bd19807b6d07966775a2d4e8`
- executed config SHA:
  `7a35493e85634b2d4295bbeb6903dda0e976f9fa18b169eb06f7d1cc2fb9e43b`
- kernel metadata SHA:
  `7d2bf783af0c11d5d6206b326e45dcb9464bba7095dba85f2f65b2821f59b018`

### version 5実行

- same canonical kernel version 5 push: 成功。
- id_no / initial status: `129180524` / `KernelWorkerStatus.RUNNING`。
- remote pull notebook SHA256:
  `9af3080cd261ee664d91619b5c7b4abf9927d5e8f4c072cb662a0be3e70b76c7`
- remote metadataでprivate、CPU、GPU/TPU/internet off、competition source 1、
  exp404 dataset source 1、exp209/exp226/exp115 kernel sources 3を再確認。
- version 5だけを完了まで監視し、terminal後にStage 0 gateを記録する。

### Stage 0完了

- terminal status / time:
  `KernelWorkerStatus.COMPLETE` / `2026-07-30 23:22:53 UTC`。
- fixed32 32 wells × 2 variants = 64 candidate PF well-runs、
  8,192 seed-well trajectories、4,096,000 particle startsを完了。
- control PF / HMM / Beam / model / booster / GPU rerunはすべて0。
- fixed32 descriptive RMSE（CVではない）:
  - causal EKF affine emission: `12.634359565368019`
  - bidirectional RTS affine emission: `13.391424323137391`
  - saved exp404 control: `9.616740808061033`
- causal / RTSはcontrol比でそれぞれ`-3.017618757 ft`、
  `-3.774683515 ft`。RTSはcausalより`0.757064758 ft`悪かった。
- technical checks 15/15 PASS、`stage0_gate.all_pass=true`。
- truth-access ledger:
  - expected / frozen variant-wells: `64 / 64`
  - before freeze truth / error / outcome fold / hidden role rows:
    `0 / 0 / 0 / 0`
  - after freeze truth / control / outcome fold rows:
    `156088 / 156088 / 32`
- fallback wells `0`、causal / RTS scale clip最大率はいずれも`0`、
  causal boundary jump sigma p95は`0.0049737333782097735`。
- elapsed `1191.0875504016876 sec`、peak RSS `1.8486404418945312 GB`、
  full projection `28772.208639390767 sec`。
- scientific contract SHA:
  `18743aff469f4ca1a410fdc3dda62261faccdbda020410185f43f062f83a79e3`
- input manifest SHA:
  `82a39670ba6c69d944f6f4832499b211bacda2f6f14ccfca014816b709cff743`
- freeze manifest SHA:
  `1366aca8282bd003e56df255e96e3c87a499fab57838643581e7744051597aa2`
- process-noise logical SHA:
  `aae37e5eecfc220d6c345b96000ddba395bc7eb2ecd2a8b1011f8fae1e16bca8`
- Stage 0契約上はStage 1 eligible。ただしfixed32記述値は両variantとも
  controlを明確に下回り、性能見通しは弱い。
- Stage 1、inference、submissionは実行せず、別承認待ち。
  完了後にpush / Stage 0実行フラグをすべてOFFへ戻した。
