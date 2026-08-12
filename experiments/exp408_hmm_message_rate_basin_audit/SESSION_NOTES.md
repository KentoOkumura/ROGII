# exp408_hmm_message_rate_basin_audit セッションノート

## 目的

exp209 exact HMMの長いoffsetがどのmessage段階で形成されるかを、persistent
638 episodes / 450 wellsのcurrent HMM再decodeで特定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU train version 3完了・原因監査完了
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- parity参照: `exp270_exact_hmm_posterior_mode_candidate_audit`
- steering:
  `docs/legacy/steering/20260726-exp408-hmm-message-rate-basin-audit/`
- CV / LB: 原因診断のため対象外
- inference / submission: 無効

## 2026-07-26 実行承認と固定契約

ユーザーの「実行してよいです。重い実験ならばローカルではなく、kaggleで
実行してください」を、current親HMM Stage AのKaggle private CPU実行承認として記録した。

- active scientific variants: 1（exp209 currentのみ）
- target wells / HMM well-runs: `450 / 450`
- expected suffix rows: `2,264,135`
- persistent episodes / rows: `638 / 807,710`
- LightGBM config / trained fold / booster / model: `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- position-width / exact-mean / momentum / emission treatment: 0
- 親control再計算: あり。既存exp209/270にalpha/predictive/beta/rate massが
  保存されておらず、同じcurrent passでしか観測できないため。
- base runtime見積り: `6,752.9 sec`（1.876 h）
- message集計込み保守的見積り: `10,800 sec`（3 h）
- hard guard: `32,400 sec / 25 GB`

## leakage boundary

- pre-decodeで読めるtruth由来情報は固定target well IDだけ。
- raw horizontalは`TVT`を`usecols`から除外して読む。
- exp270から読むのはid/well/row_idxとtarget-free decoder列だけ。
- HMM messageとposterior meanを生成し、parityとSHAをfreezeした後だけ、
  そのwellのepisode境界とraw `TVT`を読む。
- decoder/message kernelはtruth/error/episode/fold/hidden-role引数を持たない。

## 再現性

- RNGなし、outer worker 1、well/row stable sort、Numba thread 4。
- target-well manifest SHA:
  `ce245abce24dae98d37b6e0a2adf73fa57a29e0e53864bee983aa916238ea51e`
- persistent episode manifest SHA:
  `031067fa77c195b77920a0997401310fbdd16532a2d0e99a9c3b5044de28913c`
- Kaggle packageはpush前にloose/bootstrap configのbyte一致を確認する。
- deterministic submission anchorではない。model/submissionは生成しない。

## 2026-07-26 実装・push前検証

### Notebook構成

- exp270 parity shard source: 2,680行
- exp391 compact message/mode source: 3,407行
- exp408 compact self-contained train: 2,434行
- 10章構成で、path/SHA、target-free input、exp209 preprocessing、
  exact forward-backward、message freeze、truth-late basin、episode attribution、
  Kaggle orchestration、生成物保存をNotebook上に展開した。
- 同じexp内helper importはなく、`__file__`参照も0件。
- compact候補を正規`exp408_hmm_message_rate_basin_audit_train.ipynb`へ採用した。
  inferenceはfail-closed Notebookだけを置いた。

### 専用test

```text
7 passed
```

- exp209参照kernelとのsmall-trellis posterior / log-likelihood同値
- predictive / filtered / smoothed正規化
- fixed asset 450 wells / 638 episodes / 807,710 rows / SHA
- truth/episodeを受け取れないdecoder interface
- basin/rate interval mass
- local full-run fail-closed

py_compile、Ruff F821、Jupytext round-trip、strict `make validate-exp`もPASSした。
Notebookの初回実行はローカルでは行っていない。

### Kaggle package

- kernel id:
  `kentookumura/exp408-hmm-message-rate-basin-audit-train`
- title:
  `exp408 hmm message rate basin audit train`
- private / CPU / GPU無効 / internet無効 / run-on-push
- kernel sources:
  - `kentookumura/exp270-exact-hmm-posterior-mode-audit-train`
  - `kentookumura/exp226-k16-kappa-repro-train`
- 両sourceの必要生成物は`kaggle kernels files`で存在確認済み。
- local / bootstrap `config.yaml`: byte一致
- target-well / episode asset: bootstrap内byte一致
- final package Notebook SHA256:
  `dcbbaf7fbaa2ab8f384f99d6a6ab5e004db5b468e300e8c0c13b05b804bdefe0`
- final embedded config SHA256:
  `fd485be7de2e619bdb0fbba690d09b5cc5c70c9e2111254bc457126c435ac627`
- final bootstrap ZIP SHA256:
  `8ca69925c49904c66ebae5e18d02c5c8484e676f27dc090b5cb93fb172012eac`

### 全体test

repo全体は`1,178 passed / 7 skipped / 5 failed`。exp408専用7件はPASSした。
5 failureは既存exp293のcontract SHA 2件、完了後configと古い期待がずれたexp296
2件、exp407の実行承認flag期待1件で、exp408変更箇所ではない。

## 次のアクション

version 1を完了まで監視し、必要なrow/episode/summary生成物を取得して原因分類を監査する。

## 2026-07-26 Kaggle private CPU version 1開始

- push:
  `make push-kaggle-train EXP=exp408_hmm_message_rate_basin_audit`
- result: `Kernel version 1 successfully pushed`
- kernel:
  `kentookumura/exp408-hmm-message-rate-basin-audit-train`
- URL:
  <https://www.kaggle.com/code/kentookumura/exp408-hmm-message-rate-basin-audit-train>
- id_no: `128636642`
- start確認: `2026-07-26 02:18 UTC`
- initial status: `KernelWorkerStatus.RUNNING`
- pull後metadata:
  private / CPU / GPU無効 / TPU無効 / internet無効 /
  `machine_shape: None`
- competition source 1、kernel source 2を確認した。
- exp404 CPU version 1も実行中だが、exp408は別private CPU workerとして開始された。

### version 1 technical ERROR

- status: `KernelWorkerStatus.ERROR`
- failed at: 先頭wellの`prepare_hmm_inputs`、HMM well-runs 0
- error: `ValueError: horizontal missing ['id']`
- cause: exp209/270のraw horizontal必須列は`MD/Z/GR/TVT_input`だが、
  exp408で保存exp270側のidentity列`id`をraw側にも誤って要求した。
- fix: rawはexp209と同じ4列契約へ戻した。`id`が存在する環境では追加照合し、
  競技rawのように存在しない場合は`row_idx`とsuffix行数でexp270へ照合する。
- scientific HMM、対象well、message、basin、分類、実行量は変更なし。
- regression testを追加し、専用testは`8 passed`。
- 同じcanonical kernel IDへversion 2として再pushする。

### version 2開始

- result: `Kernel version 2 successfully pushed`
- same kernel / id_no: canonical exp408 / `128636642`
- scientific contract: version 1から変更なし
- package Notebook SHA256:
  `4cc13d863eeb8be580da32b1c092a36534e0e89a2859866a8e138ae37c565d99`
- embedded config SHA256:
  `d59c987e59bf636aeed390f5c96e36f33221b8e8cf9a363201e2c07187aa73bb`
- bootstrap ZIP SHA256:
  `6435377788b9668f22b2abd4ce2fff9b3f377b004fdba9cad9e1f21b0ca8a67b`
- local / bootstrap config、target-well / episode asset byte一致: PASS
- initial status: worker確認中

### version 2 technical ERROR

- 先頭wellのcurrent HMMは完了したが、保存exp270とのposterior mean parityが
  max `0.0546875 ft`、mean `0.0105164 ft`となり`1e-5 ft` gateで停止した。
- HMM well-runs: 1未満（先頭wellのfreeze前で停止）、truth/episode read 0。
- 原因はbackwardでjoint posteriorを保存する際、exp270のposition marginalが使う
  `p -> r`順のfloat64累積をNumPy `sum`へ置換したこと。長系列float32 messageでは
  reduction順の差がposterior meanへ蓄積した。これはexp391で見られた
  `0.35 ft`級parity failureと同型である。
- fix: exp270と同じnested accumulationとposition marginal正規化を完全に復元し、
  その同じtotalでjoint posteriorを別bufferへ保存する。診断用sufficient statisticsは
  HMM外で行正規化し、decodeへfeedbackしない。
- parity toleranceは緩めず`1e-5 ft`を維持する。
- 64-row small trellisでexp209参照posterior / log-likelihood同値を再確認し、
  専用test `8 passed`。
- scientific contract、対象、分類規則、実行量は変更なし。同じkernel IDのversion 3へ進む。

### version 3開始

- result: `Kernel version 3 successfully pushed`
- same kernel / id_no: canonical exp408 / `128636642`
- package Notebook SHA256:
  `5b3a567bb9157e091eaddc592512ee162ca6d03885388a8dcd0f3e9651b856d1`
- loose package config SHA256:
  `8ea408189c81e12e15828869ffc41787fef33ebc54ed4cdd49ffe03fdf7ec2bb`
- scientific contract: version 1/2から変更なし
- initial status: worker確認中

## 2026-07-26 Kaggle private CPU version 3最終結果

- kernel / version / id_no:
  `kentookumura/exp408-hmm-message-rate-basin-audit-train / 3 / 128636642`
- status: `COMPLETE`
- scope:
  `450 wells / 2,264,135 suffix rows / 638 episodes / 807,710 episode rows`
- elapsed / peak RSS:
  `15,930.997034 sec / 3.587806702 GB`
- HMM elapsed合計 / median / p90 / max:
  `11,721.965517 / 25.164083 / 33.696330 / 50.824171 sec`
- well elapsed合計 / median / p90 / max:
  `15,836.305779 / 34.262651 / 45.397003 / 68.348722 sec`
- posterior mean parity max abs diff: `0.0 ft`
- message normalization max abs error: `5.3375494e-8`
- truth / episode read before well freeze: `0 / 0`
- technical gate: `11 / 11 PASS`
- execution:
  `1 current HMM variant / 450 well-runs / model・LightGBM config・trained fold・
  booster・PF・Beam・GPU各0`

### 排他的cause

| cause | episodes | wells | rows | SSE fraction |
| --- | ---: | ---: | ---: | ---: |
| forward transition / prior hysteresis | 452 | 350 | 557,692 | 0.593977875 |
| backward smoothing reversal | 86 | 72 | 129,308 | 0.230443883 |
| sum-product path multiplicity | 37 | 35 | 60,641 | 0.090396484 |
| state support shortage | 18 | 18 | 23,113 | 0.063949086 |
| mixed / unresolved | 45 | 37 | 36,956 | 0.021232672 |
| raw GR alias | 0 | 0 | 0 | 0 |
| imputation alias | 0 | 0 | 0 | 0 |

重複あり条件ではforwardが469 episodes / SSE `0.657812289`、
multiplicityが276 / `0.720914892`。forward exclusive causeのfold別SSE比は
`0.494360--0.668603`、backwardは`0.183323--0.261217`で5 foldsに再現した。

### Row message readout

- predictive wrong strong:
  `0.703492590 rows / 0.691486456 SSE`
- current emission hurts truth strong:
  `0.002533087 / 0.009238062`
- emissionがtruth優勢からwrongへ新規反転: `9 rows`
- beta hurts truth strong:
  `0.675873767 / 0.669675617`
- smoothed wrong strong:
  `0.890818487 / 0.898347325`
- filtered rate zero-directed under-response:
  `0.709073801 / 0.703579808`
- current transition errorがoffsetと同方向:
  `0.633981256 / 0.670193200`
- backwardでrate massが回復しposition massが悪化:
  `0.433341174 / 0.383313137`
- quantization biasがoffsetと同方向:
  `0.329686397 / 0.282397456`

episode平均のcurrent displacement errorとoffsetはSpearman
`0.569283742`、符号一致`0.741379310` / SSE加重`0.902246087`。
quantization biasはSpearman`-0.389232844`、符号一致`0.297805643` /
SSE加重`0.208492193`だった。

### GR threshold感度

current emissionがepisode行の25% / 50% / 75%以上でtruth oddsを悪化させる件数と
SSE比:

- effect `0.1`: `28 / 6 / 2 episodes`,
  SSE `0.033491 / 0.000744 / 0.000196`
- effect `ln(1.5)`: `2 / 0 / 0`,
  SSE `0.000156 / 0 / 0`
- effect `ln(2)`または`ln(3)`: 全条件0

一方、forward wrongはeffect `ln(3)`でも
`538 / 469 / 395 episodes`、SSE
`0.855661 / 0.657812 / 0.515926`。backward reversalはeffectを
`0.1--ln(3)`で変えても50%条件が86件、SSE`0.230444`で不変だった。

### Artifact取得とローカルreadout

Kaggle CLI 2.2.3の`kernels output`は`stream=True`でも
`download_response.content`を使うため、860MB row ledgerの一括取得はexit 137になった。
これはKaggle run失敗ではない。`studies/stream_kaggle_kernel_output.py`で
8MiB chunksへ切り替えて定数メモリ取得し、raw / decompressed SHAを双方照合した。

- row ledger bytes: `860,095,821`
- raw SHA:
  `97c86f4907ec2a65200a8f83dce239cf180c33ee33c15974b0d5bf2a0a7bdde7`
- decompressed SHA:
  `74bb3c6b5593c3e01065b9feb81d4f76ee5133eef67a8e8972df22eb61ad2ffb`
- episode summary SHA:
  `b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`
- cause summary SHA:
  `53d0033f3b940585d1ffbfaac7fb1d0d56219e8bb92288a639ad7c353160d1f7`
- well manifest SHA:
  `5cd80f0e6732eafbc7edd4de45db702c5d673217e18e5fdce0db15a7079bdc3a`
- Kaggle小型artifact:
  `artifacts/kaggle_v3/`
- chunk row readout:
  `artifacts/readout_v3/`
- large local ledger:
  `/tmp/exp408_row_ledger_v3.csv.gz`（Git対象外）

`studies/analyze_exp408_message_audit.py`はwide ledgerを20,000行ずつ読み、
807,710 rowsのSSEとepisode keyを欠落なく再集計した。

## 最終判断

current rowのGR matchingがdominantな即時mode switchを起こすという全体仮説は反証された。
主因はhistoryを含むforward position-rate priorとrate under-responseで、
future betaとsum-product multiplicityがwrong datum massを増幅する。
GR aliasはcandidate-strong 180 episodes / SSE `0.334158`に存在するが、
同群のSSEもexclusive forward `0.743055`、backward `0.225465`であり、
弱い相関evidenceのhistory seed / lock増幅器として位置付ける。

local packageは`run_stage=none`、execution approval / rerunをfalseへ戻し、
inference / submissionを引き続き無効とする。

## 完了後package再生成

2026-07-26、Kaggle version 3完了後の正規状態からtrain packageをstrict再生成した。
pushはしていない。再生成後も`run_stage=none`、
`kaggle_execution_approved=false`、`rerun_enabled=false`、
`run_on_push=false`、GPU / internet falseを確認した。

- `kernel-metadata.json` SHA:
  `e1be049b648061f1f07038261f44a42ca339d4ade92b656457fb3e3ddf88998c`
- packaged `config.yaml` SHA:
  `49ffdb352772a609a51d4c22ffbd70713c86f9d87753f9bde12f8505e95a2d1f`
- packaged train notebook SHA:
  `53b0a629a783825a02b6e63ae192d3178966f0b79ccc208e0e1b07e03fd5ed0b`
