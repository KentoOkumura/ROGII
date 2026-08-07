# exp400 セッションノート

## 目的

Kaggle discussion 728712で共有された `lik_pf` のGR観測ノイズ幅x1.3を、
exp072 deterministic likelihood-PFへ全well一律で移し、保存済みx1.0
controlとのpaired train-side auditとして原因分離する。

## 現在の状態

- Route: `pf_beam`
- 状態: train-side scientific gate FAIL・terminal close
- CV: `12.221810980460939`
- Public / Private LB: 未提出
- implementation: 承認済み・完了
- 正規train Notebook採用 / package / run: 承認済み・完了
- inference / submission: 未承認・未実施

## 2026-07-25 design-only

- `kaggle-review-exp`でdesign-only実験の構成と記録要件を確認した。
- `kaggle-strategy`で現行backlog、exp072、exp398、関連PF実験の
  結果と優先順位を確認した。
- `docs/agent-playbooks.md`と`docs/06_reproducibility.md`を確認した。
- `make new-steering EXP=exp400_all_well_1p3_sigma_gr_likelihood_pf`
  でsteering scaffoldを作成した。
- `make new-exp EXP=exp400_all_well_1p3_sigma_gr_likelihood_pf SOURCE=templates/experiment`
  でtemplate experiment scaffoldを作成した。
- requirements / design / tasklist、config、README、SESSION_NOTES、
  result、metrics、backlogをdesign-onlyに更新した。

## 固定した設計

- parent: exp072 deterministic v2 likelihood-PF
- treatment: 全773 wellsでclip後`gs`を1.3倍、再clipなし
- primary: `likpf_mean_x1p3`
- control: saved exp072 `likpf_mean`、再実行0
- secondary: scale 3/5/8/12、best選択なし
- PF: 500 particles × 128 stable seeds
- seed: `stable_seed("likpf", "train", well) + seed_index`
- fixed downstream guard: saved exp209 HMMとの50:50
- truth: candidate predictionとcontent SHA freeze後だけjoin
- implementation / run / inference / submission: design-only時点では別承認

## 実行量

設計上:

- scientific variants: 1
- candidate PF well-runs: 773
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- prediction readouts: 5
- reporting folds: 5
- LightGBM configs / trained folds / boosters: 0 / 0 / 0
- parent PF control / HMM / Beam reruns: 0 / 0 / 0
- CPU only、GPU / internetなし
- runtime上限: 30,600秒

このdesign-only時点ではすべて未実行。その後、承認済みcandidate PFだけを
version 1で実行した。

## 公開sourceの注意

現行リンク先Notebookでは後半の`lik_pf`にだけ`* 1.3`があり、
最終selectorから使われる別`run_particle_filter`のscaleはx1.0のまま。
よってexp400はfull public pipeline replayではなく、exp072に対応する
`lik_pf` mechanismのlocal deterministic transferとする。

## 再現性メモ

- stochastic components:
  particle初期化、process noise、resampling、roughening。
- stable seed:
  well ID / split / familyのSHA256 base + seed index。
- parallel:
  fixed 8 well threads、kernelへ明示seed、well/row/seed順固定。
- saved exp072 control:
  raw gzip SHA
  `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`、
  decompressed SHA
  `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`。
- prediction logical content SHA:
  `009a1d73e187c4126a70231214f14fbe1ae44edee47d9a166818ab1bd928a3bf`
- artifact manifest SHA:
  `59c877025e81713639c97822e14b6da1f77bee1d99274dc1f8e933d329ce8dfa`
- deterministic anchor: false。初回train-side候補でsubmission anchorではない。

## 2026-07-25 implementation-only

- ユーザーの「exp400を実装してください」をimplementation-only承認として記録した。
- 正規`*_train.ipynb` / `*_inference.ipynb`は上書きせず、次の別名候補を作成した。
  - `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_train.py`
  - `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_train.ipynb`
  - `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_inference.py`
  - `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_inference.ipynb`
- train候補は11章 / 2,016行。parent exp072にはcompact sourceがないため、
  1,598行の`public_notebook_replay_audit.py`からlikelihood-PF kernel、
  stable seed、input preparationを科学参照し、truth-late、gate、SHA、
  生成物orchestrationをself-containedに展開した。
- exp072と同じ乱数消費順を保つkernelへ、resampling count、minimum ESS、
  particle TVT clip countを乱数非消費のpassive diagnosticとして追加した。
- x1.0 synthetic fixtureでcandidate kernelのprediction / log-likelihoodが
  parent exp072 kernelとexact一致するcontract testを追加した。
- candidateは全wellで
  `gs_candidate = 1.3 * clip(nanstd(fillna(prefix_GR, 0) - TW_GR), 10, 60)`
  を1回だけ適用し、再clipしない。
- horizontal candidate loaderは`MD / Z / GR / TVT_input`だけを`usecols`で読み、
  candidate predictionとlogical content SHAをfreezeした後にだけraw suffix
  `TVT`、exp226 fold、exp115 role、saved controlを読む。
- primary direct、fold、raw observed/missing、high-missing、1000+、
  hidden-like 2面、by-well tail、fixed exp209-HMM 50:50の全gateを実装した。
- inference候補はsample submissionをコピーせず、全inference /
  submission flagがfalseであることを検証して必ず停止する。

### 保存exp072 scale列の実入力制約

固定raw SHA
`14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
のexp072 cache headerを確認したところ、`last_known_tvt`と`likpf_mean_d`はあるが、
`likpf_scale_3/5/8/12_d`は保存されていない。

primary saved x1.0 `likpf_mean`比較は設計どおり実行可能。x1.3 scale
3/5/8/12はcandidate-only nonselective diagnosticとして保存・評価し、
x1.0 scale別比較のためのparent PF再実行はしない。これはprimary promotion
gate、実行量、科学variant数を変更しない。

### 実装検証

- `py_compile`: train / inference PASS
- Ruff `--select F821`: train / inference / test PASS
- 専用test: `9 passed`
- 専用test + 共通Notebook / scaffold test: `20 passed`
- Jupytext train / inference変換と`--test`: PASS
- strict `validate-exp`: PASS
- `__file__`依存: 0
- compact train source / Notebook SHA:
  `446bfb5aab6ab231b22bb86e4e1c3965bbbbd62c11edb352abd558473ba06b00` /
  `c13cca08334a6cf1e67ba52c31770e9fb3803f602eddc1771cb5e913ed740adc`
- fail-closed inference source / Notebook SHA:
  `dbe0b6fdc870465eb5d857e044c6bf124d4b751326dc2d5aacede1f380e75c61` /
  `b00f7cf10187cd136aaa8b78b2f650577a54dc94639e50524b05acf44eb1e5f8`
- config / dedicated test SHA:
  `160c5e0a42a0c89f846c952f327baa8f052252b933755af1364d92d6fc38fa7f` /
  `fbf5b08949676fbb5bdb3d534e2d4f05e43613b1d7be17eefaa11cc6d30b2f70`
- canonical train Notebook SHA:
  `f36bf10ee2841b6026f137b532b019cc5131d35d2aad7d395ea80edb76221ac3`
- Kaggle push Notebook / metadata / embedded support zip SHA:
  `0bd8c460577415da652fabe596ca7acea831a3dcdf5130b7eb188af7a6bd8d21` /
  `224a10f8e1b88578a0993a4c8cf26724a19a8d9a1e2adaf5e313c8180f2349c8` /
  `b5b3d4d5147ab9565eb7247729a752e516efc9f728d4b487b06ac6d8d6cccc44`
- package内config / compact train sourceは正本とbyte-exact一致。
- Kaggle package: 1、PF実行: 1（complete）

## 意図的に未実行

- parent PF control / HMM / Beam / model再実行
- inference / submission

## 2026-07-25 Kaggle CPU実行承認

- ユーザーの「実行してください」により、正規train Notebook採用と
  Kaggle private CPU package / push / runを承認済みとして記録した。
- push前実行量:
  - scientific variant: 1
  - candidate PF well-runs: 773
  - seeds per well: 128
  - seed-well trajectories: 98,944
  - particles per seed: 500
  - particle starts: 49,472,000
  - prediction readouts: 5
  - reporting folds: 5
  - LightGBM config / trained fold / booster: `0 / 0 / 0`
  - parent PF control / HMM / Beam rerun: `0 / 0 / 0`
  - GPU / TPU / internet: off
- 保存済みexp072 PF、exp209 HMM、exp226 fold、exp115 roleをload-onlyで使う。
- canonical kernel:
  `kentookumura/exp400-all-well-1p3-sigma-gr-likelihood-pf-train`
- canonical title:
  `exp400 all well 1p3 sigma gr likelihood pf train`
- inference / submissionは今回の承認に含めない。
- canonical train Notebookを採用し、Jupytext round-trip、構文、F821、
  strict experiment validation、専用+共通test 20件を再検証して全てPASSした。
- canonical kernelのpush前pullは`403 Forbidden`で、既存の同名kernelが
  ないことを確認した。
- `2026-07-25T13:34:16Z`にprivate CPU / internet off /
  `run_on_push=true`のtrain packageを作成した。
- `2026-07-25T13:39:40Z`にkernel version 1をpushし、
  初期status `KernelWorkerStatus.RUNNING`を確認した。
- `2026-07-25T14:17:13Z`までversion 1が継続して
  `KernelWorkerStatus.RUNNING`であることを確認した。
- ユーザー指示により`2026-07-25T14:17:45Z`にassistant側のread-only
  status監視だけを終了した。Kaggle kernel自体は停止していない。
- 完了後のlogs、metrics、成果物確認結果は下の
  「2026-07-26 Kaggle version 1結果確定」に記録した。
- URL:
  https://www.kaggle.com/code/kentookumura/exp400-all-well-1p3-sigma-gr-likelihood-pf-train

## 次

Kaggle private CPU version 1のFAILをterminal記録し、inference /
submissionへ進まない。

## 2026-07-26 Kaggle version 1結果確定

- canonical kernel:
  `kentookumura/exp400-all-well-1p3-sigma-gr-likelihood-pf-train`
- version / id_no / status:
  `1 / 128585102 / KernelWorkerStatus.COMPLETE`
- metrics generated:
  `2026-07-25T16:32:53.664364+00:00`
- runtime:
  `10496.299889 sec`（約2.916時間、上限30,600秒以内）
- rows / wells:
  `3,783,989 / 773`
- candidate `likpf_mean_x1p3` RMSE:
  `12.221810980460939`
- saved exp072 control RMSE:
  `11.594894395642696`
- improvement:
  `-0.6269165848182432 ft`
- folds non-regressed:
  `1 / 5`。fold 3だけ`+0.099666 ft`で、fold 4は`-1.883659 ft`。
- required scope improvement:
  raw observed `-0.453077`、raw missing `-0.998656`、
  high missing `-0.884439`、1000+ `-0.708353`、
  hidden-like spatial/typewell-purged `-0.706604 / -0.738688 ft`。
- by-well:
  305 improved / 468 regressed、median `-0.179186 ft`、
  p95 regression `+5.059698 ft`、worst `+32.160524 ft`
  （well `708caea9`）。
- fixed exp209-HMM 50:50:
  candidate/control `10.659967680 / 10.269692505`、
  `0.390275 ft` regression。
- technical gate:
  PASS。全input SHA、773/773 fallbackなし、finite coverage 1.0、
  multiplier誤差0、post clip 0、実行量、control metric parity、
  truth-late ledgerがすべて一致。
- scientific gate:
  FAIL。
- decision:
  `all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`

### 生成物監査

- `kaggle/output/train_v1`へoutput archive全体ではなく、小型metrics /
  manifest / audit 10 filesとkernel logだけを取得した。
- artifact manifest:
  `59c877025e81713639c97822e14b6da1f77bee1d99274dc1f8e933d329ce8dfa`
- prediction logical content:
  `009a1d73e187c4126a70231214f14fbe1ae44edee47d9a166818ab1bd928a3bf`
- prediction raw gzip / decompressed:
  `cd4aa9ee5afa30b0047fdffaa74d808ad4b812708833ad691bac62d8f21d730a` /
  `64e82ee810a19074344e7bdf42ed63bd84af715c055b7bab31e3abe46211a2bd`
- manifest記録の小型8生成物は実ファイルSHAと全件一致した。
- Kaggle output `metrics.json` SHA:
  `a46c2003abb2c2169dd6626629d55da374ac5480dff03738c960d7cafa426d83`
- kernel log SHA:
  `ecaa45ce317c4b75e3be7d3007ca45fee9b933de97baf00802fcf6c476de6bc5`
- 86,759,305 bytesのcandidate prediction本体は、後続利用しないFAIL
  branchなので取得していない。

### 解釈と閉鎖

- secondary scale 3/5/8/12 RMSEは
  `11.271336 / 11.174615 / 11.243685 / 11.342899 ft`。
  保存x1.0 scale controlがないためcandidate-only nonselective
  diagnosticのままとし、scale 5をpost-hoc primaryへ差し替えない。
- 探索的well readoutでは、改善率はmissing fraction下位/上位四分位で
  `51.0% / 24.4%`、base `gs`下位/上位四分位で`56.2% / 28.5%`。
  一律にGR evidenceを弱めたことで、high-missing / high-base-scale側の
  有効な拘束まで失った可能性がある。
- この探索結果を同じOOFのadaptive multiplier / well gate救済には使わない。
  multiplier、clip、particle、seed、scale、resampling、blend、selector、
  version 2、inference、submissionなしでbranchを閉じる。
