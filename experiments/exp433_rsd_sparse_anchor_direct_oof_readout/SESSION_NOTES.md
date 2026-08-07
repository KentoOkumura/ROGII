# exp433_rsd_sparse_anchor_direct_oof_readout セッションノート

## 目的

exp426の凍結済みRSD scoreを実際のexp226 OOFへ直接適用し、疎なabsolute
anchorが累積offsetを改善するかを、offset一致率ではなく全row RMSEで判定する。

## 現在の状態

- Route: `pf_beam`
- 状態: compact実装完了、正規Notebook未採用、未実行
- 優先度: P2
- CV / LB: なし
- compact実装: 承認済み・完了
- 正規Notebook / Kaggle run: 未承認
- inference / submission: 無効

## 2026-07-28 設計セッション

### ユーザー依頼

ユーザーから、exp426の固定offset診断より実データへの直接適用を優先すべきとの
指摘があり、その後「バックログ、実験ディレクトリ、steeringを作成して
設計を確定。実装はまだ」と明示依頼された。

これをexp433のdesign-only作成承認として扱う。exp426のterminal-close判断、
score contract、生成物は変更しない。

### 作成

```bash
make new-steering EXP=exp433_rsd_sparse_anchor_direct_oof_readout
make new-exp EXP=exp433_rsd_sparse_anchor_direct_oof_readout
```

- steering:
  `.steering/20260728-exp433-rsd-sparse-anchor-direct-oof-readout/`
- experiment:
  `experiments/exp433_rsd_sparse_anchor_direct_oof_readout/`
- route:
  `pf_beam`
- status:
  `design_frozen_not_implemented`

### 確定した設計

- exp426 version 1のscore / support / rankをSHA固定入力として再利用する。
- RSD score、bin 0.5 ft、512-row block、固定13 offsetsは再生成・変更しない。
- primaryはexp426で事前登録済みのfixed Viterbi 1個だけ。
- unsupported blockはtransition-only carry、partial invalidは`-inf`。
- blockwise top-1はreport-only診断とし、primaryへ差し替えない。
- coverageはvalid offset数、well coverage、anchor gap、distance bucketを必須報告する。
- coverage不足でtruth readを止めず、prediction SHA freeze後に実際のtruthをjoinする。
- primary outcomeはexp226全OOF 3,783,989行のRMSE。
- exp226比、fold、1000+、persistent SSE、hidden-like、by-well tailをAND判定する。
- 結果を見たdecoder / threshold / support / clip / blend / activation救済は禁止。

### 実行量

- primary deterministic decoder: 1
- diagnostic blockwise replay: 1
- wells / reporting folds: `773 / 5`
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0`
- parent / control / score regeneration:
  0

### 戦略上の位置

- コンペはlate phase。
- ML Public-LB基準はexp335の`7.517`であり、本実験はそれを変更しない。
- PF/Beam routeの比較基準はexp226 OOF `9.427109596582213`。
- P2とし、P1のexp413 Stage Dを追い越すtrain/submission候補とは扱わない。
- ただし0-model・低コストでexp426の未回答点を直接反証できるため、
  高コストPFやexact HMMのP3候補より先に置く。

## 再現性メモ

- `docs/06_reproducibility.md`: 確認済み
- seed:
  RNGなし
- stochastic components:
  なし
- ordering:
  well / block / offset / rowをstable key順に処理・reduce
- runtime:
  Kaggle private CPU、GPU / internetなし、single worker予定
- input:
  exp426 score / well / input manifestとexp226 OOFのlogical /
  decompressed content SHAを固定
- output:
  support、datum path、prediction、metricsのlogical SHAを保存予定
- rerun:
  fixed probeと全decoderのindependent prediction SHA parityを必須化
- model / submission SHA:
  非該当
- deterministic anchor:
  未実行のためfalse

## 未実施

- compact self-contained source / tests
- 正規train / inference Notebook編集
- Kaggle package / push / run
- direct OOF evaluation
- inference / submission

## 次のアクション

compact候補をレビューし、別承認後に正規train Notebookへ採用する。
Kaggle runは正規Notebook採用後も別承認とする。

## 設計検証

- YAML / JSON parse: PASS
- `make validate-exp EXP=exp433_rsd_sparse_anchor_direct_oof_readout`:
  strict PASS
- experiment docs review:
  core evidence categories present
- `make update-summary`:
  exp426 -> exp433 lineageとfinal design-only statusを反映
- train / inference Notebook:
  template scaffoldのまま。exp433固有ロジックは未実装
- implementation / Notebook replacement / package / push / run /
  inference / submission authorization:
  すべてfalse

## 2026-07-28 実装セッション

### ユーザー承認と境界

ユーザーの「exp433を実装してください」を、compact self-contained train候補、
対応する未実行Notebook候補、contract tests、config / 文書更新の承認として
扱った。既存の正規train / inference Notebookは上書きせず、Kaggle package /
push / run、inference、submissionは実施していない。

### 実装

- Jupytext source:
  `exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.py`
- 未実行Notebook候補:
  `exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.ipynb`
- contract tests:
  `tests/test_exp433_rsd_sparse_anchor_direct_oof_readout.py`
- config:
  `implementation_ready_not_run`へ更新。implementationだけtrueとし、
  canonical replacement / package / push / runはfalseを維持した。

実装した固定処理:

1. exp426 score / well / input manifestをdecompressed / logical / schema SHAで照合
2. score / valid / rank / top-3を再生成せず、support診断だけを作成
3. initial sigma 5 ft、transition sigma 10 ft、first 20 ft、adjacent 40 ftの
   sole-primary Viterbiをstable tie orderで実行
4. suffix offset 0のcorrection 0から固定512-row block centerへ線形補間
5. support / datum path / row prediction SHAをfreezeし、全decoderを独立rerun
6. freeze後だけtruth、hidden-like role、persistent episodeを読む
7. full OOF / fold / distance / raw-GR / hidden-like / by-well / episodeを集計
8. technical / scientific AND gateを判定し、PASS / FAILともsame-OOF救済を禁止

corrected pathの新規persistent episode SSEは、corrected errorの絶対値が
10 ft以上で128行以上連続するrunを再検出し、そのうち凍結済みoriginal episode
union外の行だけをnew SSEへ加えるrow-level定義に固定した。

### 実行量

- primary decoder: 1
- diagnostic blockwise replay: 1
- planned wells / reporting folds: `773 / 5`
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0`
- parent / control / score regeneration:
  0

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp433_rsd_sparse_anchor_direct_oof_readout/\
exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.py \
  tests/test_exp433_rsd_sparse_anchor_direct_oof_readout.py
.venv/bin/ruff check \
  experiments/exp433_rsd_sparse_anchor_direct_oof_readout/\
exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.py \
  tests/test_exp433_rsd_sparse_anchor_direct_oof_readout.py
.venv/bin/pytest -q \
  tests/test_exp433_rsd_sparse_anchor_direct_oof_readout.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp433_rsd_sparse_anchor_direct_oof_readout/\
exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.py
```

- contract tests: `9 passed`
- syntax / ruff / ruff format: PASS
- Jupytext parity: PASS
- compact Notebook: 26 cells（Markdown 13 / code 13）、output 0
- `__file__`: compact sourceに残存なし
- 保存済みlate-input SHA:
  exp226 OOF decompressed SHA、persistent episode / hidden-like assignmentの
  file SHAをread-onlyで照合し一致
- 保存入力inventory:
  exp226 `3,783,989 rows / 773 wells / folds 0..4`、hidden-like 773 rows、
  persistent episode `645 episodes / 718,744 rows`
- `make validate-template`: PASS
- `make validate-exp EXP=exp433_rsd_sparse_anchor_direct_oof_readout`:
  strict PASS
- 親compact構成比較:
  exp426は11章 / 1,992行、exp433は12章 / 2,204行。exp433は入力、
  support、decoder、freeze、late join、全metrics / gateをNotebook上で追える。
- 正規train / inference Notebook: untouched placeholder
- Kaggle package / run / CV / LB: 未実施 / なし

全リポジトリの`make test`も実行したが、1,405 testsの実行前collectionで既存の
exp297 / exp301 / exp333 / exp336 / exp349 sourceが各experiment configを
解決できず5 errorsで停止した。exp433対象test 9件、strict validation、
Jupytext parity、ruff、py_compileは独立にPASSしており、既存5実験は変更していない。

### 次のアクション

別承認後にcompact候補を正規train Notebookへ採用し、package前にembedded config、
exp426 / exp226 / exp115 kernel sources、bootstrapされたpersistent episode SHAを
照合する。push前実行量は1 decoder / 1 diagnostic / 773 wells / 5 folds /
0 model・booster・HMM・PF・Beam・GPUを再確認する。

## 2026-07-28 Kaggle実行セッション

### ユーザー承認

ユーザーの「実行してください」により、compact候補の正規train Notebook採用、
Kaggle private CPU package / push / runを承認済みとした。inference、
submissionは引き続き未承認。

### push前実行量

- primary deterministic decoder: 1
- diagnostic blockwise replay: 1
- wells decoded / reporting folds: `773 / 5`
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0`
- parent / control / exp426 score再生成:
  0
- runtime:
  Kaggle private CPU、internetなし、single worker
- fixed input:
  exp426 version 1 score / well / input manifest、exp226保存OOF、
  exp115 hidden-like assignment、SHA固定persistent episode

既存の親・controlを再学習せず、保存済みOOFとscoreだけを読むため、
GPU control再学習の承認対象はない。

### canonical package

- kernel id:
  `kentookumura/exp433-rsd-sparse-anchor-direct-oof-readout-train`
- title:
  `exp433 rsd sparse anchor direct oof readout train`
- 正規train Notebook:
  compact self-contained候補を採用
- 正規inference Notebook:
  placeholderを維持
- run-on-push:
  true

### Kaggle version 1 technical ERROR

- kernel id_no / version:
  `128939253 / 1`
- status:
  `ERROR`
- 終了確認:
  `2026-07-28 14:31:06 UTC`
- bootstrap、実行承認、scientific contractはPASSした。
- 失敗位置:
  exp426凍結入力の読込直後。decoder、truth join、評価には未到達。
- error:
  `score_logical_sha`と`well_manifest_logical_sha`の2件だけ不一致。
- version 1 package SHA:
  - config:
    `02a714beef561e12ea816a8cb2debe477d19179bff722ad1d08871560aaf5daf`
  - notebook:
    `1359b68b748932a9d6ee1862cdc421b4c4415804c1d5fb1f0fc7105178d47688`
  - metadata:
    `1aa0e518bea6b46586b42e9c0ea5cf2b7719b6ebe8200089a047bf3bd9a46e0d`

実ファイル確認が必要なtechnical errorだったため、exp426 version 1 outputを
`/tmp/exp426-output-diagnose.dh5rE9`へ一時取得した。score / well manifest /
input manifestのdecompressed SHAはすべてconfigと一致し、score schema SHAも一致。
input manifestはproducer logical SHAも再現した。一方、float列を持つscore bankと
well manifestは、producerがCSV保存前のin-memory値から記録したlogical SHAを
CSV再読込後のfloat値から再現できなかった。観測したpost-read logical SHAは次の通り。

- score:
  `6eca191a00eeb10a032f192ea1cde4aefbedc59e385c7303660256857b3388b9`
- well manifest:
  `d7672fa743c4e46094c4a79d007b7e1de6a0204fe3da0615b5771bc7136711a8`

凍結artifactの変更や科学仮説の失敗ではなく、非round-trip-safeなproducer-side
logical SHAをconsumer-sideで再計算した実装契約の誤りと判定した。

### version 2最小修正

score / well / input manifestのproducer logical SHAは、exp426 summaryに記録された
固定lineageとしてscientific contractへ含める。一方、consumer-side artifact
identity gateは、完全一致するdecompressed SHA、typed schema SHA、row / block /
structure contractで判定する。post-read logical SHAは診断として記録するが、
float CSVの同一性gateには使わない。decoder、offset、transition、support、
truth-late freeze、評価gate、実行量は変更しない。同一canonical kernelを
version 2へ更新する。

version 2 push前検証:

- dedicated tests: `9 passed`
- Ruff check: PASS
- Jupytext round-trip: PASS
- strict `make validate-exp`: PASS
- exp426実artifact replay:
  `101,231 score rows / 773 well rows / 773 input rows`、全20 contract checks PASS
- 実行量:
  1 decoder / 1 diagnostic / 773 wells / 5 folds /
  0 model・booster・HMM・PF・Beam・GPU・parent regeneration
- version 2 package SHA:
  - config:
    `097b9d25d41d4202a8484a2809ef5232155b92bf54af2dcbbe83e70889da7741`
  - notebook:
    `c04f60cc9ddf3987e99c8290d41b21ff4ca492993ef3ad2b7ca3db5bd8a69a96`
  - metadata:
    `1aa0e518bea6b46586b42e9c0ea5cf2b7719b6ebe8200089a047bf3bd9a46e0d`

### Kaggle version 2 technical ERROR

- kernel id_no / version:
  `128939253 / 2`
- status:
  `ERROR`
- 終了確認:
  `2026-07-28 14:37:06 UTC`
- runtime:
  約`124.8 sec`
- version 1で失敗したexp426 input contractはPASSした。
- decoder、independent rerun、prediction freeze、truth / hidden / episodeの
  late joinを完了し、scope metricsの構築時に停止した。
- error:
  `ValueError: unknown report scope fold`

`validation.report_scopes`の`fold`は専用fold tableで集計する予約語だが、
scope table loopが`by_well`だけを除外し、`fold`を通常scopeとして二重に
routeしていた。科学予測やgateの失敗ではなくmetrics routingの実装欠陥である。

version 3ではscope loopから`fold`と`by_well`を除外し、それぞれ既存の専用tableへ
だけrouteする。全report scope、fold、by-well tableの分離を検証する回帰testを
追加する。decoder、prediction SHA、truth freeze、科学gate、実行量は変更しない。

version 3 push前検証:

- dedicated tests: `10 passed`
- Ruff check / format: PASS / PASS
- Jupytext round-trip: PASS
- strict package preparation: PASS
- 実行量:
  1 decoder / 1 diagnostic / 773 wells / 5 folds /
  0 model・booster・HMM・PF・Beam・GPU・parent regeneration
- version 3 package SHA:
  - config:
    `097b9d25d41d4202a8484a2809ef5232155b92bf54af2dcbbe83e70889da7741`
  - notebook:
    `2f612a71c3c467b0db0c8e6747d0fc6a45fe2e4cbe1a43bb215c24d1c3a781f5`
  - metadata:
    `1aa0e518bea6b46586b42e9c0ea5cf2b7719b6ebe8200089a047bf3bd9a46e0d`

### Kaggle version 3 terminal result

- kernel:
  `kentookumura/exp433-rsd-sparse-anchor-direct-oof-readout-train`
- kernel id_no / version:
  `128939253 / 3`
- status:
  `COMPLETE`
- experiment runtime / Notebook wall:
  `122.701148 / 200.432629 sec`
- peak RSS:
  `2.708778 GB`
- inventory:
  `3,783,989 rows / 773 wells / 5 folds / 7,787 blocks`
- support:
  `1,993 / 7,787 blocks = 25.593939%`、`690 / 773 wells`

technical gateは全PASSした。exp426 exact decompressed / schema / inventory、
exp226 parent RMSE、correction slope、runtime / memory、freeze前truth /
hidden / episode read 0、prediction / datum / fixed probeの独立rerun SHA parityを
すべて確認した。

primary direct OOF:

- base / primary RMSE:
  `9.427109597 / 9.692148252`
- gain:
  `-0.265038655 ft`
- improvement folds:
  `0 / 5`
- distance 1000+ gain:
  `-0.298535385 ft`
- persistent episode SSE reduction:
  `-2.797279%`
- persistent episode wells improved:
  `160 / 449 = 35.634744%`
- new episode SSE fraction:
  `5.228087%`
- by-well delta RMSE p95 / worst:
  `+3.282839 / +15.926322 ft`

fold gainは0--4で
`-0.188002 / -0.405441 / -0.350996 / -0.256130 / -0.115329 ft`。
near 0--50 / 50--100 / 100--500だけ
`+0.010300 / +0.026029 / +0.015655 ft`改善したが、500+、
raw-GR missing、hidden-like、persistent episodeを悪化させた。

scientific gateは全9条件FAIL。decisionは
`scientific_fail_close_sparse_anchor_branch_without_rescue`。
exp426の`technical_failed_closed`判断を維持し、same-OOF decoder /
transition / support / activation / clip / blend / well gate救済なし、
inference / submissionなしでbranchをterminal closeする。

再現性:

- prediction logical SHA:
  `c461a14708ffc951060a77e0016a7947f7e2cae1abeb28b539465c0289100377`
- datum path logical SHA:
  `e3b4f9afbe0f431c5f80add93f11abb15af44dbae64fd9511be579e2d8bef96e`
- fixed probe prediction SHA:
  `639fb28ff2397123b24d44fe3aaaa56570aa0840412f541072260a9f7af46b9a`
- row prediction raw / decompressed SHA:
  `11114f62215d7d55bce3bd9f790b279bb53032fc24c51c451d72641d0834524f` /
  `ff3147120055b854ebe5a49125e75c81c41cb40546ef46eab94768720f4bfc36`

train-side結果はKaggle logsとversion 3のsmall metrics / summary / scope /
fold / by-well / episode / support / block diagnosticだけを取得して確認した。
3.8M行prediction outputはダウンロードしていない。

最終記録検証:

- dedicated tests: `10 passed`
- Ruff check / format: PASS / PASS
- Jupytext round-trip: PASS
- strict `make validate-exp`: PASS
- `make validate-template`: PASS
- `make update-summary`:
  429 experimentsを更新し、exp433を
  `completed_scientific_failed_closed / CV 9.692148251575704`として反映
- `KAGGLE_DIRECTION.md`:
  完了結果を判断メモへ移し、exp433をアイデアバックログから削除
