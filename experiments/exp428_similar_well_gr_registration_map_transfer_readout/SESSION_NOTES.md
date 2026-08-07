# exp428_similar_well_gr_registration_map_transfer_readout セッションノート

## 目的

GR波形が似たwell間で、他wellのType Well–Horizontal GR registration offsetを
再利用できるかを、exp423のtruth-warp転写とは分離して設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 technical FAIL / no-rescueで閉鎖
- CV / LB: technical coverage不足のため無効 / 対象外
- 正規train notebook: compact self-contained実装を採用
- 正規inference notebook: 対象外のscaffold placeholder
- Jupytext source / test: 実装済み / 15件PASS
- Kaggle package / run: version 2まで完了
- inference / submission: 対象外・未承認

## 2026-07-28 設計セッション

### 実行したscaffoldコマンド

```bash
make new-steering EXP=exp428_similar_well_gr_registration_map_transfer_readout
make new-exp EXP=exp428_similar_well_gr_registration_map_transfer_readout
```

steeringを実験scaffoldより先に作成した。

### 確定した解釈

- donorから転写するのは正解TVT pathではなく、donor内で測るGR registration offset。
- primaryは似たdonorのglobal shift（ft）。
- local shift曲線、stretch、local warpはmapping shape診断。
- same-Type-WellはType Well GR波形、donor順位はHorizontal GR波形で決める。
- Type Well CSVのrow lagはtrim差を含むためft補正に直接使わず、exact-overlapのTVT差を使う。
- registration offsetをTVT補正量として直接使わない。

### Type Well axisのtarget-free確認

`exp065` pair artifactを読み、axis graphに使うedge条件を確認した。

- native-overlap pair candidate: 10,713
- `exact_match_rate=1.0` edge: 10,697
- exact edgeの`TVT_b-TVT_a`: 全件0 ft
- exact edgeのうちrow-lag ftが非ゼロ: 10,656
- exact edgeのTVT delta min/max span: 0 ft

非exact candidate lagには同一pairの近傍lagも含まれるため、axis graphでは除外する。
この確認から、row lagをregistration shiftへ直接流用しない契約を固定した。

### 固定した実行量

- audit variant: 1
- reporting folds: 5
- LightGBM config: 0
- trained fold: 0
- booster: 0
- PF / HMM / Beam well-run: 0 / 0 / 0
- GPU run: 0
- parent/control rerun: 0

## 再現性メモ

- seed policy: real matchingは乱数なし。random controlのみstable SHA256 per query well。
- stochastic components: なし。
- runtime: CPU-only、single process、GPU/AMP/internet off。
- query truth: primary/control/artifact freeze前read count 0を必須assert。
- donor truth: outer-train donor自身のregistration map推定だけに使用。
- SHA: raw input、config、fold/row inventory、Type Well axis graph、schema、
  target-free logical content、gzip decompressed contentを記録する設計。
- model / prediction / submission SHA: 対象物を作らないためnot applicable。
- deterministic anchor: 独立rerun一致前は不可。実行後もsubmission anchorではない。

## 今回行っていないこと

- inference、submission

## 2026-07-28 実装セッション

追加依頼「exp428を実装してください」により、design-onlyの承認境界を実装まで解除した。
固定済みのshift / block / similarity / support / gateは変更していない。

### 実装内容

- notebook-safeなcompact self-contained train sourceを新規作成。
- exact-overlap TVT差だけを使うType Well axis graphとcycle/conflict guard。
- donor outer-train truthだけで作る512-row、固定13 shiftのregistration map。
- suffix GRのrobust preprocessing、256点constrained DTW、rank-1/global primary。
- zero、stable-random、same-group median、post-freeze top-5 oracle。
- target-free well/block/donor/ranking生成物のlogical SHA freeze。
- freeze後だけ開くquery truth reader、query reference map、ZNCC / hidden-like /
  by-well / local-shape / mapping-shape readout。
- technical / scientific / local-shapeの固定AND gateとfail-closed decision。
- 正規train notebookへ採用。inference notebookはplaceholderのまま。

### 検証

```bash
.venv/bin/pytest -q tests/test_exp428_similar_well_gr_registration_map_transfer_readout.py
.venv/bin/python -m py_compile experiments/exp428_similar_well_gr_registration_map_transfer_readout/exp428_similar_well_gr_registration_map_transfer_readout_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp428_similar_well_gr_registration_map_transfer_readout/exp428_similar_well_gr_registration_map_transfer_readout_compact_selfcontained_train.py tests/test_exp428_similar_well_gr_registration_map_transfer_readout.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp428_similar_well_gr_registration_map_transfer_readout/exp428_similar_well_gr_registration_map_transfer_readout_compact_selfcontained_train.py
make validate-exp EXP=exp428_similar_well_gr_registration_map_transfer_readout
```

- 専用test: `14 passed`
- `py_compile`: PASS
- Ruff `F821,E501`: PASS
- Jupytext round-trip: PASS
- strict experiment validation: PASS
- `task validate-exp`は環境に`task`実行ファイルがなく使用不能だったため、
  Makefile同等コマンドで検証した。
- `__file__`はtrain sourceに0件。

### 親compactとの構造比較

- exp423 compact: 2,230行、11章。
- exp428 compact: 2,620行、12章。
- exp428はType Well axis graph、registration-map estimation、global/local/mapping-shape
  readoutを独立章として追加しており、親より薄いhelper呼び出し構成ではない。

## 実装時点の次のアクション（完了）

Kaggle CPU package / push / runを行う場合は別途ユーザー承認を得る。run前にaudit 1、
reporting folds 5、LightGBM config / trained fold / booster / PF / HMM / Beam / GPU /
parent replayがすべて0であることを再確認する。

## 2026-07-28 Kaggle CPU実行承認

ユーザーの追加依頼「実行してください」により、Kaggle CPU package / push / Stage 0 runを
承認済みとした。inference / submission、別実験への統合は承認範囲外のまま。

### push前の実行量再確認

- audit variant: `1`
- reporting folds: `5`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- PF / HMM / Beam well-run: `0 / 0 / 0`
- GPU run: `0`
- parent/control replay: `0`
- runtime: CPU、single process、internet off
- 親実験・controlの再学習: なし

### preflight

- Kaggle CLI: `2.2.3`
- credential: OAuth / legacy CLI credential PASS。API tokenは未設定だがCLI実行には不要。
- `make validate-template`: PASS
- `make validate-exp EXP=exp428_similar_well_gr_registration_map_transfer_readout`: PASS
- dedicated tests: `14 passed`
- train sourceの`__file__`: 0件
- canonical kernel:
  `kentookumura/exp428-gr-registration-map-transfer-readout-train`
- canonical title: `exp428 gr registration map transfer readout train`
- 初回pull: `403 Forbidden`。既存private kernelを確認できなかったため、新規canonical
  kernelとして同じslug/titleを使う。

### package / push

- package:
  `make prepare-kaggle-notebooks EXP=exp428_similar_well_gr_registration_map_transfer_readout EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp428-gr-registration-map-transfer-readout-train --title 'exp428 gr registration map transfer readout train' --run-on-push --strict"`
- metadata: private、CPU、GPU/TPU/internet off、run-on-push true
- bootstrap: 29 files、config含有、bytecode 0
- source/package config SHA一致:
  `9b097729a198e6bcd3d97163767ea695213d77fef22c853b69697a9f9c8edf54`
- push: Kaggle kernel version 1 successfully pushed
- kernel URL:
  `https://www.kaggle.com/code/kentookumura/exp428-gr-registration-map-transfer-readout-train`
- 状態: version 1 COMPLETE

### version 1 technical fail

- runtime: 約`174.6 sec`（audit summaryは約`164.6 sec`）
- target-free logical content SHA:
  `37d4f110fcc791c686b8c601bb23ebc55eebb295122e805b4cb46563d5a27531`
- query truth read: freeze前`0`行、freeze後`3,783,989`行
- fold / donor-query separation / truth freeze / Type Well axis graph:
  PASS
- identifiable query block fraction: `0.5826377295492488`、PASS
- supported query well fraction: `0.0`、FAIL
- supported prediction finite fraction: `NaN`、FAIL
- decision: `invalid_or_insufficient_registration_support`

実ファイル確認が必要なtechnical failureだったため、Kaggle outputを
`/tmp/kaggle-output/exp428-v1`へ取得した。donor registration mapは
`1,432`行生成されていた一方、donor rankingは空、全`773` query wellの
eligible donor countが0だった。

### version 1 root cause / version 2修正

`preprocess_suffix_gr`が、support判定用のraw GR欠損maskをnormalized DTW入力にも
適用していた。固定契約はfinite/support 70%以上を許容するが、constrained DTWは
NaN格子を通過できないため、内部欠損が1点でもあるquery-donor pairは経路を失う。
親exp423はsupport maskをgateに残しつつ、DTW入力には決定的な線形補間値を保持する。

固定済みのsupport threshold、DTW制約、候補、gateは変更せず、exp423と同じ前処理へ
戻す最小修正を採用した。内部欠損があってもsupport 70%以上なら補間DTWが有限となる
回帰testを追加する。version 1は科学的な支持不足としては扱わず、実装上のtechnical
failureとして記録し、同一canonical kernelをversion 2へ更新する。

version 2 push前検証:

- dedicated tests: `15 passed`
- `py_compile`: PASS
- Ruff `F821,E501`: PASS
- Jupytext round-trip: PASS
- `make validate-exp`: PASS
- `make validate-template`: PASS
- existing canonical kernel pull: PASS、`id_no=128932184`
- 実行量: audit `1`、reporting folds `5`、model/config/booster/PF/HMM/Beam/GPU/
  parent replayは引き続き全て`0`

## 2026-07-28 Kaggle CPU version 2 terminal result

- kernel:
  `kentookumura/exp428-gr-registration-map-transfer-readout-train`
- kernel id_no / version: `128932184 / 2`
- status / runtime: `COMPLETE / 約225.6 sec`
- scientific contract SHA:
  `f3f084e9769da5b24c7c18497c2de101e025e8b340bbec797a554d1bb8f2cdf9`
- target-free logical content SHA:
  `54127363c066b75180af274e8bb4e076536d97c94554ccec2846f4986ff849d7`
- query truth rows: freeze前`0`、freeze後`3,783,989`
- supported query wells: `306 / 773 = 0.3958602846054334`
- query-referenceまで評価可能: `290 wells`
- identifiable query block fraction: `0.5826377295492488`
- supported prediction finite fraction: `1.0`
- fold complete / donor-query separation / truth freeze / Type Well axis:
  全PASS

supported coverageの固定下限`0.70`を満たさないためtechnical FAIL。support FAILだけで
terminal decisionが確定するので、logical SHAをanchor化する独立rerunは行わない。
deterministic anchorはfalseのままとする。

supported-only参考値:

- primary MAE: `2.529310 ft`
- zero MAE: `1.105172 ft`、primary gain `-1.424138 ft`、nonworse `0 / 5 folds`
- stable-random MAE: `1.808621 ft`
- same-group median MAE: `1.398276 ft`
- top-5 oracle MAE: `1.118966 ft`、zero比gain `-0.013793 ft`
- DTW cost-error pooled Spearman: `0.075211`
- mean ZNCC gain vs zero: `-0.057438`
- local-vs-global block MAE gain: `-5.050144 ft`

technical / scientific / local-shape gateはすべてFAIL。terminal decisionは
`invalid_or_insufficient_registration_support`。同一OOFでsupport/group/DTW/shift/
block/primaryを救済せず、inference、submission、HMM/PF/Beam observation offset統合へ
進まない。Kaggle logsは`kaggle/output/train_v1`と`train_v2`へ保存した。
