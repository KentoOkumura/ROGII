# exp458_acceleration_state_exact_runtime_engine_audit セッションノート

## 目的

exp444の科学仕様を固定したまま、scaled probability-space因子化/fused計算、
exact-bit `delta_MD` cache、4-well外側並列でruntime上限へ入るかを、
別実験のexact-equivalence/runtime auditとして検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0A親数値parity FAIL・terminal close
- 親/構造参照: `exp444_acceleration_state_exact_hmm`
- CV: まだなし
- LB: まだなし
- 実装: 2026-07-30のユーザー依頼で承認・完了
- 正規train Notebook採用 / package / Stage 0A run: 完了
- inference / submission: 未承認

## コマンドログ

### 2026-07-30 Kaggle version 2完了・Stage 0A FAIL closed

- canonical kernel:
  `kentookumura/exp458-accel-state-exact-runtime-eng-audit-train`
- version / id_no: `2` / `129168013`
- Kaggle status: `COMPLETE`
- execution contract: 1 scientific variant、1 runtime engine、2 repeats、
  fixed4、合計8 candidate HMM well-runs。parent/control/model/booster/
  PF/Beam/GPU各0。
- runtime:
  - repeat 1: `72.250458299 sec`
  - repeat 2: `72.755703128 sec`
  - 遅いrepeatのexp444比: `10.258353118x`
  - fixed32投影: `582.045625024 sec`
  - full 773-well投影: `14,114.606406832 sec`
  - peak RSS: `13.033187866 GiB`
  - effective outer workers: 4、worker内Numba/BLAS threads: 1
- runtime、fixed32/full投影、RSS、worker/threadは全PASS。
- numerical:
  - parent prediction mean最大差:
    `1.0413506242912263e-4 ft > 1e-5`、FAIL
  - parent prediction std最大差:
    `6.35657412058066e-5 ft > 1e-5`、FAIL
  - parent acceleration posterior最大差:
    `8.97726402104837e-6 > 1e-7`、FAIL
  - parent rate diagnostic最大差:
    `1.0862973659972464e-6 <= 5e-6`、PASS
  - small dense prediction / acceleration posterior:
    `3.637978807091713e-12 ft / 3.3306690738754696e-16`、PASS
  - normalization: `4.085620730620576e-14`、finite coverage: `1.0`
- repeat 1/2のprediction/posterior/diagnostic SHAは完全一致。
- truth/role/fold/episode/cause read before freezeは0。
- `all_pass=false`、`stage0b_eligible=false`、
  最終判定`stage0a_fail_closed`。
- output取得先:
  `kaggle/output/train_v2/`。metrics、kernel log、6 gzip生成物、
  runtime manifest、summaryを取得した。
- raw/decompressed SHA、runtime manifest
  `6faad84d...bf869`、summary `831bc160...e3c7`を実ファイルで照合した。
- runtime高速化だけは成立したが、固定したexact-equivalence数値gateを
  満たさない。favorable rerun、gate緩和、worker/thread/cache/precision/
  state/parameter変更による救済をせず、exp458を閉じる。

### 2026-07-30 Stage 0A実行承認・push前契約

- ユーザーの「実行してください」により、正規train Notebook採用、Kaggle
  package、private CPU Stage 0A runを承認済みとした。
- 実行stage: `stage0a_fixed4_runtime_equivalence`
- scientific variant: 1
- runtime engine config: 1
- repeat: 2
- fold: 0
- candidate HMM: 4 wells/repeat、合計8 well-runs
- parent/control HMM rerun: 0
- LightGBM config / trained fold / booster / fitted model: 0 / 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0
- 保存exp444 kernel version 1の3生成物をload-onlyで参照し、再生成しない。
- Stage 0B/1、inference、submissionは未承認のまま維持する。

### 2026-07-30 初回push 400とcanonical slug短縮

- 実験名全体から生成した58文字の
  `exp458-acceleration-state-exact-runtime-engine-audit-train`は、
  id/titleのslugが一致していたがKaggle `SaveKernel 400 Bad Request`で
  実行開始前に拒否された。
- 直前の同slug metadata pullは403で、既存kernel/versionは確認されなかった。
- repo内で反復確認済みのKaggle 50文字slug上限と一致するため、科学contract、
  notebook、入力、実行量を変えず、`acceleration`を`accel`、`engine`を`eng`
  へ短縮した48文字のcanonical id/title
  `kentookumura/exp458-accel-state-exact-runtime-eng-audit-train` /
  `exp458 accel state exact runtime eng audit train`へ揃えて再packageする。
- 拒否された長slugへは再pushせず、短縮canonical slugをversion 1として扱う。

### 2026-07-30 Kaggle version 1 technical error

- canonical kernel:
  `kentookumura/exp458-accel-state-exact-runtime-eng-audit-train`
- version / id_no: `1` / `129168013`
- pull-back metadata: private、CPU、GPU/TPU/internet off、
  exp444 kernel source一致。
- candidate計算は2 repeatsとも完走した。
  - repeat 1: `71.139464264 sec`、peak RSS `13.004089355 GiB`
  - repeat 2: `72.194301871 sec`、peak RSS `13.033439636 GiB`
  - effective outer worker PID: `[55, 56, 57, 58]`
  - prediction decompressed SHA:
    `99c3fc141b39188e21dc1b8fef1c55998ef3878a6a0b0a7922f82ddeedf1aefb`
  - posterior bundle SHA:
    `4299cbbdfb2aa9b54e4d372d4abba7479ecafaf06628bb5ec5ee22c42c4dffe9`
  - diagnostic SHA:
    `0220cd1408695e77c6d60d64e7849bb86a4e5037b8af4db8c81c47fd4c83457d`
  - 3種SHAはいずれもrepeat間で一致した。
- 計算完了後のsummary JSON保存で、NumPy `bool_`が`to_jsonable`から
  Python `bool`へ変換されず`TypeError`となった。科学計算、数値gate、
  runtime gate、入力、worker/thread契約の失敗ではない。
- 修正範囲をserializerの`np.bool_ -> bool`変換とregression testだけに限定し、
  同じcanonical slugのversion 2としてtechnical retryする。
- version 1はKaggle status `ERROR`であり、Stage 0A最終判定や
  favorable runtime選択には使わない。version 2の遅いrepeatだけで判定する。

### 2026-07-30 実装

```text
.venv/bin/pytest -q experiments/exp458_acceleration_state_exact_runtime_engine_audit/tests/test_exp458_acceleration_state_exact_runtime_engine_audit.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact train/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact train/inference.py>
.venv/bin/python -m py_compile <compact train/inference.py>
.venv/bin/ruff check <compact train/inference.py> <test.py> --select F821
make validate-exp EXP=exp458_acceleration_state_exact_runtime_engine_audit
```

- `task` commandは環境に存在しなかったため、同等の`make validate-exp`を使用した。
- 専用testは`9 passed in 0.71s`。
- Jupytext train/inference round-trip、構文、F821、strict experiment validationを
  すべてPASSした。
- local notebook実行、Kaggle package、Kaggle push/runは行っていない。
- 親compact trainは10章・2,537行、exp458 compact trainも対応する10章・
  3,090行で、入力準備、kernel、forward/backward、dense reference、
  freeze、gate、orchestrationの役割をすべて維持した。exp458の増分は
  saved-parent load/parity、2-repeat outer4、runtime/RSS/SHA manifestである。
- 全suite `make test`は今回の変更外にあるexp297/301/333/336/349の
  config path解決でcollection errorとなった。5件を除外した確認では
  exp458を含む`1,503 passed / 7 skipped`、既存の別実験で
  `15 failed / 94 errors`だった。exp458専用9件は両方の実行でPASSした。

### 2026-07-30 design-only作成

```text
make new-steering EXP=exp451_acceleration_state_exact_runtime_engine_audit
make new-exp EXP=exp451_acceleration_state_exact_runtime_engine_audit
```

- 作成開始時点ではexp450が使用済みだったためexp451を選んだ。
- 並行するworkspace更新で別系統のexp451、続いてexp456まで追加されたため、
  内容を変えず最終的に空いていたexp458へexperiment/steering/notebook名を改番した。
- 最終確認時点でexp458は本実験だけであり、旧runtime-audit exp451/exp456 pathは残していない。

## 変更点

- exp444 scientific contract SHA
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`
  を完全固定した。
- runtime engine候補を1つに固定した:
  scaled float64 probability-space、factorized/fused transition、
  exact-bit `delta_MD` OU cache、outer workers 4。
- pruning、state/support/parameter変更、float precision低下、GPUを禁止した。
- Stage 0Aは保存exp444 fixed4 load-only、candidate fixed4 2 repeatsとした。
- Stage 0A実行量はscientific variant 1、runtime engine 1、
  candidate HMM well-runs 8、control rerun/model/booster/PF/Beam/GPU各0。
- 2 repeatsの遅い方で`>=4.75x`、fixed32/full`<=3,600/30,600 sec`、
  RSS`<=25 GB`を判定する。
- Stage 0B/1はexp444と同じscientific gateを継承し、各stageは別承認とした。
- float64 scaled probability-space forward/backwardを実装し、各rowのscaleと
  emission offsetを保存してreverse factorized operatorに再利用する。
- acceleration 3x3、destination-acceleration別OU 41x41、position 5-offsetを
  dense joint transitionなしで適用する。
- exact float64 bit patternだけをkeyにするwell-local `delta_MD` cacheを実装した。
  `nextafter(10.0,+inf)`は別key、同一10.0は同じkeyになるtestを固定した。
- Linux `fork` process 4本、worker内Numba/OMP/MKL/OpenBLAS/VecLib/NumExpr
  thread各1、completion order破棄、`well,row_idx` stable sortを実装した。
- 保存exp444 3生成物はcandidate 2 repeatsの全well freeze後だけ読み、
  decompressed SHA検証後に数値parityへ使用する。
- repeatごとのprediction decompressed SHA、posterior bundle SHA、diagnostic SHA、
  process-tree RSS、worker PID/thread、input/prepared/cache/kernel/scale SHAを記録する。

## 実装時の数値確認

- small dense prediction最大絶対誤差:
  `1.8189894035458565e-12 ft`
- small dense acceleration posterior最大絶対誤差:
  `3.3306690738754696e-16`
- synthetic入力でexp444 log-space engineに対し、prediction/std `1e-5 ft`、
  acceleration posterior `1e-7`、rate diagnostic `5e-6`の全許容差をPASSした。
- exact-bit OU cache展開とdirect kernel、position precomputeとparent kernelは
  最大誤差`0.0`でPASSした。

## 再現性メモ

- seed policy: RNGなし。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle private CPU version 2、outer 4、
  worker内Numba/BLAS 1、GPUなし。
- output order: `well_id,row_idx` stable sort。
- parent input: exp444 Kaggle kernel version 1、保存fixed4 3生成物をload-only参照。
- parent prediction decompressed SHA:
  `4927083191857ebf03dfd3ec755d2852afeb6125b4190e86796bc67552a2cfb1`
- parent acceleration posterior decompressed SHA:
  `8d9f3b657e0904d79af7bfc07fa4b08ac71fbbfaed28338cf38ee2526a6498e3`
- parent diagnostic decompressed SHA:
  `b538e024c4f904fc210314929266b7be7b8cb73d375cbb01a2e4d30580d519d7`
- candidate source contract SHA:
  `acba22633c2985c78152b19e3147253b4bd7a85b44bd33cbf2ef09f9ff2df84b`。
- candidate prediction/posterior/diagnostic decompressed SHA:
  `99c3fc14...1aefb` / `4299cbbd...ffe9` / `0220cd14...457d`。
- runtime manifest SHA: `6faad84d...bf869`。
- model / submission SHA: artifactを生成しないため非該当。
- deterministic anchor: false。Stage 0A内repeat SHAは一致したが、独立した
  成功runは1回だけでありanchor扱いしない。

## 次のアクション

1. exp458はterminal closeとし、Stage 0B/1、inference、submissionへ進まない。
2. 原因検証が必要なら、保存済みexp458 v2/exp444を使うtarget-free
   long-trellis first-divergence studyを別管理する。新規HMM runは行わない。

## 禁止事項

- exp444のterminal FAILを再分類しない。
- exp444 control HMMを再実行しない。
- state数、span、transition、prior、emission、grid、readout、runtime/RSS gateを変えない。
- pruning、approximation、precision低下、GPU、favorable rerun selectionで救済しない。
- Stage 0A PASSだけでmechanism、CV、inference、submissionへ昇格しない。
