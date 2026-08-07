# exp445_tvt_to_u_coordinate_parity_exact_hmm セッションノート

## 目的

exp209 exact HMMの固定TVT state indexを変えず、各rowの状態値を
`U_t,j=P_j+Z_t`へ再ラベルしたcandidateが、親と数値的に一致することを
確認するtechnical parity auditを設計する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU Stage 0 v2完了、`coordinate_parity_verified`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 比較参照: `exp438_u_state_fixed_lattice_exact_hmm`
- 初回設計時の実装承認: なし
- 実装承認: 2026-07-29の追加依頼「exp445を実装してください」
- 正規Notebook採用 / Kaggle package / push / run:
  2026-07-30の追加依頼で承認され完了
- full OOF / inference / submission: 設計対象外
- CV / LB: なし

## 2026-07-29 設計セッション

ユーザー依頼:

```text
確認のためTVTを単にUにする実験をしたいです。
バックログ、実験ディレクトリ、steeringを作成して設計を確定させてください。
実装はまだです。
```

実行コマンド:

```bash
make new-steering EXP=exp445_tvt_to_u_coordinate_parity_exact_hmm
make new-exp EXP=exp445_tvt_to_u_coordinate_parity_exact_hmm
```

設計上の確定事項:

- 親のTVT格子を`P_j`とし、candidate U状態値を
  `U_t,j=P_j+Z_t`とする。
- `U_t,j-Z_t=P_j`なのでemission supportとTVT readoutは親と同じ。
- moving U gridのphysical edgeは`(P_k-P_j)+delta_Z`であるため、
  candidateのindex-space meanも親と同じ
  `r_current*delta_MD-delta_Z`とする。
- exp438の`P_j+Z_last`固定absolute-U格子と、
  index mean `r_current*delta_MD`は使わない。
- technical parityだけを判定し、CV、RMSE改善、LB、candidate promotionは
  判定しない。
- parity PASSはexp438の再評価やexp209改善を意味しない。

## 変更点

- 親の確率モデルは変更せず、position stateの数値表現だけを
  `P_j -> P_j+Z_t`へ変える設計を追加した。
- exp438の固定absolute-U格子を明示的に禁止した。
- 評価をtruth-free technical parityへ限定し、性能評価と提出flowを外した。

## 予定実行量

実装・実行が別承認された場合の固定値:

- coordinate candidate: 1
- manifest wells: 32
- candidate HMM well-runs: 32
- paired parent HMM well-runs: 32
- total HMM well-runs: 64
- reporting folds: 0
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`
- full OOF / inference / submission: なし

保存済みparent predictionのload-only比較ではposterior、log-likelihood、
transition/emission parityを確認できないため、将来のtechnical auditでは
fixed32に限ってparentをpaired rerunする。現在は実行していない。

## 2026-07-29 compact self-contained実装セッション

作成:

- `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_train.py`
- `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_train.ipynb`
- `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_inference.py`
- `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_inference.ipynb`
- `tests/test_exp445_tvt_to_u_coordinate_parity_exact_hmm.py`

実装内容:

- raw observed seriesを固定した後、parent TVT経路とcandidate row-shifted U経路で
  emission、initial prior、transition meanを別々に組み立てる。
- parent position kernelは
  `offset*h-(r*dMD-dZ)`、candidateはmoving-U physical edge
  `offset*h+dZ-r*dMD`から独立に重みを計算する。
- exact forward/backwardはcoordinate modeごとに別実行し、fixed32では
  candidate 32 + paired parent 32 = 64 HMM well-runsを行う。
- suffix truth、fold、role、episode、errorを読む関数を実行経路に置かず、
  leakage ledgerの各read countを0でAND gateする。
- coordinate、emission、prior、transition、likelihood、position/rate posterior、
  TVT mean/std、`E[U]-Z`と、入力・posterior・prediction・ledger SHAを保存する。
- gzip生成物はdeterministic gzipとし、decompressed content SHAを
  readback gateの主証拠にする。
- synthetic variable-Z / constant-Zとtiny全path列挙referenceを実装した。
- inference guardはhidden-test predictionとsubmissionを常に例外停止する。

実装中の数値修正:

- 最初のtestで、約20,000 ftのabsolute Uから約8,000 ftのZをfloat64で引くと
  `U-Z=P`差が`1.8189894e-12 ft`となり、事前固定`1e-12 ft` gateを超えた。
- gateは緩めず、Uの座標表示とreadoutだけを`longdouble`で保持した。
  HMMのgrid、emission、transition probability、forward/backwardは親と同じ
  float64/float32契約を維持する。
- initial prior gateはlog-densityのtail差ではなく、正規化したprior probability
  のmax absolute differenceとして判定する。HMM本体には独立に組み立てた
  log priorを渡す。
- exp209直接比較testでは、最初の実装のposition posteriorが親と最大
  `4.0e-8`異なった。原因は最後にjoint probabilityをpositionへ周辺化する
  加算順であり、rateをposition内で加算後に全positionを正規化するexp209の
  順序へ合わせ、position posteriorとlog-likelihoodのarray/値完全一致を確認した。
- exp209自身のposition posteriorは浮動小数点加算によりrow sumが約`1e-7`
  ずれる場合がある。HMM posterior / log-likelihoodはbyte-parityのまま保持し、
  coordinate expectationではparent/candidateを明示正規化する。
  exp209互換のraw matrix-product mean/stdもreport-only列として併記し、
  normalization由来差を隠さない。

検証コマンド:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp445_tvt_to_u_coordinate_parity_exact_hmm/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp445_tvt_to_u_coordinate_parity_exact_hmm/\
exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp445_tvt_to_u_coordinate_parity_exact_hmm/\
exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp445_tvt_to_u_coordinate_parity_exact_hmm/*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp445_tvt_to_u_coordinate_parity_exact_hmm/*compact_selfcontained*.py \
  tests/test_exp445_tvt_to_u_coordinate_parity_exact_hmm.py --select F821
.venv/bin/pytest -q tests/test_exp445_tvt_to_u_coordinate_parity_exact_hmm.py
.venv/bin/pytest -q \
  tests/test_exp438_u_state_fixed_lattice_exact_hmm.py \
  tests/test_exp445_tvt_to_u_coordinate_parity_exact_hmm.py
make validate-exp EXP=exp445_tvt_to_u_coordinate_parity_exact_hmm
```

検証結果:

- 専用pytest: `17 passed`
- exp438 / exp445関連pytest: `29 passed`
- Jupytext `--test`: train / inferenceともPASS
- py_compile: PASS
- Ruff F821: PASS
- strict experiment validation: PASS
- project template / strict config validation: PASS
- `__file__` / `Path(__file__)` / `from settings import`: compact 2本とも0件
- 親compact比較:
  - exp438 train: 2,780行、9 numbered sections
  - exp445 train: 2,703行、9 numbered sections
  - exp445はinput、独立座標組立、physical kernel、exact HMM、
    brute-force、fixed32 freeze、gate、metricsをNotebook上で追える。
- 正規`*_train.ipynb` / `*_inference.ipynb`: 未変更
- Kaggle package / push / run: 未実施

全体test:

- `make test`は1,527件をcollectする途中、今回の変更対象外である既存
  exp297 / exp301 / exp333 / exp336 / exp349のconfig-contract不一致5件により
  collectionで停止した。
- exp445 testの失敗ではなく、専用17件と直接関連exp438込み29件は独立にPASSした。
- 今回のscope外である既存5実験のconfig/doc/source整合は変更していない。

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- seed policy: RNGなし、well / row / position / rate / edge / message順固定。
- stochastic components: なし。
- runtime予定: Kaggle private CPU、1 worker、Numba 1 thread、internet無効。
- input / coordinate ledger / transition-emission / posterior / prediction SHAを
  記録する。
- gzipはdecompressed content SHAを主証拠にする。
- model manifest / model SHA / submission SHA: 非該当。
- 初回runをdeterministic parity anchorとせず、独立rerunでinput /
  posterior / prediction SHA一致後だけanchorとみなす。

## 次のアクション

1. 正規Notebookへcompact候補を採用する場合は別承認を得る。
2. Kaggle package / push / runはさらに別承認とし、実行前に
   candidate 32 + paired parent 32 = 64 HMM runsを再確認する。
3. 初回parity PASSだけをdeterministic anchorとせず、独立rerun SHA一致を待つ。
4. parity PASS後もfull OOF、inference、submissionへは進まない。

## 2026-07-30 実行承認・pre-push確認

ユーザー追加依頼:

```text
実行してください
```

承認範囲:

- compact self-contained train / inference guardの正規Notebook採用
- Kaggle private CPU package / push
- fixed32 Stage 0の初回1回実行

承認範囲外:

- 独立rerun
- full OOF
- inference
- submission

push前に再確認した実行量:

- coordinate candidate: `1`
- manifest wells: `32`
- candidate HMM well-runs: `32`
- paired parent HMM well-runs: `32`
- total HMM well-runs: `64`
- reporting folds / LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

保存済みparent predictionではposterior / likelihood / transition parityを
検証できないため、fixed32に限りparent 32 well-runsをpaired rerunする。
exp438 Stage 0の32 HMM runsは約24分だったため、今回の64 runsは概ね
約48分を見込む。CPU 1 worker / Numba 1 thread、internet無効で実行する。

正規Notebook採用・package監査:

- canonical train SHA:
  `6a8e6ffa4a3d8b063b29f97f06fa61f7f24475a58b39b23e7fca833ceb0edc37`
- packaged notebook SHA:
  `3b6eb840e47b5f1d505d30166c81194b23e7f0d99d923e0f2a17352d2fa6cad3`
- packaged metadata SHA:
  `f7351c0a5abffa42696d438e5bd93f7c6e66a1954d45350b440c900422fae8d8`
- packaged / bootstrap config SHA:
  `57e67088f721d7b29401396aa797482ec3b7ae0064dfbb01e2892856d876c741`
- fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- metadata: private / CPU / internet無効 / `run_on_push=true`
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp209-joint-exact-parity-train`
- loose config、bootstrap config、source configはbyte一致

Kaggle push:

- kernel:
  `kentookumura/exp445-tvt-to-u-coordinate-parity-exact-hmm-train`
- version: `1`
- pushed at: `2026-07-29 21:07:57 UTC`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp445-tvt-to-u-coordinate-parity-exact-hmm-train`
- push直後status: `RUNNING`

### Kaggle version 1失敗

- final status: `ERROR`
- failure observed: `2026-07-29 21:08:32 UTC`
- HMM well-runs completed: `0`
- failure class: environment / Numba runtime initialization
- 最初の意味のあるtraceback:
  `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched`
- 原因:
  Numba import後の`run_stage0`内で`os.environ["NUMBA_NUM_THREADS"]="1"`を
  設定したため、Kaggle側で初期化済みのthread poolとNumba config reloadが
  衝突した。
- 修正:
  exp438でKaggle実績のある`set_num_threads(1)`だけを残し、実行後の環境変数
  書き換えを削除した。
- 科学contract、fixed32 scope、gate、実行量、CPU 1 worker /
  Numba active thread 1は変更していない。
- 同じcanonical kernel idへversion 2として再pushする。

Kaggle version 2 package / push:

- canonical train SHA:
  `7c732391d6a5d43ef02bbcc34e9c540e2de77b8245a390d65c0da83514dba60b`
- packaged notebook SHA:
  `4336ec9d0e7e4a74cd5b8ef8d9fa85adb8c9d9672ce095fe43ddf52b2fcf3baf`
- packaged metadata SHA:
  `f7351c0a5abffa42696d438e5bd93f7c6e66a1954d45350b440c900422fae8d8`
- packaged / bootstrap config SHA:
  `f1b8ba169ad78539aeac94d9aaf860e06947ee7d40653cb7000904ac61205c37`
- dedicated pytest: `17 passed`
- Jupytext / py_compile / Ruff F821 / strict experiment validation: PASS
- three-way config、manifest SHA、metadata / source / CPU / offline:
  全項目再監査PASS
- version: `2`
- pushed at: `2026-07-29 21:10:10 UTC`
- push直後status: `RUNNING`

## 2026-07-30 Kaggle Stage 0 version 2結果

- Kaggle status: `COMPLETE`
- kernel id_no: `129095337`
- completion observed: `2026-07-29 22:13:45 UTC`
- notebook-reported runtime: `1920.670088 sec`（約32分01秒）
- peak RSS: `1.189617 GiB`
- runtime: Python `3.12.13`、NumPy `2.0.2`、Pandas `2.3.3`、
  Numba `0.60.0`
- execution:
  candidate 32 + paired parent 32 = 64 HMM well-runs、
  reporting fold / LightGBM / model / booster / PF / Beam / GPU各0
- final status: `coordinate_parity_verified`

technical gate:

- 16/16 PASS
- real log-likelihood max abs: `0`
- smoothed position posterior max abs: `0`
- smoothed rate posterior max abs: `0`
- TVT mean/std max abs: `1.8189894035458565e-12 ft`
- candidate `E[U]-Z` max abs: `8.881784197001252e-16 ft`
- physical edge residual max abs: `1.1102230246251565e-16 ft`
- position kernel max abs: `2.220446049250313e-16`
- brute-force max abs: `2.9459638284379253e-08`
- finite coverage: `1.0`
- forbidden truth / fold / role / episode / error read: `0`
- manifest wells / paired HMM runs: `32 / 64`
- artifact readback SHA: PASS

artifact:

- local:
  `artifacts/kaggle_v2/`
- metrics SHA:
  `51786ac378eaf9366eeee62e3d9466c7584a0667fd9d32bc1d9fc7d690a3d783`
- Stage 0 report SHA:
  `acfa67077273adb911e6f9bf56da0479d804ca30dacb9534dd524690a4b695e1`
- prediction raw / decompressed SHA:
  `28ef89ea955f644ebda14e92d05acc8c72c63c6288d3ce0f0fd840b06567f889` /
  `88c642571023dc2560ad57a59580df74977e4fe021fb0c5faa41455acc7a240c`
- posterior ledger raw / decompressed SHA:
  `45996a7582dcf58c6caf30c1ecaedbb11a5489c503a64977ea9aeaa7fc2e0e24` /
  `a3fdf9ceae971f1df47dda862a344cd000977d4c4a472d3605633bca89c00898`
- transition/emission ledger raw / decompressed SHA:
  `3c7faf8e1b3843074bb9209a85057537fb18116359d88b12c3c5e6b15d247e7e` /
  `8c698a7d501cc94d9fca76680fce9ea0a1f35db7d84924a9ddcdb21b13872824`
- row counts:
  prediction `156,088`、posterior ledger `32`、
  transition/emission ledger `32`

結論:

- exp209の固定TVT格子と`U_t,j=P_j+Z_t`は、離散exact HMM上でも同じ
  posterior、likelihood、TVT readoutを与える。
- PASSは座標再ラベルの正しさだけを示し、予測改善やexp438の
  fixed absolute-U仮説の救済を意味しない。
- 初回成功runだけなのでdeterministic anchorとはしない。
- `execution.run_hmm=false`、`runtime.run_approved=false`へ再ロックした。
- 独立rerun、full OOF、inference、submissionは実施しない。
- 完了後検証:
  専用pytest `17 passed`、exp438/exp445関連 `29 passed`、Jupytext、
  py_compile、Ruff F821、strict experiment validation、project template /
  strict config validationはすべてPASS。
