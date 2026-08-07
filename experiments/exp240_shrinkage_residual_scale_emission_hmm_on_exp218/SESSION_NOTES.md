# exp240_shrinkage_residual_scale_emission_hmm_on_exp218 セッションノート

## 目的

`shrinkage_residual_scale_emission_hmm_on_exp218` backlog を実装する。exp218 OOF center の
scalar sigma 対照を先に作り、事前固定した variance-shrinkage alpha を段階的に比較する。

## 現在の状態

- Route: `ensemble`
- 状態: closed（追加実行・推論・提出なし）
- selected stage: `shrinkage_alpha050`
- CV / LB: alpha 0.50 RMSE `8.336863897` / 未提出
- inference / submit: disabled

## 実装内容

- scalar control: exp218 `lgb_mean` OOF center、sigma `20`、lambda `0.50`。
- shrinkage: `sqrt((1-alpha)*20^2 + alpha*sigma_cf^2)`。
- predeclared alpha: `0.25 / 0.50`。1 Kaggle version につき 1 候補。
- deferred `sigma_cf`: exp234 と同じ well GroupKFold 5-fold HGB。held-out well overlap 0。
- exact HMM dynamics / direct comparison は exp234 と同じ。overall、distance、hidden-like、by-well、step-delta を保存する。
- scalar 未完了での shrinkage、複数 stage enable、未登録 alpha を fail-fast する。

## 親実験との差分

- exp234 の alpha `1.0` row-wise sigma を既存結果として参照し、再実行しない。
- exp240 は same-center scalar `sigma=20` を必須対照として新規生成する。
- shrinkage は linear sigma blend ではなく Gaussian variance blend とし、alpha を `0.25 / 0.50` に固定する。
- exp234 の comparison-only aggregate notebook は継承せず、各 stage の HMM と比較を同じ run で完結させる。

## コストガード

- 現active stage: alpha 0.25完了、追加実行なし
- HMM variant: 1
- residual-scale fit: alpha 0.25 v3で5完了
- LightGBM config / booster: 0 / 0
- fold: model training なし
- parent/control retraining: なし
- GPU: なし

Deferred shrinkage stage は active alpha 1、residual-scale GroupKFold fit 5、HMM 1、booster 0。
scalar control の結果を記録するまで enable / push しない。

## 再現性

- exact HMM RNG なし、outer worker 1、Numba thread 1。
- deferred HGB は GroupKFold shuffle なし、`random_state=42`、逐次 fit。
- gzip は decompressed content SHA を主証拠にする。
- exp218 OOF、row context、scale sidecar、HMM cache、comparison summary の SHA を記録する。
- raw-test regeneration / model manifest / submission SHA は対象外。
- `docs/06_reproducibility.md` を 2026-07-13 に確認済み。

## 実装コマンド

```bash
make new-steering EXP=exp240_shrinkage_residual_scale_emission_hmm_on_exp218
make new-exp EXP=exp240_shrinkage_residual_scale_emission_hmm_on_exp218 SOURCE=experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218
```

## 2026-07-13 実装検証

```bash
.venv/bin/python -m py_compile experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218/*.py
.venv/bin/ruff check experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218 --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218/exp240_shrinkage_residual_scale_emission_hmm_on_exp218_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218/exp240_shrinkage_residual_scale_emission_hmm_on_exp218_inference.py
make validate-exp EXP=exp240_shrinkage_residual_scale_emission_hmm_on_exp218
```

- compile、F821、Jupytext convert/test、strict experiment validation は pass。
- 親 train notebook 171行 / 6章に対し exp240 train は159行 / 6章。入力、stage、cost、実行、生成物をセル上で追える。
- notebook source に `__file__` はない。
- canonical train package: `kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train`。
- metadata/bootstrap: CPU、internet disabled、selected `scalar_control`、HMM 1、booster 0、必要 kernel sourcesを確認。
- inference package も生成したがno-output contractであり、pushしない。

## 2026-07-13 Kaggle scalar-control v1

```bash
kaggle kernels pull kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train -p /tmp/kaggle-pull/exp240-shrinkage-residual-scale-hmm-exp218-train -m
kaggle kernels push -p experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218/kaggle/train
```

- push前pull: 403。canonical kernelは未作成と判断。
- push: version 1 successful。
- kernel: `kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train`
- URL: https://www.kaggle.com/code/kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train
- runtime contract: CPU、internet disabled、scalar control 1、HMM 1、scale fit 0、LightGBM config/booster 0、parent/control再学習なし。
- v1 final status: `ERROR`。
- 最初の意味のある失敗: 1 well目の初回Numba JITで `Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched (currently have 1, trying to set 4)`。
- 原因: notebookがexact-HMM helper（Numba import）後に`NUMBA_NUM_THREADS=1`を設定しており、Kaggle runtimeのthread初期値と再読込が衝突した。
- 影響: scalar source / stage / cost / input previewはpass。HMMは1 well目のJIT前で停止し、予測・比較生成物はない。
- 修正: notebook最初のimports cellで、exact-HMM helper import前に`os.environ["NUMBA_NUM_THREADS"]="1"`を固定し、後段のlate assignmentを削除した。
- v2: 同じcanonical kernelへpush successful。v1の即時JIT errorは再発しなかった。

## 2026-07-13 Kaggle scalar-control v2 完了

- final status: `COMPLETE`。
- rows / wells: `3,783,989 / 773`、runtime `30,618.584 sec`（約8.50時間）。
- scalar HMM RMSE: `8.361307776`。
- exp218 point OOF比: `-0.114496982`、exp234 row-wise sigma HMM比: `-0.065923625`。
- distance bucketは6個中5個でexp218を改善し、`500_1000`のみ`+0.022970`悪化。
- hidden-like spatial / typewell-purged RMSEはそれぞれ`-0.111178 / -0.116946`改善したが、within10は`-0.004834 / -0.003853`悪化。
- by-wellは501改善 / 272悪化、最大悪化は`2e63d9de`の`+4.940864` RMSE。
- step deltaの`>5 / >10 / >25` rateはすべて0。
- HMM stdと誤差は単調でなく、row-wise uncertaintyをそのまま強く使う根拠は得られなかった。
- 実ファイル確認が必要なoverall / distance / hidden-like / by-well / step-delta / calibrationのCSV・JSONだけを取得し、大容量HMM cache archiveは取得しなかった。
- comparison artifactとdecompressed HMM featureのSHAは`metrics.json`へ記録した。
- 判定: scalar controlを支持するがguardは全面通過ではない。inference / submitは引き続き無効。

## 次

1. ユーザー承認後に限り、alpha `0.25`を同じexp240の次versionで単独実行する。
2. 実行契約はactive alpha 1、residual-scale GroupKFold fit 5、HMM 1、LightGBM booster 0、control再学習なし。
3. alpha `0.50`は0.25がsame-center scalar対照を上回った場合だけ検討する。

## 2026-07-13 alpha 0.25 実行承認

- ユーザーがalpha `0.25`の単独ablation実行を明示承認。
- active variant / alpha: `variance_shrinkage_alpha025` 1本 / `0.25` 1個。
- residual-scale config / fold / fit: 1 / well GroupKFold 5 / 5 fits。
- exact HMM variant: 1。
- LightGBM config / fold / booster: `0 / 0 / 0`。
- parent/control再学習: なし。保存済みscalar v2 RMSE `8.361307776`を対照に使う。
- CPU / internet disabled / inference disabled / submissionなし。

## 2026-07-14 alpha 0.50 Kaggle v4

- pre-push pullで同じcanonical kernelを確認し、version 4 push successful。
- package contract: selected `shrinkage_alpha050`のみ、scalar/alpha025 disabled。
- runtime contract: CPU、internet disabled、scale fit 5、HMM 1、booster 0、control再学習なし。
- initial status: `RUNNING`。
- URL: https://www.kaggle.com/code/kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train

## 2026-07-14 alpha 0.50 Kaggle v4 完了

- final status `COMPLETE`、runtime `28,718.496 sec`、rows / wells `3,783,989 / 773`。
- RMSE `8.336863897`。alpha 0.25比`-0.014258376`、scalar比`-0.024443880`、exp218比`-0.138940861`。
- alpha 0.25比でdistance bucket 4 / 6改善、hidden-like 2群悪化、352 wells改善 / 421悪化。
- MAE `+0.041423585`、within10 `-0.001093555`。最大well悪化`6a8fa194` `+2.898434`。
- scale guard pass、全fold well overlap 0、step delta `>5/10/25` rate 0。
- 比較CSV/JSONだけ取得し、大容量prediction / HMM gzipは取得しなかった。
- 判定: alpha 0.50は有限grid RMSE最良だがsecondary guard mixed。追加grid、inference、submitなしで終了。

## 2026-07-14 方向性close

- ユーザーがexp240の方向性を閉じる判断を明示。
- config上のscalar / alpha 0.25 / alpha 0.50をすべてdisabledとし、再push時はstage contractで停止する。
- canonical train packageも全stage disabledのclosed configで再生成した。Kaggleへの追加pushは行っていない。
- 禁止: 同一仮説のalpha grid拡張、再実行、raw-test inference、submission。
- train-side最良alpha 0.50の結果と再現性SHAは履歴として保持する。
- alpha `0.50`はdisabledのまま。alpha 0.25がscalar対照を上回るまで実行しない。

## 2026-07-13 alpha 0.25 Kaggle v3

```bash
kaggle kernels pull kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train -p /tmp/kaggle-pull/exp240-alpha025-prepush -m
make prepare-kaggle-notebooks EXP=exp240_shrinkage_residual_scale_emission_hmm_on_exp218 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train --title 'exp240 shrinkage residual scale hmm exp218 train' --run-on-push --strict"
kaggle kernels push -p experiments/exp240_shrinkage_residual_scale_emission_hmm_on_exp218/kaggle/train
```

- pre-push pullでcanonical kernel `id_no=126893705`を確認。
- package contract: selected `shrinkage_alpha025`、scalar/alpha050 disabled、CPU、internet disabled。
- push: version 3 successful。
- initial status: `RUNNING`。
- URL: https://www.kaggle.com/code/kentookumura/exp240-shrinkage-residual-scale-hmm-exp218-train
- 完了まではlogs/statusを確認し、scalar v2 `8.361307776`とのsame-center比較を記録する。

## 2026-07-14 alpha 0.25 Kaggle v3 完了

- final status: `COMPLETE`、runtime `29,341.119 sec`（約8.15時間）。
- rows / wells: `3,783,989 / 773`、active alpha 1、scale fit 5、HMM 1、booster 0。
- fold-safe scale guard pass。5 foldsすべてwell overlap 0、sigma-error Spearman `0.326486`、top/bottom RMSE比`3.578534`。
- alpha 0.25 RMSE `8.351122273`、scalar v2比`-0.010185503`、exp218比`-0.124682485`、exp234 alpha 1.0相当比`-0.076109129`。
- scalar比で全6 distance bucketは改善したが、MAE `+0.027283850`、within10 `-0.001543345`。
- hidden-likeはspatial `+0.005420`、typewell-purged `+0.004342` RMSEと両方悪化。
- by-wellはscalar比352改善 / 421悪化、median `+0.005307`。最大悪化`b3388334` `+1.216115`、最大改善`2e63d9de` `-4.456674`。
- 比較用CSV/JSONだけを`/tmp/kaggle-output/exp240-alpha025-v3`へ取得し、大容量HMM / residual-scale prediction gzipは取得しなかった。
- 判定: primary RMSEは事前条件を満たすがsecondary guardは混在。inference / submitなし。
- alpha 0.50は事前条件上検討可能だが、同規模CPU実行のためユーザー判断待ち。

## 2026-07-14 alpha 0.50 実行承認

- ユーザーがalpha `0.50`の追加単独ablationを明示承認。
- active variant / alpha: `variance_shrinkage_alpha050` 1本 / `0.50` 1個。
- residual-scale config / fold / fit: 1 / well GroupKFold 5 / 5 fits。
- exact HMM variant 1、LightGBM config / fold / booster `0 / 0 / 0`。
- parent/control再学習なし。保存済みscalar v2とalpha 0.25 v3を対照にする。
- CPU / internet disabled / inference disabled / submissionなし。
