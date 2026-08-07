# exp178_supervised_gr_window_matcher_from_known_tvt_prefix セッションノート

## 目的

`supervised_gr_window_matcher_from_known_tvt_prefix` backlog を実装する。既知 `TVT_input` prefix の真の alignment から GR window pair を作り、real GR が shuffled/no-GR control よりも正しい候補 window を識別できるかを 1 fold / row cap smoke で確認する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_smoke_supported
- CV: pair AUC 0.765413549
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix
make new-exp EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_inference.py
.venv/bin/python -m py_compile experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.py experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_inference.py
.venv/bin/ruff check experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.py experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_inference.py
make validate-exp EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix
make prepare-kaggle-notebooks EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp178-supervised-gr-window-matcher-from-known-tvt-prefix-train --title 'exp178 supervised gr window matcher from known tvt prefix train' --run-on-push --strict"
kaggle kernels pull kentookumura/exp178-supervised-gr-window-matcher-from-known-tvt-prefix-train -p /tmp/kaggle-pull/exp178-supervised-gr-window-matcher-from-known-tvt-prefix-train -m
make prepare-kaggle-notebooks EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp178-supervised-gr-window-matcher-train --title 'exp178 supervised gr window matcher train' --run-on-push --strict"
make push-kaggle-train EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix
kaggle kernels pull kentookumura/exp178-supervised-gr-window-matcher-train -p /tmp/kaggle-pull/exp178-supervised-gr-window-matcher-train -m
kaggle kernels logs kentookumura/exp178-supervised-gr-window-matcher-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp178-supervised-gr-window-matcher-train
kaggle kernels output kentookumura/exp178-supervised-gr-window-matcher-train -p experiments/exp178_supervised_gr_window_matcher_from_known_tvt_prefix/kaggle/output/train_v1
```

### 失敗と復旧

- 初回 push は `kentookumura/exp178-supervised-gr-window-matcher-from-known-tvt-prefix-train` で `SaveKernel 400`。`pull` は 403 で kernel 作成済みとは扱わなかった。
- slug が 63 文字で長かったため、同じ exp のまま `kentookumura/exp178-supervised-gr-window-matcher-train` / `exp178 supervised gr window matcher train` に短縮して再 prepare。
- 短縮 canonical kernel は version 1 として push 成功。

### 結果ログ要約

- Kernel: `kentookumura/exp178-supervised-gr-window-matcher-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp178-supervised-gr-window-matcher-train
- Kaggle runtime: CPU, GPU false, internet false
- runtime_seconds: 38.580180168151855
- pair rows: 102,400
- anchors: 10,240
- wells: 160
- split: train 128 wells / valid 32 wells
- status: `completed_train_side_smoke_supported`
- real GR logistic AUC: 0.7654135492112901
- shuffled GR logistic AUC: 0.6623459392123752
- AUC margin vs shuffled: +0.10306760999891496
- real GR logistic top1 within10: 0.35595703125
- no-GR logistic top1 within10: 0.2529296875
- real GR expected-error AUC: 0.827294
- real GR expected-error top1 within10: 0.513672
- real GR expected-error top5 within10 coverage: 0.959961

## 変更点

- `exp178_supervised_gr_window_matcher_from_known_tvt_prefix_train.py` を Jupytext percent 形式で追加。
- raw train の known-prefix row から supervised GR window pair dataset を作る。
- logistic controls は real GR、shuffled GR、no-GR の 3 系統。real GR には expected-error regressor も追加。
- metrics は pair AUC/logloss、topK coverage、by-well top1 rate、feature coefficient、SHA を保存する。

## GPU / 学習コストメモ

- active variant 数: 1 smoke
- LightGBM config 数: 0
- fold 数: 1 selected fold
- 合計 booster 数: 0
- sklearn classifier/regressor: logistic 3本 + HistGradientBoostingRegressor 1本
- control 再学習: なし
- Kaggle GPU: 不使用

## 再現性メモ

- seed policy: sorted well + evenly spaced prefix row selection、shuffled GR は well id keyed SHA256 roll。
- stochastic components: sklearn logistic solver、HistGradientBoostingRegressor。`random_state=42`。
- CPU/GPU runtime: CPU only。GPU disabled。
- Kaggle kernel id / version: `kentookumura/exp178-supervised-gr-window-matcher-train` v1。
- input / feature schema SHA: `kaggle/output/train_v1/artifacts/exp178_supervised_gr_window_matcher_from_known_tvt_prefix_summary.json` に記録。
- feature content SHA: gzip raw/decompressed SHA を summary JSON に記録。
- model manifest / model SHA: 保存対象外。係数表は CSV として SHA 記録。
- prediction SHA: validation pair prediction gzip の raw/decompressed SHA を summary JSON に記録。
- submission SHA: なし。submission は作らない。
- rerun check: 未実行。
- Kaggle package metadata: `kentookumura/exp178-supervised-gr-window-matcher-train`、GPU false、internet false、competition source `rogii-wellbore-geology-prediction` を確認済み。

## 次のアクション

1. direct replacement ではなく、learned GR matcher を PF/Beam candidate confidence feature または exp148/exp092 add-only feature として評価する follow-up を切る。
2. current test では observed `TVT_input` prefix のみを使い、評価 NaN 区間を label/center にしない parity 設計を先に固める。
