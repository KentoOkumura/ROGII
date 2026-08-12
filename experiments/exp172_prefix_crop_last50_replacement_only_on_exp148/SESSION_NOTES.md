# exp172_prefix_crop_last50_replacement_only_on_exp148 セッションノート

## 目的

`prefix_crop_last50_replacement_only_on_exp148` backlog を実装する。exp161 の last50 add-only と exp166 の tail500/tail1000 replacement-only が exp148 を改善しなかったため、残っている案1として last50 に限定した replacement-only を isolated test する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle CPU split train 完了 / train-side rejected / no submit
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- Runtime: CPU (`cpu_deterministic_threads8`, `runtime.kaggle.enable_gpu=false`)
- 実行構成: 2段階。prefix crop feature cache notebook を実行後、split train notebook `lgb0` / `lgb1` / `lgb2` が cache を読み込んで学習する。

## 実装メモ

- exp166 の CPU 2段階 cache/train 構成をベースに exp172 へ分離した。
- `model.prefix_crop_window_features.windows` は `last50` のみ。
- 有効 variant:
  - `prefix_crop_last50_multiobs_replacement`
- replacement 対象:
  - exp145 learned likelihood multiobs 系: `ll_multiobs_score_*`, `ll_multiobs_mae_*`, `ll_multiobs_ncc_*`
- exp072/exp092 full-prefix 系 (`sc8_d`, `cal_a`, `pfx_rmse`, `slp_all`, `ktvt_range` など) は落とさず維持する。
- `learned_likelihood_confidence_no_multiobs` と `prefix_crop_last50_multiobs` group を実行時に構成し、variant の feature list へ渡す。
- 学習 notebook は `require_prefix_crop_cache=True` で実行し、cache が見つからなければ失敗する。
- exp166 の memory fix を継承し、train 本体は cache schema を先に読み、variant に必要な last50 crop 列だけを `usecols` で読み込む。

## Kaggle train push 前ガード

- active variants: 1
  - `prefix_crop_last50_multiobs_replacement`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`cpu_deterministic_threads8`)
- 合計 booster: 15
- split train notebook ごとの booster: 5
- control 再学習: なし

## 実装確認

- 2026-07-03: `docs/legacy/steering/20260703-exp172-prefix-crop-last50-replacement-only-on-exp148/` を作成。
- 2026-07-03: `experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/` を exp166 からコピーし、last50 multiobs replacement-only 用に設定を変更。
- 2026-07-03: `.venv/bin/python -m py_compile experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/*.py` は PASS。
- 2026-07-03: `.venv/bin/ruff check experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/*.py --select F821` は PASS。
- 2026-07-03: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...` は prefix crop features / train / train_lgb0 / train_lgb1 / train_lgb2 / inference で PASS。
- 2026-07-03: 同 `.py` から `.ipynb` を再生成し、コピー元 exp166 の古い import / header を除去。
- 2026-07-03: `make validate-exp EXP=exp172_prefix_crop_last50_replacement_only_on_exp148` は PASS。
- 2026-07-03: feature cache package を `kentookumura/exp172-prefix-crop-last50-exp148-features` / title `exp172 prefix crop last50 exp148 features` で prepare。metadata は CPU、internet off、run_on_push true、kernel sources は exp072/exp145。
- 2026-07-03: split train package を以下の CPU / internet off / run_on_push kernel として prepare。いずれも kernel source に `kentookumura/exp172-prefix-crop-last50-exp148-features` を含む。
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb0`
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb1`
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb2`
- 2026-07-03: feature cache notebook を push。
  - command: `kaggle kernels push -p experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/kaggle/prefix_crop_features`
  - kernel: `kentookumura/exp172-prefix-crop-last50-exp148-features`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp172-prefix-crop-last50-exp148-features
  - metadata pull: PASS (`id_no=125793351`, `enable_gpu=false`, `machine_shape=None`, `enable_internet=false`)
  - push 後 status: `KernelWorkerStatus.RUNNING`
  - train split notebooks は feature cache output 完了後に push する。
- 2026-07-03: push から約5分後の再確認でも `kentookumura/exp172-prefix-crop-last50-exp148-features` は `KernelWorkerStatus.RUNNING`。同系統 exp166 cache は長時間実行だったため、logs 空や RUNNING を理由に再 push しない。
- 2026-07-03: ユーザー連絡「完了しました」により feature cache v1 の完了を確認。
  - status: `KernelWorkerStatus.COMPLETE`
  - rows: 3,783,989
  - wells: 773
  - feature_count: 48 (`last50`)
  - elapsed_seconds: 6,536.996
  - feature cache bytes: 452,509,427
  - feature sha256: `02890a23a49012c8ab74f50c1eec0bb64292997ee9f5be60341e79b82ac01901`
  - decompressed sha256: `2061855a1d4d352c35ab6e4e9847d34bc68758526e8fce7a6c4ef3499ccb6a1e`
  - schema sha256: `b13bc3849170858bd6758b121b3be6c47eee5eb957b4f70567a15b2a49ee89a1`
  - summary sha256: `2a714f5d1a9f49c1c3bc2b1720443519d379f78dfff42f5e566de2df48e307f5`
- 2026-07-03: feature cache 完了後、CPU split train 3本を実行。
  - `kaggle kernels push -p experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/kaggle/train_lgb0`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp172-prefix-crop-last50-exp148-train-lgb0
  - `kaggle kernels push -p experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/kaggle/train_lgb1`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp172-prefix-crop-last50-exp148-train-lgb1
  - `kaggle kernels push -p experiments/exp172_prefix_crop_last50_replacement_only_on_exp148/kaggle/train_lgb2`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp172-prefix-crop-last50-exp148-train-lgb2
  - push 後 status は 3本とも `KernelWorkerStatus.RUNNING`。
- 2026-07-04: ユーザー連絡「完了しました」により split train v1 の完了を確認。3本とも `KernelWorkerStatus.COMPLETE`。
  - `lgb0`: pooled RMSE 8.583559279034894、prediction SHA `b0afa9077ced3bf94e44e19cfab113232476fbac93dd9d1302296adecad15998`、elapsed 13,004.467 sec。
  - `lgb1`: pooled RMSE 8.57512684958155、prediction SHA `806345f927418011c54d5524653dfa183bd2cdf3a7b4aecee57034eb1e827fa9`、elapsed 11,873.848 sec。
  - `lgb2`: pooled RMSE 8.586986606095419、prediction SHA `d06b2fa8bb5be99da9891d63563bf95e815ecc389b7411a9bbe5c2b796668cb8`、elapsed 13,428.471 sec。
  - best single は `lgb1` の 8.57512684958155。
  - exp148 `lgb_mean` 8.50128118189582 から +0.07384566768572931 悪化。
  - exp161 last50 add-only best single 8.56472499591314 から +0.010401853668408734 悪化。
  - exp166 tail500 replacement-only best single 8.566426970340796 から +0.008699879240753106 悪化。
  - split kernel prediction output は未取得のため cross-kernel `lgb_mean` は計算していない。全 single config が exp148 より明確に悪いため、output download、inference port、submit は行わない。

## 次アクション

1. exp172 は完了/不採用として扱う。
2. inference port / submit はしない。
3. prefix crop-window 系は現状の add-only / replacement-only では閉じ、残すなら低優先の `last50_first_prefix_feature_rebuild_on_exp148` に限定する。
