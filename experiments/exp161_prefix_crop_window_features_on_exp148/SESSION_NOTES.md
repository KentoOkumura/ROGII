# exp161_prefix_crop_window_features_on_exp148 セッションノート

## 目的

`KAGGLE_DIRECTION.md` の `prefix_crop_window_features_on_exp148` を実験化する。exp148 の既存 294 feature は置換せず、known prefix 末尾 crop-window 版の統計 / SC-NCC / multiobs confidence を add-only で追加する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle CPU train v4 running。prefix crop feature cache v1 を kernel source として読み込む構成。
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- Runtime: CPU (`cpu_deterministic_threads8`, `runtime.kaggle.enable_gpu=false`)

## 実装メモ

- exp148 の train / inference flow を `exp161_prefix_crop_window_features_on_exp148` 用にコピー。
- `prefix_crop_window_features_on_exp148.py` に `build_prefix_crop_window_features` を追加。
- 2026-06-30: CPU v1 が 1 時間超 fold0 未到達だったため、prefix crop 前処理を軽量化。
  - raw CSV 読み込みを必要列 (`MD`, `Z`, `GR`, `TVT_input`, `TVT`) のみに制限。
  - `ncc_max_starts=256` で crop SC/NCC の prefix start 候補を均一サンプリング。
  - `prefix_crop_build_start` / `prefix_crop_well_done` / `prefix_crop_build_done` と `lgb_fold_start` を JSON log として追加。
- 2026-07-01: CPU v2 は prefix crop build 完了後に `DeadKernelError`。`prefix_crop_build_done` は 7,143.007s、3,783,989 rows / 773 wells / 144 generated features。その後 `lgb_fold_start` 前に kernel died。原因は full frame への 144 feature 追加と LightGBM 行列化直前のメモリ圧迫と判断。
- 2026-07-01: v3 CPU-safe patch。
  - prefix crop window を `last50` のみに削減し、追加特徴を 144 から 48 に縮小。
  - prefix crop feature join は巨大 merge を避け、`id` / `well` の行順一致を確認したうえで列方向 concat に変更。
  - join 前後に `prefix_crop_join_start` / `prefix_crop_join_done` を JSON log 出力。
- 追加 window:
  - `tail1000`: `MD >= anchor_md - 1000`
  - `tail2000`: `MD >= anchor_md - 2000`
  - `last50`: known prefix 末尾 50 行
- 追加 feature group: `prefix_crop_window`
  - crop prefix slope / TVT range / std / calibration / prefix RMSE
  - crop SC/NCC `sc8` / `sc15` / `sc25` / ensemble delta and score
  - crop multiobs score / MAE / NCC、score max/mean/gap、full-minus-crop 差分、top1 changed flag、candidate outside crop prefix TVT range flag
- PF/Beam candidate 生成、U-projection、learned probability / expected error は crop-window 版へ置換しない。

## Kaggle train push 前ガード

- active variants: 1
  - `prefix_crop_window_addonly`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`cpu_deterministic_threads8`)
- 合計 booster: 15
- control 再学習: なし

## 実装確認

- 2026-06-30: `docs/legacy/steering/20260630-exp161-prefix-crop-window-features-on-exp148/` を作成。
- 2026-06-30: exp148 から `experiments/exp161_prefix_crop_window_features_on_exp148/` を作成。
- 2026-06-30: CPU 実行 config に変更。
- 2026-06-30: `.venv/bin/python -m py_compile ...` は PASS。
- 2026-06-30: `.venv/bin/ruff check ... --select F821` は PASS。
- 2026-06-30: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train.py` / `...inference.py` は PASS。
- 2026-06-30: `make validate-exp EXP=exp161_prefix_crop_window_features_on_exp148` は PASS。
- 2026-06-30: prefix crop feature builder の small fake-frame smoke は PASS。12 rows / 144 generated crop features、PerformanceWarning 0。
- 2026-06-30: train package を `kentookumura/exp161-prefix-crop-exp148-train` / title `exp161 prefix crop exp148 train` で prepare。metadata は `enable_gpu=false`、`enable_internet=false`、active mode `cpu_deterministic_threads8`。
- 2026-06-30: `make push-kaggle-train EXP=exp161_prefix_crop_window_features_on_exp148` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-train
- 2026-06-30: `kaggle kernels pull kentookumura/exp161-prefix-crop-exp148-train -p /tmp/kaggle-pull/exp161-prefix-crop-exp148-train-v1 -m` は成功。`id_no=125415387`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false` を確認。
- 2026-06-30: push 直後の `kaggle kernels logs kentookumura/exp161-prefix-crop-exp148-train` は空。`kaggle kernels status ...` は `KernelWorkerStatus.RUNNING`。
- 2026-06-30: `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp161-prefix-crop-exp148-train` は 5 分間 log 空のまま timeout。再 push はしない。
- 2026-06-30: follow timeout 後の通常 `kaggle kernels logs ...` も空。`kaggle kernels status ...` は `KernelWorkerStatus.RUNNING`。
- 2026-06-30: ユーザー指示「一旦これでkaggleで実行」に対し、既存の CPU train v1 を再確認。`kaggle kernels status kentookumura/exp161-prefix-crop-exp148-train` は `KernelWorkerStatus.RUNNING`。通常 `kaggle kernels logs ...` は引き続き空。重複 push はしない。
- 2026-06-30: v1 が 1 時間超 fold0 未到達のため CPU 前処理を修正。`py_compile`、`ruff --select F821`、train/inference `jupytext --to ipynb --test`、`make validate-exp EXP=exp161_prefix_crop_window_features_on_exp148` は PASS。
- 2026-06-30: smoke で `ncc_max_starts=16` の prefix crop build は 1 well / 12 rows / 144 generated features、0.235s。進捗 JSON log 出力を確認。
- 2026-06-30: train package を再 prepare 後、`make push-kaggle-train EXP=exp161_prefix_crop_window_features_on_exp148` により Kernel version 2 を push。URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-train
- 2026-06-30: v2 push 直後の `kaggle kernels status ...` は `KernelWorkerStatus.RUNNING`。通常 `kaggle kernels logs ...` はまだ空。
- 2026-07-01: `kaggle kernels status kentookumura/exp161-prefix-crop-exp148-train` は `KernelWorkerStatus.ERROR`。
- 2026-07-01: v2 logs 確認。prefix crop build は 773/773 wells まで完了し、`prefix_crop_build_done` 後に `nbclient.exceptions.DeadKernelError: Kernel died`。
- 2026-07-01: v3 patch 後、`py_compile`、`ruff --select F821`、train/inference `jupytext --to ipynb --test`、`make validate-exp EXP=exp161_prefix_crop_window_features_on_exp148` は PASS。
- 2026-07-01: v3 smoke で `last50` のみ prefix crop build は 1 well / 12 rows / 48 generated features、0.082s。
- 2026-07-01: v3 package を再 prepare 後、`make push-kaggle-train EXP=exp161_prefix_crop_window_features_on_exp148` は成功。Kernel version 3、URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-train
- 2026-07-01: v3 push 直後の `kaggle kernels status ...` は `KernelWorkerStatus.RUNNING`。通常 `kaggle kernels logs ...` はまだ空。
- 2026-07-01: ユーザー指摘により、LightGBM 学習前の prefix crop feature 生成を別 notebook に分離する方針へ変更。学習 notebook は cache を必須入力として読み、前処理にフォールバックしない。
- 2026-07-01: `exp161_prefix_crop_window_features_on_exp148_prefix_crop_features.py/.ipynb` を追加。出力は `exp161_prefix_crop_window_features_on_exp148_prefix_crop_train_features.csv.gz`、schema、summary、manifest。
- 2026-07-01: train notebook は `require_prefix_crop_cache=True` に変更。`data.prefix_crop_train_features_local` が未指定でも Kaggle input から cache 名で探索し、見つからなければ失敗する。
- 2026-07-01: `scripts/prepare_kaggle_notebooks.py` に `prefix_crop_features` kind を追加。
- 2026-07-01: `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test`、`make validate-exp EXP=exp161_prefix_crop_window_features_on_exp148` は PASS。
- 2026-07-01: prefix crop feature kernel を `kentookumura/exp161-prefix-crop-exp148-features` / title `exp161 prefix crop exp148 features` で prepare。metadata は CPU、internet off、kernel sources は exp072/exp145。
- 2026-07-01: `kaggle kernels push -p experiments/exp161_prefix_crop_window_features_on_exp148/kaggle/prefix_crop_features` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-features
- 2026-07-01: push 直後の `kaggle kernels status kentookumura/exp161-prefix-crop-exp148-features` は `KernelWorkerStatus.RUNNING`。通常 logs はまだ空。
- 2026-07-01: exp161 train v1/v2/v3 と prefix crop feature kernel v1 で、Kaggle CLI の `kaggle kernels logs` は実行中に毎回空のまま返り、完了または ERROR 後にまとめて stdout/stderr が取得される挙動を確認。今後この環境では CLI logs は完了後にまとめて出る前提で扱い、logs 空だけを根拠に失敗・slug 間違い・再 push と判断しない。恒久メモとして `.agents/skills/kaggle-platform/SKILL.md` と `docs/05_workflow.md` に追記済み。
- 2026-07-01: prefix crop feature kernel v1 は `KernelWorkerStatus.COMPLETE`。cache manifest は rows 3,783,989 / wells 773 / feature_count 48 / elapsed 10,438.318s。feature cache `exp161_prefix_crop_window_features_on_exp148_prefix_crop_train_features.csv.gz` は 452,509,419 bytes、sha256 `86b22a14b30425b079e532de0d3796f1e33bb9a25b1f61f6a5fcfc47d951a69b`、decompressed sha256 `2061855a1d4d352c35ab6e4e9847d34bc68758526e8fce7a6c4ef3499ccb6a1e`。
- 2026-07-01: train notebook `.ipynb` を `.py` から再生成し、`require_prefix_crop_cache=True` と `kentookumura/exp161-prefix-crop-exp148-features` kernel source が package に反映されていることを確認。
- 2026-07-01: `make push-kaggle-train EXP=exp161_prefix_crop_window_features_on_exp148` は成功。Kernel version 4、URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-train
- 2026-07-01: train v4 push 直後の `kaggle kernels status kentookumura/exp161-prefix-crop-exp148-train` は `KernelWorkerStatus.RUNNING`。通常 logs は完了前のため空。
- 2026-07-01: タイムアウト対策として train notebook を LGB config 単位に分割。
  - `exp161_prefix_crop_window_features_on_exp148_train_lgb0.py/.ipynb`: `selected_lgb_config_indices=[0]`
  - `exp161_prefix_crop_window_features_on_exp148_train_lgb1.py/.ipynb`: `selected_lgb_config_indices=[1]`
  - `exp161_prefix_crop_window_features_on_exp148_train_lgb2.py/.ipynb`: `selected_lgb_config_indices=[2]`
  - 各 kernel は 1 variant x 1 mode x 1 LGB config x 5 folds = 5 boosters。3 kernel 合計 15 boosters で combined train v4 と同じ学習内容。
  - 各 kernel は prefix crop feature cache kernel `kentookumura/exp161-prefix-crop-exp148-features` を input として参照し、学習中の prefix crop 前処理は実行しない。
- 2026-07-01: `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test` は split train lgb0/lgb1/lgb2 すべて PASS。
- 2026-07-01: split train package を以下の CPU / internet off / run_on_push kernel として prepare。
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb0`
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb1`
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb2`
- 2026-07-01: split train push 状況。
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb0`: Kernel version 1 push 成功。status は `KernelWorkerStatus.RUNNING`。
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb1`: Kernel version 1 push 成功。status は `KernelWorkerStatus.RUNNING`。
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb2`: 初回 push は `Maximum batch CPU session count of 5 reached`。短時間待機後の同 slug 再 push は `Notebook not found`。
  - lgb2 の中身は同一のまま slug を `kentookumura/exp161-prefix-crop-exp148-train-lgb2a` に変更して prepare。title/slug 不一致を修正後の push は再度 `Maximum batch CPU session count of 5 reached`。CPU 実行枠が空き次第、`train_lgb2` package を再 push する。
- 2026-07-02: ユーザー連絡により lgb0/lgb1 の完了を確認。CLI status でも以下を確認。
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb0`: `KernelWorkerStatus.COMPLETE`
  - `kentookumura/exp161-prefix-crop-exp148-train-lgb1`: `KernelWorkerStatus.COMPLETE`
- 2026-07-02: `kentookumura/exp161-prefix-crop-exp148-train-lgb2a` は枠解放後も `Notebook not found`。status/list とも Not Found のため、同一 notebook 内容のまま slug/title を `kentookumura/exp161-prefix-crop-exp148-train-lgb2b` / `exp161 prefix crop exp148 train lgb2b` に変更して再 prepare。
- 2026-07-02: `kaggle kernels push -p experiments/exp161_prefix_crop_window_features_on_exp148/kaggle/train_lgb2` は成功。`kentookumura/exp161-prefix-crop-exp148-train-lgb2b` Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp161-prefix-crop-exp148-train-lgb2b
- 2026-07-02: push 後 status は `KernelWorkerStatus.RUNNING`。通常 logs は完了前のため空の前提。
- 2026-07-02: ユーザー連絡「完了しました」により split train 3本の status / logs を確認。すべて `KernelWorkerStatus.COMPLETE`。
  - lgb0 kernel: `kentookumura/exp161-prefix-crop-exp148-train-lgb0` v1、elapsed 16,193.874s、rows 3,783,989、features 342、prefix crop features 48、pooled RMSE 8.573959480512093。fold RMSE: 9.086285031312698 / 8.911342038437647 / 7.487926993230402 / 8.349104868800723 / 8.932985315562743。
  - lgb1 kernel: `kentookumura/exp161-prefix-crop-exp148-train-lgb1` v1、elapsed 9,408.618s、rows 3,783,989、features 342、prefix crop features 48、pooled RMSE 8.575152249652412。fold RMSE: 9.029034734019946 / 8.677456435310157 / 7.49670851229148 / 8.57513786353678 / 9.005239802894584。
  - lgb2 kernel: `kentookumura/exp161-prefix-crop-exp148-train-lgb2b` v1、elapsed 14,034.888s、rows 3,783,989、features 342、prefix crop features 48、pooled RMSE 8.56472499591314。fold RMSE: 8.991120983399577 / 8.703167092408991 / 7.443797796472158 / 8.601688562839596 / 8.987295797886462。
  - best single config は lgb2 の CV 8.56472499591314。親 exp148 lgb_mean CV 8.50128118189582 から +0.06344381401732 悪化。
  - split notebook 内の `lgb_mean` は各 notebook で選択済み 1 config の平均なので単体 config と同値。3 config 横断の lgb_mean ensemble は split 出力を結合しないと未評価。

## 次アクション

1. 必要なら split 3本の OOF prediction artifact を取得して横断 lgb_mean ensemble CV を計算する。
2. 現時点の best single CV は exp148 より悪いため、提出候補にはしない。
