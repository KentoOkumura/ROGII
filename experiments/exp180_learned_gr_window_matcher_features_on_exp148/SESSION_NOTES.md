# exp180_learned_gr_window_matcher_features_on_exp148 セッションノート

## 目的

`KAGGLE_DIRECTION.md` の `learned_gr_window_matcher_features_on_exp148` を実験化する。exp178 で positive だった known-prefix supervised GR window matcher を、候補 TVT の直接置換ではなく exp148 ML anchor への hidden-safe confidence feature として add-only 評価する。

## 現在の状態

- Route: `ml_model`
- 状態: `completed_train_side_rejected_no_submit`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 提出: なし

## 実装メモ

- exp161 の exp148 add-only / split train 構成を scaffold として利用し、prefix-crop 固有処理を learned GR matcher scorer へ差し替えた。
- 新規 feature group: `learned_gr_window_matcher`
  - per-candidate `grm_prob_*`
  - per-candidate `grm_expected_error_*`
  - `grm_prob_margin_top1_top2`
  - `grm_prob_entropy`
  - `grm_real_vs_shuffled_gap_*`
  - `grm_real_vs_no_gr_gap_*`
  - candidate family indicator
  - `md_since` interaction
- scorer labels は finite observed `TVT_input` prefix rows だけから作る。評価 tail true TVT、NaN `TVT_input` rows、oracle rank は使わない。
- train feature cache は `scorer_training.fold_safe_by_well=true`。validation well の matcher score は他 well の prefix pair で fit した scorer から作る。
- candidate TVT hard switch、softmax weighted TVT、midpoint、direct correction、PF/Beam 再生成は実装しない。

## Kaggle train push 前ガード

- active variants: 1
  - `learned_gr_window_matcher_addonly`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。exp148 の既存 CV / LB を historical baseline として参照する。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`cpu_deterministic_threads8`)
- 合計 booster: 15
- control 再学習: なし

## 実装確認

- 2026-07-03: `.steering/20260703-exp180-learned-gr-window-matcher-features-on-exp148/` を作成。
- 2026-07-03: `experiments/exp180_learned_gr_window_matcher_features_on_exp148/` を exp161 scaffold から作成し、exp180 用に rename。
- 2026-07-03: `learned_gr_window_matcher_features_on_exp148.py` に fold-safe learned GR matcher scorer feature builder を実装。
- 2026-07-03: `scripts/prepare_kaggle_notebooks.py` に `gr_matcher_features` notebook kind を追加。
- 2026-07-03: `config.yaml` を `learned_gr_window_matcher_addonly` / `learned_gr_window_matcher` feature group に更新。
- 2026-07-03: Jupytext で `gr_matcher_features`、train、train_lgb0/1/2、inference notebook を再生成。
- 2026-07-03: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...` は PASS。
- 2026-07-03: `.venv/bin/python -m py_compile ...` は PASS。
- 2026-07-03: `.venv/bin/ruff check ... --select F821` は PASS。
- 2026-07-03: `make validate-exp EXP=exp180_learned_gr_window_matcher_features_on_exp148` は PASS。
- 2026-07-03: `make prepare-kaggle-notebooks EXP=exp180_learned_gr_window_matcher_features_on_exp148 EXTRA_ARGS="--notebook gr_matcher_features --kernel-id kentookumura/exp180-gr-matcher-exp148-features --title 'exp180 gr matcher exp148 features' --run-on-push --strict"` は PASS。push はしていない。
- 2026-07-03: 同じ command で `gr_matcher_features` package を再 prepare。
- 2026-07-03: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/gr_matcher_features` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp180-gr-matcher-exp148-features
- 2026-07-03: `kaggle kernels pull kentookumura/exp180-gr-matcher-exp148-features -p /tmp/kaggle-pull/exp180-gr-matcher-exp148-features-v1 -m` は成功。`id_no=125817644`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072 / exp145。
- 2026-07-03: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-features` は `KernelWorkerStatus.RUNNING`。
- 2026-07-03: `kaggle kernels logs kentookumura/exp180-gr-matcher-exp148-features` は実行中のため warning 以外のログなし。空ログは失敗扱いにせず、同じ kernel id の完了を待つ。
- 2026-07-03: `make update-summary` は PASS。`experiment_summary.md` の exp180 status は `kaggle_feature_cache_running`。
- 2026-07-03: `make validate-exp EXP=exp180_learned_gr_window_matcher_features_on_exp148` は PASS。
- 2026-07-03: 最終確認の `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-features` は `KernelWorkerStatus.RUNNING`。
- 2026-07-03: ユーザーから `gr_matcher_features` 完了連絡。`kaggle kernels status kentookumura/exp180-gr-matcher-exp148-features` で `KernelWorkerStatus.COMPLETE` を確認。
- 2026-07-03: `kaggle kernels logs kentookumura/exp180-gr-matcher-exp148-features` で cache 生成完了を確認。rows 3,783,989、wells 773、features 61、feature file `exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_train_features.csv.gz`、feature sha256 `e8ac74c1c0c29e16be411511e507cf72c88e71a39c934bce00b163954c2f487b`、decompressed sha256 `bd3d0f3f59c3a35aad6b00ec29117ac63b5af9de27e8231f86ae985b3f2f2abe`、bytes 832,228,598。
- 2026-07-03: `make prepare-kaggle-notebooks EXP=exp180_learned_gr_window_matcher_features_on_exp148 EXTRA_ARGS="--notebook train_lgb0 --kernel-id kentookumura/exp180-gr-matcher-exp148-train-lgb0 --title 'exp180 gr matcher exp148 train lgb0' --run-on-push --strict"` は PASS。
- 2026-07-03: `make prepare-kaggle-notebooks EXP=exp180_learned_gr_window_matcher_features_on_exp148 EXTRA_ARGS="--notebook train_lgb1 --kernel-id kentookumura/exp180-gr-matcher-exp148-train-lgb1 --title 'exp180 gr matcher exp148 train lgb1' --run-on-push --strict"` は PASS。
- 2026-07-03: `make prepare-kaggle-notebooks EXP=exp180_learned_gr_window_matcher_features_on_exp148 EXTRA_ARGS="--notebook train_lgb2 --kernel-id kentookumura/exp180-gr-matcher-exp148-train-lgb2 --title 'exp180 gr matcher exp148 train lgb2' --run-on-push --strict"` は PASS。
- 2026-07-03: train package metadata は 3 split とも `enable_gpu=false`、`enable_internet=false`、kernel sources に exp072 / exp145 / exp180 features を含む。
- 2026-07-03: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb0` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp180-gr-matcher-exp148-train-lgb0
- 2026-07-03: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb1` は `Kernel push error: Maximum batch CPU session count of 5 reached.` で未開始。
- 2026-07-03: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb2` は `Kernel push error: Maximum batch CPU session count of 5 reached.` で未開始。
- 2026-07-03: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。
- 2026-07-03: `kaggle kernels pull kentookumura/exp180-gr-matcher-exp148-train-lgb0 -p /tmp/kaggle-pull/exp180-gr-matcher-exp148-train-lgb0-v1 -m` は成功。`id_no=125820850`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072 / exp145 / exp180 features。
- 2026-07-03: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb1` と `...train-lgb2` は 404。CPU session limit により Kaggle 側に notebook は作成されていない。
- 2026-07-03: 最終確認の `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。
- 2026-07-03: `kaggle kernels logs kentookumura/exp180-gr-matcher-exp148-train-lgb0` は実行中のため warning 以外のログなし。空ログは失敗扱いにせず、同じ kernel id の完了を待つ。
- 2026-07-03: ユーザーから `train_lgb0` 失敗連絡。`kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.ERROR`。
- 2026-07-03: `kaggle kernels logs kentookumura/exp180-gr-matcher-exp148-train-lgb0` で原因確認。`gr_matcher_cache_loaded` 後、`gr_matcher_join_start` の直後に `Kernel died while waiting for execute reply` / `nbclient.exceptions.DeadKernelError: Kernel died`。Python traceback ではなく memory/OOM 系の kernel death と判断。
- 2026-07-03: memory 対策として `learned_gr_window_matcher_features_on_exp148.py` を修正。CSV loader は numeric columns を読み込み時点で `float32` に固定し、巨大な `concat` をやめて projection / learned / gr matcher columns を列単位で追加する。`require_gr_matcher_cache=true` では exp145 source frame を gr join 前に解放し、学習不要な anchor columns を drop する。
- 2026-07-03: `.venv/bin/python -m py_compile experiments/exp180_learned_gr_window_matcher_features_on_exp148/learned_gr_window_matcher_features_on_exp148.py` は PASS。
- 2026-07-03: `.venv/bin/ruff check experiments/exp180_learned_gr_window_matcher_features_on_exp148/learned_gr_window_matcher_features_on_exp148.py --select F821` は PASS。
- 2026-07-03: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp180_learned_gr_window_matcher_features_on_exp148/exp180_learned_gr_window_matcher_features_on_exp148_train_lgb0.py` は PASS。
- 2026-07-03: `make validate-exp EXP=exp180_learned_gr_window_matcher_features_on_exp148` は PASS。
- 2026-07-03: 修正版で `train_lgb0` / `train_lgb1` / `train_lgb2` packages を再 prepare。
- 2026-07-03: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb0` は成功。Kernel version 2、URL: https://www.kaggle.com/code/kentookumura/exp180-gr-matcher-exp148-train-lgb0
- 2026-07-03: `kaggle kernels pull kentookumura/exp180-gr-matcher-exp148-train-lgb0 -p /tmp/kaggle-pull/exp180-gr-matcher-exp148-train-lgb0-v2 -m` は成功。`id_no=125820850`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072 / exp145 / exp180 features。
- 2026-07-03: v2 push 直後の `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。
- 2026-07-03: v2 push 直後の `kaggle kernels logs kentookumura/exp180-gr-matcher-exp148-train-lgb0` は warning 以外のログなし。完了または再エラーまで同じ kernel id のまま確認する。
- 2026-07-03: 追加待機後の `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。CLI logs は実行中のため warning 以外なし。v1 の `gr_matcher_join_start` 直後の即時 kernel death は再発していない。
- 2026-07-03: 修正版 package で `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb1` を試行したが `Kernel push error: Notebook not found`。`status` は 404、`pull -m` は 500。metadata の id/title/source は正常なため、slug を増やさず同じ `kentookumura/exp180-gr-matcher-exp148-train-lgb1` で後で再試行する。
- 2026-07-03: `make update-summary` は PASS。
- 2026-07-03: `make validate-exp EXP=exp180_learned_gr_window_matcher_features_on_exp148` は PASS。
- 2026-07-03: 最終確認の `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。
- 2026-07-04: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb0` は `KernelWorkerStatus.RUNNING`。
- 2026-07-04: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb1` は canonical slug `kentookumura/exp180-gr-matcher-exp148-train-lgb1` で再度 `Kernel push error: Notebook not found`。
- 2026-07-04: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb2` も canonical slug `kentookumura/exp180-gr-matcher-exp148-train-lgb2` で `Kernel push error: Notebook not found`。package 内の `.ipynb` と `kernel-metadata.json` の `code_file` は存在し一致しているため、Kaggle API 側の stale / broken notebook state と判断。
- 2026-07-04: 実行を優先し、generated package の metadata だけ retry slug に変更。`train_lgb1`: `kentookumura/exp180-gr-matcher-exp148-train-lgb1-r1` / title `exp180 gr matcher exp148 train lgb1 r1`。`train_lgb2`: `kentookumura/exp180-gr-matcher-exp148-train-lgb2-r1` / title `exp180 gr matcher exp148 train lgb2 r1`。
- 2026-07-04: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb1` は retry slug で成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp180-gr-matcher-exp148-train-lgb1-r1
- 2026-07-04: `kaggle kernels push -p experiments/exp180_learned_gr_window_matcher_features_on_exp148/kaggle/train_lgb2` は retry slug で成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp180-gr-matcher-exp148-train-lgb2-r1
- 2026-07-04: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb1-r1` は `KernelWorkerStatus.RUNNING`。
- 2026-07-04: `kaggle kernels status kentookumura/exp180-gr-matcher-exp148-train-lgb2-r1` は `KernelWorkerStatus.RUNNING`。
- 2026-07-04: `kaggle kernels pull kentookumura/exp180-gr-matcher-exp148-train-lgb1-r1 -p /tmp/kaggle-pull/exp180-gr-matcher-exp148-train-lgb1-r1-v1 -m` は成功。`id_no=125829796`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072 / exp145 / exp180 features。
- 2026-07-04: `kaggle kernels pull kentookumura/exp180-gr-matcher-exp148-train-lgb2-r1 -p /tmp/kaggle-pull/exp180-gr-matcher-exp148-train-lgb2-r1-v1 -m` は成功。`id_no=125829806`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072 / exp145 / exp180 features。
- 2026-07-04: `make update-summary` は PASS。
- 2026-07-04: `make validate-exp EXP=exp180_learned_gr_window_matcher_features_on_exp148` は PASS。
- 2026-07-04: 最終確認の status は `train_lgb0` v2、`train_lgb1-r1` v1、`train_lgb2-r1` v1 がすべて `KernelWorkerStatus.RUNNING`。
- 2026-07-04: ユーザーから 3 split 完了連絡。`kaggle kernels status` で `train_lgb0` v2、`train_lgb1-r1` v1、`train_lgb2-r1` v1 がすべて `KernelWorkerStatus.COMPLETE` であることを確認。
- 2026-07-04: `kaggle kernels logs` を `/tmp/exp180_lgb0_logs.json`、`/tmp/exp180_lgb1_logs.json`、`/tmp/exp180_lgb2_logs.json` に保存し、final metrics を確認。
- 2026-07-04: OOF ensemble 計算のため、Kaggle output を `kaggle/output/train_lgb0_v2`、`kaggle/output/train_lgb1_r1`、`kaggle/output/train_lgb2_r1` に取得。各 split の prediction、metrics、by-well、bucket、feature importance、model manifest を確認できる状態にした。
- 2026-07-04: 個別 pooled OOF は `lgb0` 8.554800137862696、`lgb1` 8.581198811251788、`lgb2` 8.5779985181889。各 split の fold metrics と best iteration は `metrics.json` に記録。
- 2026-07-04: 3 split prediction を row id / well / target で align し、`lgb_mean` ensemble OOF を計算。rows 3,783,989、wells 773、CV 8.5145263671875。結果を `kaggle/output/exp180_ensemble_lgb_mean_metrics.json` に保存。
- 2026-07-04: exp148 `lgb_mean` CV 8.50128118189582 から +0.0132451852916803 悪化。feature importance では `grm_no_gr_prob_*` / `grm_shuffled_prob_*` が中位に入るが、global OOF 改善には至らない。
- 2026-07-04: worst wells top3 は `86454a6f` RMSE 48.29558181762695、`1b1eba53` 45.27809524536133、`fb03ae90` 45.117767333984375。1000+ distance bucket は個別 split で 9.37612533569336 / 9.40521240234375 / 9.403876304626465。
- 2026-07-04: global OOF が baseline 未達のため、hidden-like stress、inference port、submit-check、submit は実行しない判断にした。

## 次アクション

1. `metrics.json`、`result.md`、`README.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` の negative close 記録を確認する。
2. `learned_gr_window_matcher_features_on_exp148` は完了/不採用として閉じる。同じ exp148 add-only 設計では追加拡張しない。
