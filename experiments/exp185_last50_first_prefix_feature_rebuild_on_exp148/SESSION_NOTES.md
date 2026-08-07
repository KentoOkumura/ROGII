# exp185_last50_first_prefix_feature_rebuild_on_exp148 セッションノート

## 目的

`last50_first_prefix_feature_rebuild_on_exp148` backlog を実装する。exp161 last50 add-only、exp166 tail500/tail1000 replacement-only、exp172 last50 multiobs replacement-only は exp148 を改善しなかったため、既存特徴へ後付けする方向ではなく、known prefix source を先に last50 へ切ってから prefix 由来特徴を作り直す。

## 現在の状態

- Route: `ml_model`
- 状態: 完了 / negative / submit なし
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- Runtime: feature cache は完了済みの Kaggle GPU metadata を参照。split train は Kaggle CPU metadata。LightGBM train mode は `cpu_deterministic_threads8`。
- 実行構成: 2段階。prefix rebuild feature cache notebook を実行後、split train notebook `lgb0` / `lgb1` / `lgb2` が cache を読み込んで学習する。

## 実装メモ

- exp172 の 2段階 cache/train 構成をベースに exp185 へ分離。
- `model.prefix_crop_window_features.windows` は `last50` のみ。
- active variant は `last50_first_prefix_rebuild`。
- active variant では exp072/exp092 full-prefix 系 (`sc*`, `cal_*`, `pfx_rmse`, `known_len`, `slp_*`, `ktvt_*`, `gr_vs_slp_all`) と exp145 learned multiobs 系 (`ll_multiobs_score_*`, `ll_multiobs_mae_*`, `ll_multiobs_ncc_*`) を落とす。
- 追加する rebuild columns は last50 crop frame 由来の TVT aggregate/stat、x/y/z trajectory/geometry、GR quality、calibration、SC/NCC、multiobs score/MAE/NCC、candidate-vs-prefix range/outside flags。
- `last_known_tvt`、anchor row、PF/Beam 候補値、U-projection、learned probability/error model は既存 exp148/exp145 surface を読む。

## Kaggle train push 前ガード

- active variants: 1
  - `last50_first_prefix_rebuild`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。
- active modes: 1
  - `cpu_deterministic_threads8`
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- 合計 booster: 15
- split train notebook ごとの booster: 5
- 親 exp148 / control 再学習: なし

## 実装確認

- 2026-07-04: `.steering/20260704-exp185-last50-first-prefix-feature-rebuild-on-exp148/` を作成。
- 2026-07-04: `experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/` を exp172 からコピーし、exp185 名へリネーム。
- 2026-07-04: `last50_first_prefix_feature_rebuild_on_exp148.py` に X/Y/Z trajectory、last50 source crop、full-prefix base column drop、GPU active mode を実装。
- 2026-07-04: `.venv/bin/python -m py_compile experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/*.py` は PASS。
- 2026-07-04: `.venv/bin/ruff check experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/*.py --select F821` は PASS。
- 2026-07-04: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...` で prefix crop features / train / train_lgb0 / train_lgb1 / train_lgb2 / inference の `.ipynb` を再生成。
- 2026-07-04: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...` は PASS。
- 2026-07-04: `make validate-exp EXP=exp185_last50_first_prefix_feature_rebuild_on_exp148` は PASS。
- 2026-07-04: feature cache package を `kentookumura/exp185-last50-prefix-rebuild-exp148-features` / title `exp185 last50 prefix rebuild exp148 features` で prepare。初回 metadata は CPU、internet off、run_on_push true、kernel sources は exp072/exp145。
- 2026-07-04: package config で `selected_variant=last50_first_prefix_rebuild`、`active_modes=[gpu_repro_guard_dp_threads8]`、`train_lgb0/1/2 enable_gpu=true` を確認。
- 2026-07-04: 初回 feature cache push は `Maximum batch CPU session count of 5 reached` で失敗。feature cache は GPU を直接使わないが、CPU session 上限を避け、ユーザー指定の GPU 実行にも合わせるため `runtime.kaggle.prefix_crop_features.enable_gpu=true` に変更。
- 2026-07-04: T4 固定 (`machine_shape=NvidiaTeslaT4`) では `Notebook not found` が続いたため、LightGBM では T4 固有要件がないと判断し、既存 GPU + kernel_sources 成功例に合わせて default GPU metadata へ変更する。
- 2026-07-04: 長い slug `kentookumura/exp185-last50-prefix-rebuild-exp148-features` では default GPU でも `Notebook not found` が続いたため、同じ exp185 のまま短い意味付き slug `kentookumura/exp185-l50rebuild-features` / title `exp185 l50rebuild features` へ変更。
- 2026-07-04: feature cache notebook を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/prefix_crop_features`
  - kernel: `kentookumura/exp185-l50rebuild-features`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-features
  - metadata pull: PASS (`id_no=125833674`, `enable_gpu=true`, `machine_shape=Gpu`, `enable_internet=false`)
  - initial logs: empty; Kaggle CLI は実行中 logs を返さないことがあるため、空ログだけでは失敗扱いしない。
- 2026-07-04: `timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp185-l50rebuild-features` は出力なしで timeout。通常 logs も空。当時の status は `KernelWorkerStatus.RUNNING`。
- 2026-07-04: ユーザー側で feature cache 完了を確認。以降の feature cache 監視は停止する。
- 2026-07-04: `runtime.kaggle.train_kernel_sources` の feature cache source を実際に push 成功した `kentookumura/exp185-l50rebuild-features` に更新。
- 2026-07-04: split train package を短い slug で prepare。
  - `kentookumura/exp185-l50rebuild-lgb0`
  - `kentookumura/exp185-l50rebuild-lgb1`
  - `kentookumura/exp185-l50rebuild-lgb2`
  - metadata: `enable_gpu=true`, `run_on_push=true`, `enable_internet=false`
  - kernel sources: exp072 cache / exp145 learned likelihood / `kentookumura/exp185-l50rebuild-features`
- 2026-07-04: `train_lgb0` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb0`
  - kernel: `kentookumura/exp185-l50rebuild-lgb0`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb0
- 2026-07-04: `train_lgb1` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb1`
  - kernel: `kentookumura/exp185-l50rebuild-lgb1`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb1
- 2026-07-04: `train_lgb2` push は Kaggle GPU 同時 session 上限で未開始。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb2`
  - error: `Maximum batch GPU session count of 2 reached.`
  - package: `experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb2`
  - この GPU run は後続の CPU split run で supersede する。
- 2026-07-04: ユーザーから GPU split 実行失敗の報告。split するなら `lgb0` / `lgb1` / `lgb2` は CPU 実行にする方針へ変更。
  - `model.training.active_modes`: `cpu_deterministic_threads8`
  - `runtime.kaggle.train_lgb0/1/2.enable_gpu`: `false`
  - `inference.selected_mode`: `cpu_deterministic_threads8`
  - `train_lgb0/1/2` notebook 見出しを Kaggle CPU run に更新。
- 2026-07-04: CPU split run 向けに `train_lgb0` / `train_lgb1` / `train_lgb2` の `.ipynb` を再生成し、`jupytext --test` / `py_compile` / `ruff --select F821` / `make validate-exp` は PASS。
- 2026-07-04: CPU metadata で split train package を prepare。
  - `train_lgb0`: `kentookumura/exp185-l50rebuild-lgb0`, `enable_gpu=false`, active mode `cpu_deterministic_threads8`
  - `train_lgb1`: `kentookumura/exp185-l50rebuild-lgb1`, `enable_gpu=false`, active mode `cpu_deterministic_threads8`
  - `train_lgb2`: 初回は `kentookumura/exp185-l50rebuild-lgb2`, `enable_gpu=false`, active mode `cpu_deterministic_threads8`
- 2026-07-04: 同じ kernel id へ再 push する前に `train_lgb0` / `train_lgb1` は `kaggle kernels pull ... -m` で既存 metadata を確認。`train_lgb2` は Kaggle 側に存在せず `GetKernel` 500。
- 2026-07-04: CPU 版 `train_lgb0` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb0`
  - kernel: `kentookumura/exp185-l50rebuild-lgb0`
  - version: 2
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb0
- 2026-07-04: CPU 版 `train_lgb1` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb1`
  - kernel: `kentookumura/exp185-l50rebuild-lgb1`
  - version: 2
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb1
- 2026-07-04: CPU 版 `train_lgb2` は original slug `kentookumura/exp185-l50rebuild-lgb2` で `Notebook not found`。同じ exp のまま復旧用 slug `kentookumura/exp185-l50rebuild-cpu-lgb2` / title `exp185 l50rebuild cpu lgb2` に再 prepare して push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb2`
  - kernel: `kentookumura/exp185-l50rebuild-cpu-lgb2`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-cpu-lgb2
- 2026-07-04: ユーザーから CPU split 失敗の報告。Kaggle logs を確認し、3本とも同じ failure signature。
  - `train_lgb0` v2: `prefix_crop_variant_join_start` 後、`Kernel died while waiting for execute reply` / `nbclient.exceptions.DeadKernelError: Kernel died`
  - `train_lgb1` v2: 同上
  - `train_lgb2` v1: 同上
  - 直前ログは prefix crop cache 76 features / 3,783,989 rows / 773 wells の読み込み完了。`variant_frame` 横結合時のメモリ pressure と判断。
- 2026-07-04: メモリ対策を実装。
  - `load_prefix_crop_feature_cache`: schema / requested columns から numeric dtype を `np.float32` 指定して CSV 読み込み。
  - `load_prefix_crop_feature_cache`: finite check を全列一括 `to_numpy` ではなく列単位に変更し、一時巨大配列を避ける。
  - `run_last50_first_prefix_feature_rebuild_on_exp148`: prefix crop cache path では `frame` / `projection_features` / `learned_features` / `learned_features_source` を prefix join 前に解放。
  - `run_last50_first_prefix_feature_rebuild_on_exp148`: `variant_frame` を `full_frame` 全列 + prefix ではなく、`id` / `well` / `target` / `last_known_tvt` / selected non-prefix features / selected prefix features だけで構築。
- 2026-07-04: メモリ対策後の静的確認。
  - `.venv/bin/python -m py_compile ...` は PASS。
  - `.venv/bin/ruff check ... --select F821` は PASS。
  - `make validate-exp EXP=exp185_last50_first_prefix_feature_rebuild_on_exp148` は PASS。
- 2026-07-04: CPU split package を再 prepare し、`enable_gpu=false` と kernel sources exp072 / exp145 / `kentookumura/exp185-l50rebuild-features` を確認。
- 2026-07-04: メモリ対策版 CPU `train_lgb0` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb0`
  - kernel: `kentookumura/exp185-l50rebuild-lgb0`
  - version: 3
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb0
- 2026-07-04: メモリ対策版 CPU `train_lgb1` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb1`
  - kernel: `kentookumura/exp185-l50rebuild-lgb1`
  - version: 3
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-lgb1
- 2026-07-04: メモリ対策版 CPU `train_lgb2` を push。
  - command: `kaggle kernels push -p experiments/exp185_last50_first_prefix_feature_rebuild_on_exp148/kaggle/train_lgb2`
  - kernel: `kentookumura/exp185-l50rebuild-cpu-lgb2`
  - version: 2
  - URL: https://www.kaggle.com/code/kentookumura/exp185-l50rebuild-cpu-lgb2
- 2026-07-04: ユーザー側で完了確認。Kaggle logs を取得し、3本とも完了を確認。
  - `lgb0` v3 pooled CV: 8.636150399376218
  - `lgb1` v3 pooled CV: 8.583238238261155
  - `lgb2` v2 pooled CV: 8.583791509222095
  - feature count: 334
  - rows: 3,783,989
  - feature join coverage: 3,783,989 rows / 773 wells / dropped rows 0
  - prefix crop cache: 76 features / decompressed SHA `d7a95c9449d48ffe89efb80d281fcc24c6d1d007d61220599ea6e5491d1d23ea`
- 2026-07-04: 3-config ensemble 確認のため Kaggle output を `/tmp/kaggle-output/exp185_lgb0`, `/tmp/kaggle-output/exp185_lgb1`, `/tmp/kaggle-output/exp185_lgb2` に取得し、OOF predictions を結合。
  - `lgb_mean_split3` pooled CV: 8.544817143008956
  - prediction SHA: `df0367a0caf94334f6c8f4d2e2eb0658a20d312c67364b0e2de1af9391ead15d`
  - exp148 `lgb_mean` CV 8.50128118189582 から +0.043535961113136525 悪化。
  - exp172 best single 8.57512684958155 は上回ったが、親 exp148 anchor に届かないため不採用。
- 2026-07-04: 結論。`last50_first_prefix_feature_rebuild_on_exp148` は完了/negative。inference port / submit はしない。

## 次アクション

1. `KAGGLE_DIRECTION.md` から実装済み backlog を削除する。
2. `experiment_summary.md` を更新する。
