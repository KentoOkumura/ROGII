# exp166_prefix_crop_midwindow_replacement_on_exp148 セッションノート

## 目的

案2を先に実施する。exp161 の last50 add-only は exp148 を改善しなかったため、last50 へ進む前に `tail500` / `tail1000` の中間窓を replacement-only で評価する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle CPU split train lgb0/lgb1/lgb2 v2 complete / train-side rejected / no submit
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- Runtime: CPU (`cpu_deterministic_threads8`, `runtime.kaggle.enable_gpu=false`)
- 実行構成: 2段階。prefix crop feature cache notebook を完了させてから、split train notebook が cache を読み込んで学習した。

## 実装メモ

- exp161 の prefix crop feature builder と split train 構成をベースに、実験名を `exp166_prefix_crop_midwindow_replacement_on_exp148` へ分離した。
- `model.prefix_crop_window_features.windows` は `tail500` と `tail1000` のみ。
- 有効 variant:
  - `prefix_crop_tail500_replacement`
  - `prefix_crop_tail1000_replacement`
- replacement 対象:
  - exp072/exp092 full-prefix 系: `sc8_d`, `sc8_sc`, `sc15_d`, `sc15_sc`, `sc25_d`, `sc25_sc`, `sc_cons_d`, `sc_ens_d`, `sc_trust`, `cal_a`, `cal_b`, `pfx_rmse`, `slp_all`, `slp_z`, `slp_b_d_all`, `ktvt_range`, `ktvt_std`
  - exp145 learned likelihood multiobs 系: `ll_multiobs_score_*`, `ll_multiobs_mae_*`, `ll_multiobs_ncc_*`
- `learned_likelihood_confidence_no_multiobs` と `prefix_crop_tail500` / `prefix_crop_tail1000` group を notebook 実行時に構成し、variant の feature list へ渡す。
- 学習 notebook は `require_prefix_crop_cache=True` で実行し、cache が見つからなければ失敗する。
- 推論側にも学習側と同じ group 展開を入れ、保存モデル manifest の replacement variant を解決できるようにした。

## Kaggle train push 前ガード

- active variants: 2
  - `prefix_crop_tail500_replacement`
  - `prefix_crop_tail1000_replacement`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`cpu_deterministic_threads8`)
- 合計 booster: 30
- split train notebook ごとの booster: 10
- control 再学習: なし

## 実装確認

- 2026-07-02: `docs/legacy/steering/20260702-exp166-prefix-crop-midwindow-replacement-on-exp148/` を作成。
- 2026-07-02: `experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/` を exp161 からコピーし、案2 replacement-only 用に設定を変更。
- 2026-07-02: `.venv/bin/python -m py_compile experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/*.py` は PASS。
- 2026-07-02: `.venv/bin/ruff check experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/*.py --select F821` は PASS。
- 2026-07-02: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...` は prefix crop features / train / train_lgb0 / train_lgb1 / train_lgb2 / inference で PASS。
- 2026-07-02: 同 `.py` から `.ipynb` を再生成し、コピー元 exp161 の古い import / header を除去。
- 2026-07-02: `make validate-exp EXP=exp166_prefix_crop_midwindow_replacement_on_exp148` は PASS。
- 2026-07-02: feature cache package を `kentookumura/exp166-prefix-crop-midwindow-exp148-features` / title `exp166 prefix crop midwindow exp148 features` で prepare。metadata は CPU、internet off、kernel sources は exp072/exp145。
- 2026-07-02: 初回 `kaggle kernels push -p experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/kaggle/prefix_crop_features` は Kaggle API 応答 parse error `Expecting value: line 1 column 1 (char 0)`。続く `kaggle kernels list --mine --page-size 5` も `api.kaggle.com` connect timeout だったため、一時的な API 接続問題と判断。
- 2026-07-02: 120 秒待機後の同一 canonical slug 再 push は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp166-prefix-crop-midwindow-exp148-features
- 2026-07-02: `kaggle kernels pull kentookumura/exp166-prefix-crop-midwindow-exp148-features -p /tmp/kaggle-pull/exp166-prefix-crop-midwindow-exp148-features-v1 -m` は成功。`id_no=125650033`、`enable_gpu=false`、`machine_shape=None`、`enable_internet=false`、kernel sources は exp072/exp145。
- 2026-07-02: push 後 status は `KernelWorkerStatus.RUNNING`。通常 `kaggle kernels logs ...` は空。今後も完了まで CLI logs は空の前提で扱い、空ログだけで再 push しない。
- 2026-07-02: split train package を以下の CPU / internet off / run_on_push kernel として prepare。いずれも kernel source に `kentookumura/exp166-prefix-crop-midwindow-exp148-features` を含む。
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb0`
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb1`
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb2`
  - cache 完了前に push すると source output が未生成で失敗する可能性があるため、train push は cache 完了後に行う。
- 2026-07-02: ユーザー指示「先ほど作ったノートブックを実行してください」に対し、feature cache notebook は既に v1 が `run_on_push=true` で実行中であることを確認。`kaggle kernels status kentookumura/exp166-prefix-crop-midwindow-exp148-features` は `KernelWorkerStatus.RUNNING`。通常 logs は完了前のため空。
- 2026-07-02: ユーザー連絡「完了しました」により feature cache v1 を確認。`KernelWorkerStatus.COMPLETE`。
  - rows: 3,783,989
  - wells: 773
  - feature_count: 96 (`tail500` / `tail1000`)
  - elapsed_seconds: 9,879.882
  - feature cache bytes: 1,090,133,321
  - feature sha256: `43775b5e6a9d5a83bc817991cb0c57f5ea34071f4b7261f69912a83ae822a4cc`
  - decompressed sha256: `5af911c88878822a88fc48207284bb72bdb70821728efea7fa8c37d59fba1e1b`
  - schema sha256: `8844a3fbd94297a638394ee4a2a41b3b33927f196a89580a9118999dfaebef0f`
  - summary sha256: `e41f806960f91b788669b2affd762dcf0fd1a8b53e121e103d499cfdb13d3b38`
- 2026-07-02: cache 完了後、CPU split train 3本を実行。
  - `kaggle kernels push -p experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/kaggle/train_lgb0`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb0
  - `kaggle kernels push -p experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/kaggle/train_lgb1`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb1
  - `kaggle kernels push -p experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/kaggle/train_lgb2`: Kernel version 1、URL https://www.kaggle.com/code/kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb2
  - push 後 status は 3本とも `KernelWorkerStatus.RUNNING`。通常 logs は完了前のため空。
- 2026-07-02: ユーザー連絡「失敗しました」により split train v1 を確認。3本とも `KernelWorkerStatus.ERROR`。
  - lgb0/lgb1/lgb2 はすべて `prefix_crop_cache_loaded` 後、`prefix_crop_join_start` で kernel died。
  - lgb0 は rows 3,783,989 / features 96 cache load 後、`prefix_crop_join_start` の約 10 秒後に `nbclient.exceptions.DeadKernelError: Kernel died`。
  - lgb1/lgb2 も同じ。LightGBM fold 開始前なので、原因は 1.09GB の `tail500/tail1000` cache 全96列を full frame へ一括 concat するメモリピークと判断。
- 2026-07-02: メモリ修正。
  - `load_prefix_crop_feature_cache` に `selected_feature_columns` / `usecols` を追加。
  - train 本体は prefix crop 96列を一括 join せず、variant ごとに必要な window 48列だけを cache から読み込み、学習後に解放する構造へ変更。
  - `lgb0/lgb1/lgb2` の3分割は維持。各 notebook は `tail500` variant を 48列だけ join して学習し、解放後に `tail1000` variant を 48列だけ join して学習する。
  - `.venv/bin/python -m py_compile experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/*.py` は PASS。
  - `.venv/bin/ruff check experiments/exp166_prefix_crop_midwindow_replacement_on_exp148/*.py --select F821` は PASS。
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb0.py ...train_lgb1.py ...train_lgb2.py` は PASS。
  - `make validate-exp EXP=exp166_prefix_crop_midwindow_replacement_on_exp148` は PASS。
- 2026-07-02: 同じ canonical kernel id/title で split train package を再 prepare し、version 2 として再 push。
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb0`: Kernel version 2 push 成功、status `KernelWorkerStatus.RUNNING`。
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb1`: Kernel version 2 push 成功、status `KernelWorkerStatus.RUNNING`。
  - `kentookumura/exp166-prefix-crop-midwindow-exp148-train-lgb2`: Kernel version 2 push 成功、status `KernelWorkerStatus.RUNNING`。
- 2026-07-03: ユーザー連絡「完了しました」により split train v2 の完了を確認。3本とも `KernelWorkerStatus.COMPLETE`。
  - `lgb0`: `tail500` 8.566426970340796、`tail1000` 8.615045272753367。elapsed 30,436.394 sec。
  - `lgb1`: `tail500` 8.59529670454605、`tail1000` 8.589982014957261。elapsed 21,954.595 sec。
  - `lgb2`: `tail500` 8.638434980896186、`tail1000` 8.574216682757848。elapsed 18,444.034 sec。
  - best single は `prefix_crop_tail500_replacement` / `lgb0` の 8.566426970340796。
  - exp148 `lgb_mean` 8.50128118189582 から +0.065145788444976 悪化。
  - exp161 last50 add-only best single 8.56472499591314 からも +0.001701974427656 悪化。
  - split kernel prediction output は未取得のため lgb0/lgb1/lgb2 cross-ensemble は計算していない。全 single config が exp148 より明確に悪いため、output download と submit は行わない。

## 次アクション

1. exp166 は完了/不採用として扱う。
2. 案1 last50 replacement-only は isolated test として backlog に残すが、今回の tail500/tail1000 replacement-only と exp161 last50 add-only がともに negative なので優先度は下げる。
3. もし案1を実行する場合は、同じ 2段階 cache/train 構成にし、置換対象を今回より狭めた ablation を優先する。
