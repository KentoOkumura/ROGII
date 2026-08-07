# exp234_crossfitted_residual_scale_emission_hmm_on_exp218 セッションノート

## 目的

`crossfitted_residual_scale_emission_hmm_on_exp218` backlog を実装する。exp218 `lgb_mean` の保存済み OOF を HMM Gaussian emission center に固定し、well-grouped inner cross-fit の residual scale だけを row-wise sigma に使う。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train-side 完了、不採用
- CV / LB: RMSE 8.427231 / 未提出
- Parent/control retraining: なし
- 推論 / submit: disabled。raw-test residual-scale regeneration が未設計のため実行しない。
- 実行実績: v1 は 1 active variant × 5 residual-scale cross-fit fits × 1 HMM variant、v2 readout は HMM / scale / LightGBM を再実行せず完了。

## 実装メモ

- `residual_scale_crossfit.py`
  - saved exp218 `lgb_mean` OOF と exp072 row context を ID 一対一で merge する。
  - `log1p(md_since)`、center displacement、center step delta、last-known TVT から `log1p(clipped squared residual)` を well GroupKFold 5 folds で cross-fit する。
  - held-out sigma、fold separation、decile calibration、floor/cap rate、input/output SHA を保存する。
- `residual_scale_emission_hmm_audit.py`
  - scale guard を先に保存・判定し、失敗時は HMM を実行しない。
  - 通過時のみ exact HMM の single `lambda=0.50` variant と direct comparator を実行する。
- `exact_hmm_smoother.py`
  - exp229 の row-wise sigma interface を保ち、exp218 OOF center / cross-fitted sigma の記録文言へ更新した。

## 親実験との差分

- exp218 の 15 LightGBM booster と point OOF は completed artifact として読むだけで、control / baseline の再学習はしない。
- exp221 の fixed sigma を、same-well residual を含まない cross-fitted sigma にだけ差し替える。
- exp229 の q16 / q50 / q84 quantile train、lambda grid、inference / submit は実行経路から除外した。

## 実装コマンド

```bash
make new-steering EXP=exp234_crossfitted_residual_scale_emission_hmm_on_exp218
make new-exp EXP=exp234_crossfitted_residual_scale_emission_hmm_on_exp218 SOURCE=experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148
```

- `task` はこの環境にないため `make` を使用した。
- exp229 は HMM row-wise sigma の実装参照としてのみコピーし、quantile LightGBM train は exp234 の実行経路から外した。

## 再現性メモ

- seed / split: `GroupKFold(n_splits=5)` by `well`、shuffle なし。residual scale estimator は `random_state=42`、`early_stopping=false` に固定する。
- HMM RNG: なし。`outer_workers=1`、`numba_num_threads=1` を固定する。
- GPU: 使用しない。exp218 booster は再学習せず保存済み OOF を読む。
- SHA: exp218 source OOF、row context、residual-scale gzip/decompressed、HMM gzip/decompressed、readout summary を記録する。
- model manifest / submission SHA: 新規 LightGBM model / inference / submission がないため該当なし。

## 2026-07-12 実装検証

```bash
.venv/bin/python -m py_compile experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/*.py
.venv/bin/ruff check experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218 --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/exp234_crossfitted_residual_scale_emission_hmm_on_exp218_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/exp234_crossfitted_residual_scale_emission_hmm_on_exp218_inference.py
make validate-exp EXP=exp234_crossfitted_residual_scale_emission_hmm_on_exp218
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp234_crossfitted_residual_scale_emission_hmm_on_exp218 --notebook train --kernel-id kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train --title "exp234 crossfitted residual scale hmm exp218 train" --run-on-push --strict
```

- 結果: すべて pass。Jupytext から正規 train / inference notebook を生成し、strict experiment validation と Kaggle train package の Python compile / F821 を通過した。
- generated train metadata: CPU (`enable_gpu=false`)、internet disabled、canonical kernel id `kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train`。
- notebook source には `__file__` / `Path(__file__)` を残していない。
- Kaggle push / train 実行は未実施。ユーザー依頼は実装までのため、CPU runtime を消費する audit は開始していない。

## 2026-07-12 Kaggle train v1

```bash
kaggle kernels push -p experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/kaggle/train
```

- canonical kernel: `kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train`
- runtime: CPU、internet disabled、1 active variant、residual-scale GroupKFold 5 fits、HMM 1 variant、LightGBM booster 0、parent/control retraining なし。
- push 前の `kaggle kernels pull` は API 403 で既存 kernel を取得できなかった。push は v1 successful。直後の `kaggle kernels logs` は空で、実行中には CLI log が空になり得る既知挙動として監視を継続する。

### v1 failure diagnosis

- final status: `ERROR`。
- 失敗箇所: `direct_hmm_comparison.py` の `load_hidden_like_masks()`。HMM feature cache 生成後、hidden-like subgroup readout の fold assignment を解決できず停止した。
- 最初の意味のある traceback: `FileNotFoundError: No non-empty candidate path exists`。
  - 実際に存在する artifact / Kaggle input basename: `exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv`
  - exp234 config が指定した誤った basename: `exp115_hidden_like_spatial_holdout_from_public_ppt_fold_assignments.csv`
- 影響範囲: exp218 OOF center の読み込み、residual scale 5-fold cross-fit、pre-HMM guard、773 well / 3,783,989 rows の HMM cache までは完了した。direct comparison の overall / distance bucket / hidden-like / by-well / step-delta summary と audit summary は未生成。
- v1 evidence:
  - residual-scale guard: pass。Spearman `0.326486`、top/bottom scale-decile RMSE ratio `3.578534`、floor rate `0.180690`、cap rate `0.0`、fold well overlap `0`。
  - HMM cache: elapsed `26,687.541` sec、773 / 773 well、rows `3,783,989`、decompressed feature SHA `45c3b4a60a1f83e55c0b2aa965a4971adeb79bb637688b31de78fa88cfa6a911`。
- HMM を再計算する train v2 ではなく、v1 cache を input にした comparison-only readout を同じ exp234 に追加する方針へ切り替えた。

## comparison-only readout 実装

- v1 HMM cache は output に存在し、raw SHA `86d6cebc43ceeb21af230a23a131629454f512047d433c8908cd642d784eac60`、decompressed SHA `45c3b4a60a1f83e55c0b2aa965a4971adeb79bb637688b31de78fa88cfa6a911` を確認した。
- `comparison.hidden_like.fold_assignment_candidates` を実在する exp115 `_from_ppt_` basename に修正した。
- `comparison_readout.py` と `exp234_*_train_aggregate.ipynb` を追加。v1 HMM cache を kernel source として読む `run_direct_comparison()` のみを実行し、residual-scale fitting / exact HMM generation は呼ばない。
- planned comparison-only cost: residual-scale fit 0、HMM variant 0、LightGBM booster 0、parent/control retraining なし、CPU / internet disabled。
- canonical readout kernel: `kentookumura/exp234-residual-scale-hmm-exp218-readout`。

### 静的検証

```bash
.venv/bin/python -m py_compile experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/comparison_readout.py experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/exp234_crossfitted_residual_scale_emission_hmm_on_exp218_train_aggregate.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/exp234_crossfitted_residual_scale_emission_hmm_on_exp218_train_aggregate.py
make validate-exp EXP=exp234_crossfitted_residual_scale_emission_hmm_on_exp218
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp234_crossfitted_residual_scale_emission_hmm_on_exp218 --notebook train_aggregate --kernel-id kentookumura/exp234-residual-scale-hmm-exp218-readout --title "exp234 residual scale hmm exp218 readout" --run-on-push --strict
```

- result: pass。package metadata は CPU、internet disabled、v1 HMM cache / exp115 / exp218 / exp148 / exp193 / exp072 kernel source を含む。

### comparison-only execution block

```bash
kaggle kernels push -p experiments/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/kaggle/train_aggregate
```

- result: Kaggle API が `Maximum batch CPU session count of 5 reached.` を返し、readout kernel は作成・実行されなかった。
- `kaggle kernels list --mine --page-size 20` では既存 notebook 一覧は読めるが、同時 session を特定して安全に停止できる情報は返らない。
- 他の Kaggle CPU session を停止する権限はないため、slot が空いた後に同じ canonical readout kernel を push する。HMM / residual-scale / LightGBM の再実行は不要。

### 2026-07-13 readout kernel 作成エラーと復旧先

- CPU quota の解消後に readout 専用 slug を再 push したが、`Notebook not found` が返った。`dataset_sources=[]`、`run_on_push=false` にした最小 metadata でも同じため、v1 HMM cache Dataset や input attachment が原因ではない。
- 新規 slug の登録を繰り返さず、既存の `kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train` を同一実験の comparison-only **version 2** として使う。train v1 の入力・HMM cache の SHA と failure evidence は維持し、v2 は修正済み exp115 fold assignment と cache Dataset の direct comparison だけを実行する。
- v2 の計算量: residual-scale fit 0、HMM variant 0、LightGBM booster 0、parent/control retraining なし、CPU / internet disabled。
- 2026-07-13: 上記既存 kernel に comparison-only notebook を version 2 として push し、Kaggle status `RUNNING` を確認した。URL: https://www.kaggle.com/code/kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train

### 2026-07-13 comparison-only v2 完了

- final status: `COMPLETE`。Kaggle kernel `kentookumura/exp234-crossfitted-residual-scale-hmm-exp218-train` v2 は `comparison_only_readout_completed` で終了した（elapsed `212.952` sec）。
- 実行契約を確認: residual-scale fit 0、HMM variant 0、LightGBM booster 0、parent/control retraining なし。v1 HMM cache content SHA は期待値どおり `45c3b4a60a1f83e55c0b2aa965a4971adeb79bb637688b31de78fa88cfa6a911`、ID mismatch は 0。
- best candidate: `hmm_lgb_exp218_lgb_mean_band_sf0250_sc4000_l0500`。3,783,989 rows / 773 wells で RMSE `8.427231402`、MAE `5.155675839`、exp218 OOF 比 `-0.048573356`、exp148 比 `-0.074059583`、exp193 比 `-0.029444651`。
- hidden-like: spatial RMSE `9.578578`（exp218 比 `-0.083018`）、typewell-purged RMSE `9.547965`（`-0.088034`）。distance bucket は全 6 bucket で exp218 より改善、`1000_plus` は `-0.050661`。
- by-well: exp218 比 526 / 773 wells 改善、247 wells 悪化。最大悪化は `f88ddb26` の `+1.257773` RMSE。step-delta p99 `0.078`、`>5/10/25` rate はすべて 0。
- 生成物 SHA: comparison-only summary `86648cd25b8663ca63c83caba406ca81a293cf85f4b0697100240778bb73aeea`、overall metrics `8b9e9b4ea37ee8e5c9fb47aecfa79f2b85acb8ec2f726f518d95f93a89923f64`。
- 判定: exp218 point OOF を中心に保つ効果は確認できたが、exp221 fixed-sigma HMM の RMSE `8.327736951` より `+0.099494451` 悪い。raw-test residual-scale 再生成、inference、submission は禁止のまま train-side 不採用とする。

## 次のアクション

1. exp234 は完了・不採用として維持し、inference / submit を作らない。
2. 次に HMM sigma を検討するなら、同一 exp218 center の scalar sigma 対照を先に固定し、cross-fitted scale の fixed-sigma 縮小を有限 ablation として比較する。
