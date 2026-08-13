# exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148 セッションノート

## 概要

`backlog/KAGGLE_DIRECTION.md` の backlog `typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` を実験化する。目的は、exp196 pct40 hard-window base surface を exp148 ML surface に差し替えつつ、既存 exp145 `ll_*` learned-likelihood features を残したときの train-side 効果を低コストに診断すること。

## 実装方針

- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- base surface: `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement`
- learned-likelihood features: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- active variant: `pct40_base_surface_keep_exp145_ll_mixed_provenance`
- active feature groups: base 196 features、`projection_correction`、`u_disagreement`、`learned_likelihood_confidence`
- control 再学習: なし。exp148 historical CV / Public LB を baseline とする。
- inference / submit: 初期スコープ外。混合 provenance のまま submit しない。

## Kaggle train コストガード

- active variant 数: 1
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- active mode 数: 1 (`gpu_repro_guard_dp_threads8`)
- 合計 booster 数: 15
- 親 exp148 / control 再学習: なし

## 作業ログ

- 2026-07-05: `docs/legacy/steering/20260705-exp199-typewell-hard-window-pct40-base-surface-keep-exp145-ll-on-exp148/` を作成。
- 2026-07-05: `experiments/exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148/` を exp148 から作成。
- 2026-07-05: `typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148.py` を作成し、base cache default を exp196 pct40 hard-window artifact に変更。
- 2026-07-05: `config.yaml` を exp196 base surface + exp145 `ll_*` keep の mixed provenance 診断として更新。
- 2026-07-05: train notebook source を `data.base_surface_train_feature_cache_local` 参照に変更。inference notebook は train-side only の no-op 契約に変更。
- 2026-07-05: Jupytext で canonical train / inference `.ipynb` を再生成。
- 2026-07-05: `jupytext --to ipynb --test` は train / inference ともに PASS。
- 2026-07-05: `.venv/bin/python -m py_compile` は `typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148.py`、`settings.py`、train / inference source で PASS。
- 2026-07-05: `.venv/bin/ruff check ... --select F821` は PASS。
- 2026-07-05: `make validate-exp EXP=exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` は strict PASS。
- 2026-07-05: exp196 base cache / schema / summary と exp145 full-train learned-likelihood cache / schema / summary のローカル存在を確認。
- 2026-07-05: exp148 からコピーされた古い self-contained 派生ファイルを削除し、canonical train / inference と必要 helper に絞った。
- 2026-07-05: 削除後に `make validate-exp`、全 `.py` の `ruff --select F821`、全 `.py` の `py_compile` を再実行して PASS。
- 2026-07-05: train package を `kentookumura/exp199-pct40-base-keep-ll-train` / title `exp199 pct40 base keep ll train` で strict prepare した。metadata は `enable_gpu=true`、`enable_internet=false`、kernel sources は `kentookumura/exp196-typewell-hard-window-pct40-train` と `kentookumura/exp145-train`。
- 2026-07-05: ユーザー依頼により Kaggle train 実行へ進む。push 前コストガードは active variant 1、active GPU mode 1、LightGBM configs 3、folds 5、合計 15 boosters。control / parent 再学習なし。
- 2026-07-05: 実行コマンドは `make push-kaggle-train EXP=exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148`。対象 kernel は `kentookumura/exp199-pct40-base-keep-ll-train`。
- 2026-07-05: `make push-kaggle-train EXP=exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp199-pct40-base-keep-ll-train
- 2026-07-05: `kaggle kernels pull kentookumura/exp199-pct40-base-keep-ll-train -p /tmp/kaggle-pull/exp199-pct40-base-keep-ll-train-v1 -m` は成功。`id_no=126002124`、`enable_gpu=true`、`machine_shape=Gpu`、`enable_internet=false`、kernel sources は `kentookumura/exp145-train` と `kentookumura/exp196-typewell-hard-window-pct40-train`。
- 2026-07-05: `kaggle kernels status kentookumura/exp199-pct40-base-keep-ll-train` は `KernelWorkerStatus.RUNNING`。通常 `kaggle kernels logs ...` は実行中のため空。再 push はしない。
- 2026-07-05: 記録更新後の再確認でも `KernelWorkerStatus.RUNNING`。通常 logs はまだ空。`make validate-exp EXP=exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` は strict PASS。
- 2026-07-05: ユーザー報告後に `kaggle kernels status kentookumura/exp199-pct40-base-keep-ll-train` を再確認し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-07-05: `kaggle kernels output kentookumura/exp199-pct40-base-keep-ll-train -p /tmp/kaggle-output/exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148/train_v1` で output を取得。metrics / summary / manifest / feature schema / predictions / feature importance / by-well / bucket artifacts を確認した。
- 2026-07-05: train v1 は 3,783,989 rows / 773 wells / 294 features / 15 boosters、elapsed_seconds 12568.078 で完走。feature join coverage は base rows 3,783,989、learned rows 3,783,989、joined rows 3,783,989、dropped rows 0、full_train_coverage_pass true。
- 2026-07-05: pooled RMSE は `lgb0` 8.551067730689416、`lgb1` 8.533458031963196、`lgb2` 8.570960612427667、`lgb_mean` 8.496204218351805。exp148 GPU `lgb_mean` 8.50128118189582 から -0.005076963544015 の小改善。
- 2026-07-05: `lgb_mean` prediction SHA256 は `4e4ba51f815a8d64939c8b4acf4c91ef52af0666565b33a3b1de14f00fdf8585`。model manifest SHA256 は `516fe14fabd30c34dab8c85da2166e9ce0d0bc9ce629537975b2a9194f62ad21`、feature schema SHA256 は `da85f659658d3b50bb88aa863ceb89546dd99e361c669f79aa3dfe131259d944`、summary SHA256 は `9fd4d991255d64b38b9b1e969de929c6b24afac55139fe0412ea7e2266e54f63`。
- 2026-07-05: exp196 base cache SHA256 は gzip `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b` / decompressed `106cdfb266f93a0e45f25b281d3238c1fab0a24a84dac4c23187044022b5127e`。exp145 learned-likelihood cache SHA256 は gzip `298fdafd7376d0d551083cac26491901658001ed717b4caa7a9f8b32103886ff` / decompressed `e1c276d69e9355f6c03c18ac51a0883ee99ec6d80d040a5c62e5d55048bb7456`。

## 完了時判断

- train-side CV はわずかに positive なので、exp196 pct40 base surface は exp148 downstream ML に完全に拒否されてはいない。
- ただし `lgb2` は exp148 同 config 比で悪化し、改善幅も小さい。さらに `ll_*` が exp145/exp072 由来のままなので、hidden-safe inference の根拠にはしない。
- この exp199 は `completed_train_side_supported_no_submit` として閉じる。2026-07-05 のユーザー判断により、clean regeneration follow-up は追加価値が薄いとしてバックログから削除した。
