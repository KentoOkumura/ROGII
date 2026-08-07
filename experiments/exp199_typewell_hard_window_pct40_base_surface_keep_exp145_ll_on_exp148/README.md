# exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148

## 状態

- Route: `ml_model`
- Status: `completed_train_side_supported_no_submit`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- base surface 親: `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement`
- learned likelihood 親: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- 提出候補: いいえ。混合 provenance の train-side 診断として扱う。

## 仮説

exp196 pct40 hard-window cache は exp072 比で `likpf_mean`、`pf_ancc`、`beam_mean` を改善し、pct50 の early-range 破綻を緩めた。exp148 の base 196 features と projection / U-disagreement を exp196 由来に置換し、既存 exp145 `ll_*` を残した場合、ML が pct40 surface の改善を拾えるかを低コストに診断する。

ただし `ll_*` は exp145/exp072 由来の learned-likelihood surface なので、これは clean replacement ではない。改善しても直接 inference / submit には進めない。2026-07-05 のユーザー判断により、exp196 surface から `ll_*` を再生成する clean 版 follow-up も追加価値が薄いとしてバックログから外した。

## 検証方針

GroupKFold 5 folds by well。active variant は `pct40_base_surface_keep_exp145_ll_mixed_provenance` の 1 個だけ。GPU mode は `gpu_repro_guard_dp_threads8`、LightGBM config は exp063 family の 3 configs、合計 15 boosters。control / exp148 親の再学習はしない。

比較基準は exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960、exp196 direct PF/Beam metrics、pct50 exp192、near / longtail / true typewell pct bucket / worst-well / feature importance。

## 所見

2026-07-05 に Kaggle train v1 `kentookumura/exp199-pct40-base-keep-ll-train` が COMPLETE。3,783,989 rows / 773 wells / 294 features / 15 boosters で完走し、`lgb_mean` pooled RMSE は 8.496204218351805。exp148 GPU `lgb_mean` 8.50128118189582 から -0.005076963544015 の小改善だった。

一方で、これは `exp196 base + exp145/exp072-derived ll_*` の混合 provenance 診断なので、直接 inference / submit には進めない。改善幅が小さいため、clean replacement follow-up も実施しない方針に変更した。
