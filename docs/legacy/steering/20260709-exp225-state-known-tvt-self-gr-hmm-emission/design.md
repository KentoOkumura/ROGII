# 設計

## アプローチ

exp223 をコピー元にし、exp209 exact HMM の grid、transition、typewell GR emission、direct comparison flow は維持する。変更点は self-GR emission surface の作り方だけに限定する。

exp223 は visible prefix GR motif matching から self-GR likelihood surface を grid 全体へ作った。exp225 では motif matching をやめ、known prefix の finite `TVT_input` と finite `GR` から `TVT_input -> GR` 曲線を作る。HMM candidate state `grid[j]` が known-prefix TVT 範囲内のときだけ、評価 row の GR とその曲線上の GR の Gaussian likelihood を weak boost として足す。範囲外 state は self-GR neutral、つまり boost 0 とする。

## 実験範囲

- 対象実験: `exp225_state_known_tvt_self_gr_hmm_emission`
- Route: `ensemble`
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`、HMM base は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: self-GR emission method、state mask、active variant 1 本
- 固定する変数: HMM grid / transition / typewell GR emission、exp072 baseline cache、hidden-like readout、direct comparison metrics

## 実装詳細

`exact_hmm_smoother.py` に `build_state_known_tvt_self_gr_likelihood_surface()` を追加する。

- known prefix から `curve_tvt` / `curve_gr` を作る。
- duplicate `TVT_input` は TVT ごとに GR 平均へ集約する。
- `curve_gr` は小さい rolling median で平滑化する。
- `grid` に対して `state_mask = known_tvt_min <= grid <= known_tvt_max` を作る。
- `state_mask` 外の self-GR centered log-likelihood は 0 のままにする。
- `boost_only` では positive centered log-likelihood だけを `clip=1.0` で足す。
- row-level diagnostic として `self_gr_quality`、`self_gr_peak_tvt`、`self_gr_peak_gap`、`self_gr_typewell_agreement`、`self_gr_valid`、`self_gr_state_valid_rate` を保存する。

## 再現性設計

- seed policy: HMM と self-GR curve evaluation は no RNG。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072 baseline cache は保存済み生成物を読むだけ。exp225 では再生成しない。
- 並列処理と乱数の関係: `outer_workers=2` は well 並列だが RNG を使わない。floating order による微小差は metrics と decompressed content SHA で記録する。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: train feature cache の raw gzip SHA と decompressed content SHA を分けて記録する。raw-test regeneration は初回範囲外。
- model manifest / prediction / submission SHA 記録方針: booster 0、submission なし。model/prediction/submission SHA は対象外。
- Kaggle package bootstrap 確認方針: push 前に `prepare_kaggle_notebooks --strict` を使い、metadata と bootstrap config の整合を確認する。

## リスク

- リークリスク: known prefix の finite `TVT_input` だけを使う。unknown suffix の true `TVT`、absolute error、oracle rank は surface 作成や alpha/clip 設定に使わない。
- CV/LB 不一致リスク: train-side diagnostic なので、この段階では LB 判断をしない。pass した場合も同じ exp 内で raw-test-safe regeneration を実装してから判断する。
- ランタイム/メモリリスク: exp223 は 2 variants で約10h50m。exp225 は 1 variant なので短縮見込みだが、HMM grid は同じため 12h guard を維持する。
- 再現性リスク: outer parallel による微小 floating 差はあり得る。deterministic submission anchor とは扱わない。
