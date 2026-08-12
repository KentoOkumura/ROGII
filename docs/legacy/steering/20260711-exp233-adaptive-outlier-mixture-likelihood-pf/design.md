# 設計

## アプローチ

exp072 likelihood-PF の各粒子について、標準化 GR residual を `r` とする。
通常 row は既存の `L_gaussian=exp(-0.5*r^2)` を使う。exp232 と同じ target-free
gate が発火した row だけ、次の mixture を使う。

```
L = (1 - epsilon) * L_gaussian + epsilon * L_uniform
L_uniform = sqrt(2*pi) * gr_sigma / (uniform_gr_max - uniform_gr_min)
```

`L_uniform` は Gaussian 密度を既存の unnormalised likelihood と同じ尺度に揃えた
一様密度で、row 内の全粒子に同じ値である。そのため、outlier component 自体が
particle state / candidate TVT を選好しない。`epsilon=0`、temperature、broad
Gaussian component はこの実験から除外する。

Gate は高 innovation を必須とし、raw GR change point、short/long GR novelty、
pre-update ESS ratio、pre-update max particle weight の少なくとも一つを裏付けと
する。gate は seed ごとの pre-update state に依存するため、diagnostic の rate は
seed 平均であり、全 variant に共通の row mask とは主張しない。

## 実験範囲

- 対象実験: `exp233_adaptive_outlier_mixture_likelihood_pf`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 比較実験: `exp232_adaptive_robust_likelihood_pf`、exp214 public raw PF diagnostic。
- 変更する変数: gated particle observation likelihood の `epsilon` のみ。
- 固定する変数: raw GR/typewell GR、gate、500 particles、128 seeds、surface
  transition、resampling threshold/noise、seed aggregation、score rows、interval
  sampling、hidden-like split。
- 除外する変数: temperature、global mixture、process noise、step prior、particle
  reinjection、Beam、ML、inference、submission。

## 再現性設計

- seed policy: well id、variant name、`public_likpf`、seed index から stable SHA256
  seed base を作る。variant 名が異なるため trajectories は独立だが再実行可能である。
- stochastic 処理: PF propagation と systematic resampling。Numba kernel 内で
  per-well / per-variant seed を固定し、`num_workers=1` とする。
- CPU/GPU: CPU-only、GPU/internet disabled。
- 入力/出力 SHA: exp072 cache、exp115 split、exp232 comparison artifacts、row
  candidate gzip の raw/decompressed SHA、各 metrics CSV SHA を記録する。
- Kaggle bootstrap: package 生成後に bootstrap 内 config、kernel source、CPU、
  internet、seed 設定を照合する。

## リスク

- リーク: tail true TVT / error を gate、likelihood、Uniform support に混ぜること。
  support は steering で事前固定し、kernel に target列を渡さない。
- 評価: user-approved parallel train では exp232 artifacts がまだない exp233 run を
  許容するが、summary に `pending_exp232_artifacts` を記録する。exp232 artifact を
  結合した同一 id 比較が終わるまで、mixture variant は採用しない。
- runtime: full 773 wells x 2 mixture variants x 128 seeds は重い。exp072 control
  は再生成せず、interval weighted-quantile は固定 64 wells、64 row stride と gate
  rowに限定する。
- 再現性: variant ごとに resampling 回数が変わる。kernel version と artifact SHA を
  記録し、deterministic submission anchor とは扱わない。
