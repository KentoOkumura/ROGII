# 設計

## アプローチ

exp072 likelihood-PF の各粒子で、標準化 GR residual を `r` とする。通常 row は既存どおり `L=exp(-0.5*r^2)` を使う。high innovation と少なくとも一つの裏付け signal がある row だけ、variant ごとに `L=exp(-0.5*r^2/T)` とし、`T=2/4` を比較する。

裏付け signal は raw GR change-point、short/long GR novelty、pre-update ESS ratio、pre-update max particle weight である。gate 外では temperature は必ず 1 で、transition、seed aggregation、resampling は exp072 と同一にする。

## 実験範囲

- 対象実験: `exp232_adaptive_robust_likelihood_pf`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 補助根拠: `exp214_public_raw_gr_residual_scale_control` の seed-scale 診断、exp200 の global prior long-tail regression、exp209 の enriched row-level comparison cache。
- 変更する変数: gated particle observation temperature (`T=2/4`) のみ。
- 固定する変数: 500 particles、128 seeds、raw GR/typewell GR、known-prefix sigma、surface transition、ESS threshold、resampling noise、seed aggregation、score rows。
- 除外する変数: outlier mixture、global temperature、process noise、step prior、particle reinjection、Beam、ML、推論、提出。

## 再現性設計

- seed policy: well id、variant name、`public_likpf`、seed index から SHA256 の stable seed base を作る。
- stochastic 処理: PF propagation と systematic resampling。Numba kernel 内で well/variant ごとに `np.random.seed` を固定する。
- 並列処理: `num_workers=1`。thread scheduling に乱数系列を依存させない。
- CPU/GPU: CPU-only、GPU/internet disabled。
- 入力/出力 SHA: exp072 cache と exp115 split、row candidate gzip の raw/decompressed SHA、すべての metric CSV SHA を保存する。
- T=1 control: exp209 cache の `hmm_mean_tvt - hmm_minus_likpf_mean` で復元し、ID、well、target、last known TVT、`md_since` の一対一整合を確認する。復元値は比較だけに使い、gate/PF update へ渡さない。
- timeout recovery: Kaggle CPU の12時間上限に対し、同じ exp232 内で `temp_t2` と `temp_t4` を別 kernel version に分割する。各 run は同じ full validation surface・同じ stable seed・同じ control を使い、科学的変数は変えない。
- checkpoint: Kaggle cancellation 後の `/kaggle/working` は永続化されないため、split run では cross-run resume を前提にしない。16 wells ごとの progress flush と、完走した kernel の最終 artifact を正とする。
- Kaggle bootstrap: package 生成後に bootstrap 内の config、kernel source、CPU/internet/seed設定を照合する。

## リスク

- リーク: tail true TVT を gate/likelihood に混ぜること。gate input と scoring input を分離し、target列を kernel に渡さない。
- CV/LB: 本実験は train-side pseudo-tail であり、raw-test transfer を保証しない。inference は別判断にする。
- runtime: 全 773 wells × 2 variants × 128 seeds は12時間を超えたため、controlを再生成せず variant ごとに分割する。interval weighted-quantile は固定64 wells・64 row strideとgate rowに限定する。
- control provenance: exp209 cache の exp072 v2 full artifact exact parity は未証明であるため、v2 の control は同一比較 cache 内の T=1 として解釈する。
- 再現性: resampling回数がtemperatureで変わるため、variantごとの乱数列は独立だが stableにする。実行環境差は SHA と kernel version で監査する。
