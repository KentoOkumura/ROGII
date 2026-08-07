# 要件

## 依頼

`exact_hmm_smoother_exp072_compatible_cache_audit` backlog を実験化し、amerhu 公開 notebook `amerhu/rogii-wellbore-geology-exact-hmm-smoother` の exact HMM smoother を exp072 と比較可能な train feature cache として監査する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 初回は train-only feature cache audit とし、test feature、model training、inference、submit は作らない。
- HMM default は公開 notebook の `step=0.35`、`n_rates=41`、`rate_span=0.10`、`sig_r=0.002`、`sig_p=0.02`、`mom=0.998`、`lam=1.0` を保つ。
- 公開 notebook の `_hmm2_fb` / `run_hmm2` の数式を大きくリライトせず、repo 用の path 解決、row/id 保存、summary 出力、exp072 direct comparison だけを足す。
- exp072 v2 deterministic cache を固定比較基準にし、`likpf_mean`、`pf_ancc`、`beam_mean` などと HMM / fixed blend を同じ rows で比較する。
- GPU 学習コストはなし。LightGBM config 数 0、fold 数 0、booster 数 0。

## 受け入れ基準

- `experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/` に train notebook、not applicable inference notebook、config、HMM generator、comparison helper、記録ファイルがある。
- train cache は `id`、`well`、`target`、`last_known_tvt`、`md_since`、`hmm_mean_tvt`、`hmm_std`、`hmm_loglik` を含み、exp072 cache と `id` set/order を監査できる。
- direct comparison は HMM単体、exp072 `likpf_mean` 単体、fixed blend (`w025/w050/w075`)、distance bucket、by-well improved/worsened、worst-well regression、`hmm_std` calibration を出力する。
- `SESSION_NOTES.md` に CPU train-only、variant 数、booster 数、control 再学習なし、推論/提出なしを記録する。
- deterministic anchor としては扱わず、feature content SHA と Kaggle kernel version を記録対象にする。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
