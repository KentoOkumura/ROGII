# exp233_adaptive_outlier_mixture_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: `kaggle_train_variant_split_prepared`
- CV: train-side exp072-compatible pseudo-tail audit を予定
- Public LB: なし（推論・提出は範囲外）
- Private LB: なし

## 仮説

GR innovation が高く、change point / novelty / pre-update particle collapse の裏付けが
ある row だけ、Gaussian observation likelihood に state-neutral Uniform-GR component
を少量混ぜる。これにより誤った局所 GR motif が全粒子を即座に一つの誤った state に
集中させることを抑え、particle interval coverage と long-tail の頑健性を改善できる。

## 検証方針

- 親は `exp072_exp063_full_replay_feature_cache`。exp209 enriched cache の
  `hmm_mean_tvt - hmm_minus_likpf_mean` から復元した `likpf_mean` を Gaussian
  control とし、再生成しない。row id / well / target / last_known_tvt / md_since を
  one-to-one で照合する。
- `exp232_adaptive_robust_likelihood_pf` と同じ target-free gate、500 particles、128
  seeds、transition、resampling を固定する。
- gate 内だけ `L=(1-epsilon)L_gaussian+epsilon L_uniform` を適用する。`L_uniform` は
  GR support `[0,500]` の一様密度で、row 内の粒子状態に依存しない。
- active variants は `mix_eps_0p02` と `mix_eps_0p05` のみ。temperature、global
  mixture、broad Gaussian component、control 再学習は含めない。
- overall、distance bucket、hidden-like、by-well、ESS、resampling、gate/mixture rate、
  p05-p95 coverage、first-loss を保存する。
- exp232 artifacts がそろう前の並行初回 run は comparison pending と明示する。採用には
  exp232 temperature variants との id-aligned comparison が必須である。

## 所見

Kaggle CPU train v1 は exp072 ML cache に `likpf_mean` が含まれず、PF generation 前に
ERROR となった。ユーザー承認により exp209 reconstructed control へ切り替え、v2 を
同じ canonical kernel id で実行する。temperature comparison は exp232 artifact 待ちの
ため、v2 output も comparison 完了前には採用しない。

## 次のアクション

Kaggle CPU train v2 は timeout 報告後に `CANCEL_ACKNOWLEDGED` となった。ユーザー承認済みの
再実行は variant 別であり、各 run は target well を分割せず、全 eligible well と同一 PF
設定を使う。

- `mix_eps_0p02`: `train_variant0` / `kentookumura/exp233-outlier-mixture-pf-e02`
- `mix_eps_0p05`: `train_variant1` / `kentookumura/exp233-outlier-mixture-pf-e05`

両 output は kernel ごとの標準 artifact 名で保存されるため、取得後に同一 directory へ
上書きせず、id / well / row_idx の整合を確認してから比較する。exp232 temperature output が
未接続の間は採用しない。

## 参照ファイル

- `config.yaml`
- `adaptive_outlier_mixture_likelihood_pf.py`
- `exp233_adaptive_outlier_mixture_likelihood_pf_train.py`
- `exp233_adaptive_outlier_mixture_likelihood_pf_train_variant0.py`
- `exp233_adaptive_outlier_mixture_likelihood_pf_train_variant1.py`
- `SESSION_NOTES.md`
