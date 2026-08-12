# 要件

## 依頼

`adaptive_outlier_mixture_likelihood_pf` を、`exp232` の temperature 方式とは
独立した likelihood-PF 実験として実装する。GR 観測が局所的に信用しにくい
row だけ、Gaussian 観測尤度に state-neutral な outlier mixture を加え、
誤った局所 GR motif による particle collapse を抑える。

## 制約

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`。保存済み `likpf_mean` を
  Gaussian control とし、再生成しない。
- `exp232_adaptive_robust_likelihood_pf` と同じ target-free gate、500 particles、
  128 seeds、transition、resampling、seed mean aggregation を固定する。
- Gate 外は既存と同じ `L_gaussian=exp(-0.5*r^2)` を**厳密に**使う。
- Gate 内だけ、正規化 Gaussian と固定 GR support の Uniform を同じ相対尺度に
  揃えた `L=(1-epsilon)L_gaussian+epsilon L_uniform` を使う。広い Gaussian
  component は粒子状態を選好するため使用しない。
- Gate は raw evaluation GR、typewell GR、observed prefix、pre-update particle
  stateのみを使う。true TVT、target、error、oracle、Public LB、seed-level
  aggregation weight は使用しない。
- CPU-only Kaggle train-side audit とし、inference、submission、GPU、LightGBM、
  control/parent の再学習は範囲外とする。
- `exp232` が完了して必要 artifacts を公開するまでは、exp233 の Kaggle 実行と
  採用判定を行わない。並行して行うのは実装・静的検証までである。

## 固定した設計値

ユーザー承認により、train の finite raw GR `13.8802..487.0329` を覆う Uniform
support `[0, 500]` と、`epsilon=[0.02, 0.05]` の 2 variant を固定した。

## 受け入れ基準

- 773 well の exp072-compatible pseudo-tail で、保存済み exp072 control、完了済み
  exp232 temperature artifacts、mixture variants を同じ id で比較できる。
- overall、distance bucket、hidden-like、by-well、ESS、resampling、gate rate、
  mixture rate、sampled particle p05-p95 coverage、first sampled loss を保存する。
- mixture rate が gate rate と一致し、gate 外では Gaussian baseline update が正確に
  実行されたことを config、kernel、diagnostics で確認できる。
- `epsilon`、Uniform support、component 種別の不正値と、観測 GR が support 外の
  場合は fail-fast する。
- coverage は exp232 と直接比較する。exp072 に粒子 interval artifact がないため、
  exp072 に対する coverage 差は control 再生成なしには主張しない。
- direct replacement、inference、submission は RMSE、worst-well、1000+、
  hidden-like、temperature 比較の guard を満たし、別途承認されるまで作成しない。
