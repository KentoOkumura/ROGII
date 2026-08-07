# 設計

## 1. 介入

exp209 の状態`(TVT_t, r_t)`、41 rate grid、0.35 ft TVT gridを維持する。
現行の「rate marginalization後、destination rateだけでpositionを動かす」2段処理を廃止し、
source/destination rateを両方含む1つのjoint edgeに置き換える。

各行の`h = delta_MD`、`delta_Z`、legal rate edge`i -> j`について、

```text
K_joint(delta_TVT, r_j | r_i)
  = K_rate_exp209(r_j | r_i)
    * K_position(delta_TVT | r_i, r_j)

mu_ij = 0.5 * (r_i + r_j) * h - delta_Z
Var(delta_TVT | i, j) = max(sig_p, 0.35 * position_step)^2
```

とする。`K_rate_exp209`は`sig_r=0.002`、`mom=0.998`、隣接3状態、rate-grid端の
sub-stochastic massを含めて固定する。rate端を再正規化して救済しない。

## 2. Moment-preserving lattice projection

各`(t, i, j)` edgeで、`mu_ij`周辺のTVT lattice offsetへ非負重みを置く。
重みは次の3制約を同時に満たす。

```text
sum_n w_n = 1
sum_n w_n * x_n = mu_ij
sum_n w_n * (x_n - mu_ij)^2 = sigma_position^2
```

解は有限support上のmaximum-entropy分布として一意化する。supportは現行と同じ5セルから
開始し、非負解がなければ7、9セルへだけ拡張する。tie、収束、許容誤差を固定し、
9セルで不成立ならcandidate全体をfail-closeする。

条件付き平均と分散をedgeごとに保存するため、source rateごとのmixtureでも

```text
E[delta_TVT]
Var(delta_TVT)
Cov(delta_TVT, delta_r)
```

が連続edge mixtureと一致する。これが「rate遷移後に独立position kernelを適用する」
現行処理との科学差分である。

## 3. 固定するもの

- 親: exp209。
- state: 持続`(TVT, U-rate)`。
- TVT step/band、41 rate support/rate step、`sig_r`、`sig_p`、`mom`。
- initial TVT/rate prior、prefix calibration、GR emission、likelihood scale。
- posterior mean readout、forward/backward smoother、dtype方針。
- exp209 rate-grid端とTVT-band端のmass truncation semantics。
- controlは保存済みexp209 prediction。control HMMを再実行しない。

変更しない対象は「parityで確認する固定条件」であり、同じfixed32での救済対象にしない。

## 4. Stage 0: fixed32 mechanism preflight

exp411の固定manifestを使い、persistent 16 / matched-control 16 の計32 wellsで
joint candidateを1回ずつ実行する。fixed32はmechanism-onlyでありCV/promotion evidence
とは呼ばない。suffix truth、role、fold、episode causeはtransition、prediction、
diagnostic、SHAをfreezeした後だけjoinする。

### Technical AND gate

- wells / suffix rows / folds=`32 / 156,088 / 5`。
- rate marginal max absolute difference vs exp209`<=1e-12`。
- legal edgeのweight sum error`<=1e-12`。
- conditional mean error`<=1e-10 ft`。
- conditional variance error`<=1e-10 ft^2`。
- source-row joint covariance error`<=1e-10 ft^2/ft`。
- forward/backwardが同じjoint edge tableを参照する。
- 小規模brute-force HMMのposterior/prediction差`<=1e-6`。
- posterior normalization error`<=1e-6`。
- exp209で測定した1行grid biasに対し、同一edge集合の平均bias絶対値を95%以上削減。
- truth / role / fold / episode reads before freeze=`0`。
- full 773-well runtime projection`<=30,600 sec`、peak RSS`<=25 GB`。

### Mechanism AND gate

- forward-transition/prior-hysteresis cause episode SSE reduction`>=10%`。
- persistent episode SSE reduction`>=5%`。
- persistent improved wells`>=10/16`。
- persistent improving folds`>=4/5`。
- matched-control pooled RMSE delta`<=+0.02 ft`。
- matched-control by-well delta p95`<=+0.25 ft`。

1条件でもFAILならStage 1へ進まず、support、moment solver、noise、grid、rate、
emission、prior、readout、gateを同一実験内で変更しない。

## 5. Stage 1: full OOF

Stage 0の全technical/mechanism gate PASSと別承認後だけ、同じ1 variantを773 wellsで実行する。
保存済みexp209 predictionとtruth-lateで比較し、次をAND判定する。

- pooled RMSE gain vs exp209`>=0.05 ft`。
- improving folds`>=4/5`。
- forward-cause SSE reduction`>=10%`。
- persistent SSE reduction`>=5%`。
- near 0--250、mid 250--1000、1000+、hidden-like spatial、
  hidden-like typewell-purgedの各delta`<=+0.02 ft`。
- by-well RMSE delta p95`<=+0.25 ft`、worst`<=+2.0 ft`。

PASSしてもblend、selector、inference、submissionは別設計・別承認とする。

## 6. 再現性

- RNGなし。well、row、source rate、destination rate、position offset順を固定する。
- moment solverはfloat64、固定初期値、固定反復上限、固定収束許容誤差でprecomputeする。
- HMM message dtypeとreduction順は親に合わせる。
- raw input、fixed32 manifest、rate kernel、joint edge table、moment audit、
  prediction、diagnostic、metricsのschema/logical content SHAを保存する。
- gzipはdecompressed content SHAを主証拠とする。
- 最初のrunはdeterministic anchorとせず、同一設定rerunのSHA一致後に再判定する。

## 7. 実装承認後の確定範囲

- scientific variant: 1。
- Stage 0 candidate HMM well-runs: 32、parent rerun: 0。
- Stage 1: 未承認、最大773 candidate well-runs。
- fitted ML model / LightGBM config / trained fold / booster=`0 / 0 / 0 / 0`。
- PF / Beam / GPU=`0 / 0 / 0`。
- compact self-contained train/inference候補、contract test、候補Notebookを実装する。
- 正規notebook採用、package、run、Stage 1、inference実行、submissionは0のまま保持する。
- 実装は2026-07-29のユーザー依頼で承認済み。Stage 0実行は別承認を要する。
