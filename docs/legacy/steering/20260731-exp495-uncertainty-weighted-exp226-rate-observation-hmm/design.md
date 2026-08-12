# 設計

## アプローチ

exp209のpersistent rate stateを残し、exp226 geometryが示すU-rateを
rate transitionの外部観測としてsoftに融合する。

exp226 geometry U-rateを

```text
r_geom[t] = Δ(tvt_geop[t] + Z[t]) / ΔMD[t]
```

とする。exp355と同じくknown-prefix初期rateへ相対変化だけを足し、観測中心を

```text
mu_226[t] = r_prefix_exp209 + r_geom[t] - r_geom[first_segment]
```

に固定する。exp226 final`tvt_pred`、GR correction、U projectionは使わない。

各wellの最後の128個のfinite・positive-ΔMD known-prefix transitionで、fold-safe
geometry rateと既知U-rateの残差を計算する。残差中央値を中心化し、

```text
sigma_226[w] = max(0.002, 1.4826 * MAD(prefix_rate_residual))
```

とする。valid transitionが32未満なら観測因子を無効化し、exp209 parent parityへ
fallbackする。bias補正はせず、不確実性だけを既知prefixから推定する。

exp209のrate transitionを`P209(j|i)`、rate state `j`が表すabsolute U-rateを
`r_j`とすると、唯一の科学的変更は次である。

```text
L226_t(j) = exp(-0.5 * ((r_j - mu_226[t]) / sigma_226[w]) ** 2)
P495_t(j|i) = P209(j|i) * L226_t(j)
               / sum_k(P209(k|i) * L226_t(k))
```

位置遷移は親の式を維持する。

```text
ΔTVT = r_j * ΔMD - ΔZ
```

`L226=1`またはprefix fallback時は、rate transitionから最終predictionまで
exp209と数値parityを要求する。追加のlambda、temperature、clip、scale、
activation thresholdは持たない。

## 実験範囲

- 対象実験: `exp495_uncertainty_weighted_exp226_rate_observation_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- rate schedule evidence: `exp355_exp226_dip_rate_prior_on_exp209`
- mechanism evidence: `exp408_hmm_message_rate_basin_audit`
- negative references:
  - `exp411_predictive_filtered_rate_innovation_destick`
  - `exp491_exp226_final_tvt_rate_direct_hmm`
- 変更する変数: exp209 rate transitionへ掛けるwell-level uncertainty付き
  exp226 geometry-rate Gaussian観測因子1個。
- 固定する変数: exp209 state/grid/transition base/momentum/diffusion/position kernel/
  GR emission/prior/forward-backward/readout、およびexp355 geometry schedule中心。

## 段階設計

### Stage 0A: reliability identifiability、0 HMM

保存済みfold-safe exp226 geometryとraw known prefixから`mu_226`、`sigma_226`、
fallbackを全773 wellsで生成・SHA freezeする。その後だけsuffix truthとfoldを結合し、
`sigma_226`がsuffix geometry-rate誤差を順位付けできるか確認する。

実行量:

- diagnostic variant: 1
- HMM / PF / Beam / model / trained fold / booster / GPU: 0
- parent/control再実行: 0

technical gate:

- 773 wells、3,783,989 suffix rows、5 folds、重複・欠損0
- exp226 / exp209 / fold manifest SHA一致
- known-prefix-only confidence、suffix truth read before freeze 0
- finite `mu_226` / `sigma_226` coverage 1.0、fallback理由を全件記録
- rate formula / K16 segment identity / stable order / logical SHA一致

mechanism gateは全ANDとする。

- `Spearman(sigma_226, suffix absolute geometry-rate error) >= 0.20`
- 上記Spearmanが5 folds中4 folds以上で正
- low-sigma halfのrate RMSEがhigh-sigma halfより10%以上小さい
- low-sigma halfのexp355相対rate scheduleがexp209 constant prior比5%以上改善
- 上記schedule改善が5 folds中4 folds以上
- prefix fallback well率`<=0.05`

1件でもFAILならHMMを実装・実行せずexp495を閉じる。

### Stage 0B: fixed32 mechanism HMM

Stage 0A全PASSと別承認後だけ、exp411/491と同じ16 persistent + 16 matched-control
fixed32で候補1本を実行する。保存済みexp209とexp355を比較に使い、control HMMを
再実行しない。

実行量:

- scientific variant: 1
- candidate HMM well-runs: 32
- parent/control HMM rerun、PF、Beam、model、booster、GPU: 0

mechanism gateは全ANDとする。

- all32 RMSEが保存exp355より`>=0.10 ft`改善
- persistent RMSEが保存exp355より`>=0.10 ft`改善
- matched-control delta vs保存exp209が`<=+0.02 ft`
- 改善fold vs exp355が`>=4/5`
- persistent episode SSE reduction vs exp355が`>=0.10`
- paired by-well delta p95 vs exp355が`<=+0.25 ft`
- worst-well delta vs exp355が`<=+2.0 ft`
- projected full runtime`<=30,600 sec`、peak RSS`<=8 GiB`

1件でもFAILならscale / sigma / window / gate / emission / grid / blend / selector / PFで
救済せず閉じる。

### Stage 1: full group-safe OOF

Stage 0B全PASS、結果レビュー、別承認後だけ1 variant × 773 HMM well-runsを実行する。
保存exp209 / exp355をcontrolとし、再実行しない。primaryはexp355
`11.291976616 ft`から`>=0.05 ft`改善し、4/5 folds、1000+、hidden-like 2面、
persistent episode、by-well p95、worst-wellの固定AND gateをすべて満たすこととする。
Stage 1はrate-lag mechanism CVであり、inferenceやsubmissionへの自動昇格ではない。

## Assumption

known-prefixのrobust rate残差scaleが、unknown suffixでのexp226 geometry-rate
信頼度を順位付けできると仮定する。exp285はknown-prefix offsetからfull-suffix offsetを
予測できなかったため、この仮定は未検証であり、Stage 0Aを必須の停止点にする。

## 再現性設計

- seed policy: RNGなし。fold / well / segment / row / rate-state / reduction順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: fixed well orderで独立処理し、rate-state reduction順を親exp209と一致させる。
- runtime: Kaggle private CPU、GPU / internet off。
- input: exp209 / exp226 / fold / fixed32 / hidden-like manifestのfile・schema・content SHAを記録する。
- feature: prefix residual、`sigma_226`、`mu_226`、fallback、rate-observation factorの
  logical SHAをtruth join前にfreezeする。
- prediction: raw gzip SHAとdecompressed / logical SHAを分離し、後者を主証拠にする。
- model manifest / submission SHA: 学習modelとsubmissionがないため非該当。
- Kaggle bootstrap: package作成が別承認された場合、loose / embedded config parity、
  canonical id/title、CPU/internet off、kernel source、入力SHAをpush前に確認する。
- deterministic anchor: 初回成功runだけでは指定しない。

## リスク

- leakage: known prefix TVTは許可するが、suffix truth / error / role / fold outcomeを
  confidence・schedule・prediction freeze前に読むとリークになる。
- transfer: prefix rate residual scaleがsuffix donor mismatchを説明しない可能性がある。
- tail: exp355はpooled 5/5 foldsを改善した一方、hidden-likeとworst wellを大きく悪化させた。
- coordinate: exp226 final TVT-rateとU-rateを混同すると`ΔZ`を二重補正する。
- evidence reuse: exp226 final GR補正を使わずgeometry rateだけを使い、HMM GR emissionとの
  二重利用を避ける。
- runtime: full exact HMMは高コストなので0-HMM reliability gateとfixed32を先行する。
- CV/LB: direct physical routeはCV/LB順位が安定せず、Stage 1 PASSでも提出候補とは限らない。
