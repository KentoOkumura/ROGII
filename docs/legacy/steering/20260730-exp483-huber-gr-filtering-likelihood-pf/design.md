# 設計

## 1. 検証する問い

exp404のGaussian粒子尤度

```text
z = (GR_observed - GR_typewell(TVT_particle)) / sigma_GR
log L_gauss = -0.5 * min(z^2, 600)
```

だけを、exp389と同じfixed Huber scoreへ置換する。

```text
delta = 1.345
rho(z) = 0.5*z^2                              if abs(z) <= delta
         delta*abs(z) - 0.5*delta^2           otherwise
log L_huber = -rho(z)
```

stateに依存しない正規化定数は省略し、追加clipは行わない。これによりoutlier GR
の影響を抑えつつ、粒子weight、ESS、resampling、将来軌道まで変わるかを検証する。
凍結軌道のseed evidenceだけを変えたexp430とは別の問いである。

## 2. 固定するPF契約

- 500 particles、128 stable seeds、initial position spread `4.5 ft`、
  initial rate spread `0.01`。
- momentum `0.998`、rate noise `0.002`、position noise `0.005`。
- ESS threshold `0.5`、rough position/rate `0.1 / 0.001`。
- exp404 x1.0のknown-prefix zero-fill population std、clip `[10,60]`。
- missing GR補間、Type Well step/pad、float32 output、seed bankを固定する。
- primary readoutはtemperature `5.0`のlikelihood seed aggregationだけ。
  arithmetic meanはparity用secondaryで、昇格判定には使わない。

## 3. 段階設計と実行量

- Stage 0: stable-hash fixed32、candidate 32 PF well-runs、
  `4,096` seed-well trajectories、`2,048,000` particle starts。
- Stage 1: 別承認時のみ773 candidate PF well-runs、
  `98,944` seed-well trajectories、`49,472,000` particle starts。
- 保存exp404 control rerun、HMM、Beam、model、LightGBM config、booster、GPUは0。

Stage 0はformula、finite coverage、seed identity、ESS/resampling ledger、
truth-late、artifact SHA、runtime/RSSを検査するtechnical preflightであり、CVではない。

## 4. Stage 1 promotion gate

全条件を満たす場合だけ後続候補とする。

- exp404 scale-5 x1.0 RMSE `10.914522073`から`0.05 ft`以上改善。
- 4/5 folds以上で改善。
- raw-GR observedで`0.05 ft`以上改善。
- raw-GR missing、高missing wells、1000+、hidden-like spatial、
  hidden-like typewell-purgedの各scopeでregression `<=0.0 ft`。
- by-well delta p95 `<=0.0 ft`、worst-well regression `<=0.25 ft`。
- exp209 HMMとの固定50:50 blendが保存control `10.084909680`より非悪化。

一つでもFAILならbranchを閉じ、delta、scale、temperature、Gaussian mixture、
clip、particle/seed、well/row gate、blend/selectorで救済しない。

## 5. 再現性とleakage

- seed baseは`sha256_first16("likpf::train::<well_id>")`、
  seed indexは`0..127`。variant名をseedへ入れない。
- well、row、seed、particle、reduction順を固定し、各kernelへ明示seedを渡す。
- raw train/testは別生成し、入力、scientific contract、well audit、
  prediction schema、logical content、decompressed contentのSHAを保存する。
- predictionと全SHAをfreezeするまでsuffix truth、error、fold、
  hidden-like roleを読まない。
- 初回runをdeterministic anchorとせず、独立rerunでSHA一致した場合だけanchor化する。

## 6. 承認境界

2026-07-30の追加依頼でcompact self-contained Stage 0候補と専用testを実装し、
その後の`実行してください`で正規train Notebook採用、Kaggle package、
Stage 0 runを承認・完了した。さらに`Stage1に進んでください`で、同じ科学契約の
全773 wells Stage 1実装、canonical package、push/runを別承認済みとする。
inference、submissionは未承認である。
