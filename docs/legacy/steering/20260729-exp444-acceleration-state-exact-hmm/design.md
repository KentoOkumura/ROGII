# 設計

## 1. 状態と更新順

exp441の`(TVT_t,r_t)`へ`a_t`を追加する。

```text
a_t in {-0.0005, 0, +0.0005} rate/MD-ft
P(a_t | a_{t-1}) = [0.08, 0.84, 0.08]  # adjacent/stay/adjacent
mu_rate = exp(-kappa*h)*r_{t-1} + a_t*h
Var(rate) = exp441 exact OU variance
delta_TVT = r_t*h-delta_Z
```

acceleration端の外向き0.08は端stayへ加える。initial priorは
`P(a_0=0)=1`。更新順はacceleration、rate、TVT、current GR emissionである。
同じGRを二重利用せず、joint stateをforward/backwardで同時推論する。

## 2. パラメータ固定

3状態はstate explosionを最小限にする。`±0.0005`は10行継続時にnominal
0.005 rate bin 1個相当を動かす事前固定幅である。`0.08/0.84/0.08`は
exp209のnominal 1-ft、`sig_r=0.002`、rate step 0.005から得る方向move確率を
転用する。grid探索はしない。

exp441のOU kernel、rate grid、position kernel、GR emission、prior、readoutを固定し、
科学差分をacceleration state追加だけにする。
one-factor controlは保存済みexp441 predictionとし、control HMMは再実行しない。
保存exp209 predictionはroot referenceとしてのみ報告する。exp441のfixed32/full
prediction SHAは、それぞれStage 0/1の実装・実行前に厳密固定する。ただし
exp441はStage 0でterminal closeしておりfull predictionは存在しないため、
Stage 1でexp441を再実行せず、保存exp209をpromotion比較対象にする。

## 3. 独立仮説契約

当初はexp441がtechnical PASS、matched-control safeで、persistent lagの方向は
正しいが時間的持続不足を残し、exp442が不足またはunsafeの場合だけ実装する条件を
置いた。実際のexp441はcontrol-safeだった一方、runtimeとpersistent mechanismで
FAILし、当初条件は成立しなかった。

2026-07-30のユーザー判断により、この先行条件を撤回する。exp444は、
「full-support OUだけでは不足したが、明示的なtrend-memory stateを加えれば
persistent lagを回復できるか」という独立した組合せ仮説として評価する。
exp441は構造上の一要因controlとnegative contextに限定し、positive evidence、
実行前提、parameter/gate変更には使わない。exp442の結果も実行前提にしない。
acceleration値、transition、initial prior、fixed4/fixed32、全gateは変更しない。

## 4. Stage 0A

fixed32のwell identityだけを固定SHAで並べた4 wellsを使う。role/truth/fold/episodeを
読まず、acceleration row sum、zero-acceleration exp441 kernel parity、
posterior/brute-force parity、runtime、RSSだけを判定する。

fixed32換算`<=3,600 sec`、full換算`<=30,600 sec`、RSS`<=25 GB`を超えたら、
state数、span、transition、実装方式を救済せず閉じる。

## 5. Stage 0B

Stage 0A出力を再利用し、追加28 wellsでfixed32 total 32にする。

- posterior nonzero acceleration mass fraction `0.01--0.80`。
- posterior accelerationとfuture true rate curvature sign一致`>=0.60`、
  positive`>=4/5 folds`。
- forward-cause SSE`>=10%`、persistent SSE`>=5%`削減。
- persistent改善`>=10/16 wells`、`>=4/5 folds`。
- exp441比forward/persistent改善を報告する。
- 保存exp209 control pooled`<=+0.02 ft`、p95`<=+0.25 ft`。

prediction/posterior/diagnostic SHA freeze後だけtruth-late評価する。

## 6. Stage 1・再現性・実行量

Stage 0A/0B全PASS・別承認時のみ773 wells。保存exp209比direct RMSE gain
`>=0.05 ft`、
4/5 folds、固定scope、by-well tailをAND判定する。

RNGなし。well/row/position/rate/acceleration/reduction順固定。kernel、posterior、
prediction、metrics SHAを保存する。candidate HMM runは4、fixed32 total 32、
full 773。parent rerun 0、ML/PF/Beam/GPU 0。初回runはdeterministic anchorとしない。

## 7. 実装フェーズ

2026-07-30のユーザー判断により、次だけを実装する。

- compact self-contained Jupytext train/inference候補。
- 3状態acceleration transition、acceleration-conditioned exact OU kernel。
- factorized exact forward/backward、zero-acceleration exp441 parity、
  small-state dense brute-force contract。
- identity-only SHA順のfixed4 selector、target-free prediction/posterior/diagnostic
  freeze、runtime/RSS projection gate。
- 専用test、Jupytext変換、構文、Ruff、strict experiment/template validation。

正規Notebook採用、Kaggle package、Stage 0Aは2026-07-30の別指示で承認され、
private CPU version 1を実行した。4 wells / 21,962 rowsを完走し、
exactness、normalization、leakage、RSSはPASSしたが、fixed32/full runtime投影
`5,970.829552 / 144,232.851372 sec`が上限`3,600 / 30,600 sec`をFAILした。
Stage 0B/1、inference、submission、同一branch内の救済なしでterminal closeする。
