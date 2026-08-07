# 設計

## 1. 科学差分

exp404のparticle z-scoreは維持し、log emissionだけをexp374の式へ変更する。

```text
df = 4
z = (GR_observed - GR_typewell(TVT_particle)) / sigma_GR
log L_student = -0.5 * (df + 1) * log1p(z^2 / df)
```

state-independentな正規化定数は省略し、追加clipはしない。これ以外の初期化、
dynamics、ESS resampling、roughening、missing GR、500 particles、128 seeds、
x1.0 sigma、temperature-5 seed aggregationはexp404と同一にする。

## 2. 段階と実行量

- Stage 0: stable-hash fixed32、32 PF well-runs、4,096 seed-well、
  2,048,000 particle starts。technical preflightでありCVではない。
- Stage 1: Stage 0全PASS・別承認時だけ773 PF well-runs、98,944 seed-well、
  49,472,000 particle starts。
- exp404 control rerun、HMM、Beam、model、booster、GPUは0。

## 3. Gate

Stage 0ではformula、finite coverage、stable seed、ESS/resampling ledger、
truth-late、prediction/content SHA、runtime/RSSをAND評価する。

Stage 1では次を全ANDとする。

- exp404 scale-5 x1.0 `10.914522073`から`0.05 ft`以上改善、4/5 folds以上。
- raw observed `0.05 ft`以上改善。
- raw missing、高missing、1000+、hidden-like 2面のregression `<=0.0 ft`。
- by-well delta p95 `<=0.0 ft`、worst `<=0.25 ft`。
- exp209 HMMとの固定50:50 blendが`10.084909680`より非悪化。

FAIL時はdf、scale、temperature、clip、mixture、particle/seed、gate、
blend/selectorで救済せず閉じる。

## 4. 再現性と承認境界

- exp404のstable per-well SHA256 seedを継承し、variant名はseedから除外する。
- raw train/testを別生成し、well/row/seed/particle/reduction順を固定する。
- prediction、contract、audit、schema、logical/decompressed content SHAを
  freezeした後だけtruth、fold、roleをattachする。
- 初回runはdeterministic anchorにしない。
- Stage 0はKaggle CPU kernel version 2で16/16 technical gate PASS。
- 2026-07-30の追加依頼でStage 1 package / push / runを別承認済み。
- Stage 1でもcandidate prediction、audit、schema、content SHAを全773 wellsで
  freezeした後だけ、suffix truth、保存exp404 control、exp226 reporting fold、
  exp115 hidden-like role、保存exp209 HMMを読む。
- Stage 1は同じcanonical Kaggle CPU kernel version 3で完了。technical
  `18/18 PASS`、candidate/control `10.897096923 / 10.914521913 ft`、
  改善`+0.017424990 ft`、2/5 folds、by-well p95`+1.455066656 ft`、
  worst`+16.664889733 ft`で科学gateをFAILした。
- `terminal_close_without_student_t_or_pf_rescue`を適用し、inference、
  submissionへ進まない。
