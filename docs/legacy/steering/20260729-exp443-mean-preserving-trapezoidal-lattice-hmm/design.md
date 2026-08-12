# 設計

## 1. Joint edge

exp209のlegal rate edge`i -> j`と`h=delta_MD`について、

```text
mu_ij = 0.5*(r_i+r_j)*h-delta_Z
x_lo = floor(mu_ij/0.35)*0.35
x_hi = x_lo+0.35
v_min = (mu_ij-x_lo)*(x_hi-mu_ij)
v_parent = max(sig_p,0.35*0.35)^2
v_eff = max(v_parent,v_min)
```

とする。`mu_ij`周辺の固定5 lattice offsetsへ非負weightを置き、
sum 1、mean `mu_ij`、variance `v_eff`を満たすmaximum-entropy解を使う。
同じjoint edge tableをforward/backwardで共有する。

## 2. exp439からの独立性

exp439は`v_parent`を常に厳密保存する契約だった。実データ最初のedgeで
`v_min=0.0264 > v_parent=0.01500625 ft²`となり、非負解が存在しなかった。

exp443はsolver/support救済ではなく、格子量子化が強制するvariance floorを
科学モデルの一部として採用する別仮説である。全edgeの`v_eff-v_parent`分布を保存し、
隠れたnoise変更として扱わない。

## 3. 固定条件

- exp209 adjacent rate marginal、rate boundary mass。
- position/rate grid、band、`sig_r`、`sig_p`、momentum。
- prior、Gaussian GR emission、missing処理、readout。
- 保存exp209 control、parent HMM rerun 0。
- trigger、reset、re-anchor、selector、blendなし。

## 4. Stage 0

fixed32の1候補×32 wellsをtruth-lateで評価する。

Technical:

- rate marginal parity、edge sum`<=1e-12`。
- mean/`v_eff`誤差`<=1e-10`、negative weight 0。
- posterior/brute-force差`<=1e-6`。
- signed one-step grid mean biasを95%以上削減。
- finite 1.0、pre-freeze truth read 0、runtime/RSS guard。

Mechanism:

- forward-cause SSE`>=10%`、persistent SSE`>=5%`削減。
- persistent改善`>=10/16 wells`、`>=4/5 folds`。
- control pooled`<=+0.02 ft`、p95`<=+0.25 ft`。

一つでもFAILならgrid/support/variance/noise/rate/emission/gateを救済しない。

## 5. Stage 1・再現性・実行量

全PASS・別承認時のみ773 wells。direct gain`>=0.05 ft`、4/5 folds、固定scope、
by-well tailをAND判定する。

RNGなし。solver/edge/reduction順固定。joint edge、variance audit、prediction、
metrics SHAを保存する。HMM runsはStage 0/1=`32/773`、parent rerun 0。
ML/PF/Beam/GPUは0。初回runはdeterministic anchorとしない。

## 6. 実装状態

2026-07-29のユーザー依頼により、exp439 compact実装を親構成として
`exp443_*_compact_selfcontained_train.py/.ipynb`、fail-closed inference guard、
専用pytestを実装する。科学差分は次に限定する。

- supportは固定5 cells。
- edgeごとに`v_min`と`v_eff`を事前計算する。
- joint-edge SHAへ`v_min` / `v_eff` / inflationを含める。
- truth-late前にvariance-floor auditとprediction SHAを保存する。
- exp439 failure edge `0.0264 > 0.01500625 ft²`をpositive contract testにする。

2026-07-30の実行承認後はcompact trainを正規train Notebookへ採用し、
canonical Kaggle CPU kernelでStage 0 fixed32を実行する。inference guardは維持し、
Stage 1、inference、submissionへは進まない。

## 7. Stage 0結果

Kaggle version 1で32/32 wellsを完走した。mean、effective variance、
nonnegative fixed-five support、rate marginal、brute-force、truth-late、SHAは
成立し、one-step grid mean biasはほぼ完全に削減した。一方、Stage 1 runtime投影は
`125406.237 > 30600 sec`でFAILした。

Mechanismはpersistent改善well `10/16`とfold `4/5`だけPASSし、forward-cause SSE
削減`5.517%`、persistent SSE削減`-5.766%`、control pooled RMSE delta
`+0.093698 ft`、control p95 `+1.394368 ft`はFAILした。事前設計どおり
`stage0_fail_closed`とし、同一fixed32救済、Stage 1、inference、submissionを行わない。
