# exp443_mean_preserving_trapezoidal_lattice_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を完走した。台形平均とlattice variance floorの
数値contractは成立したが、runtime projectionとmechanism 4/6項目がFAILしたため、
`stage0_fail_closed`で終了した。Stage 1、inference、submissionへは進まない。

## 実行

- kernel:
  `kentookumura/exp443-mean-pres-trapezoid-lattice-hmm-train`
- version / id_no: `1 / 129095370`
- runtime: private CPU、internet無効、Numba thread 1
- variant / HMM well-runs / reporting folds: `1 / 32 / 5`
- parent HMM rerun / ML / booster / PF / Beam / GPU: すべて0
- fixed32はmechanism preflightであり、CVやpromotion evidenceではない

## 結果

| 項目 | 値 | gate |
| --- | ---: | --- |
| Stage 0 elapsed | 5,249.411秒（約87分29秒） | report |
| Candidate HMM | 5,191.461秒 | report |
| Stage 1投影 | 125,406.237秒（約34.84時間） | FAIL |
| Peak RSS | 1.235 GiB | PASS |
| conditional mean最大誤差 | 5.719e-13 ft | PASS |
| effective variance最大誤差 | 1.911e-13 ft² | PASS |
| rate marginal最大誤差 | 0.0 | PASS |
| posterior normalization最大誤差 | 3.331e-15 | PASS |
| one-step grid mean bias削減 | 99.9999999999% | PASS |
| variance-floor active edges | 9,665,508 | report |
| variance inflation mean / max | 0.003905 / 0.015619 ft² | PASS |
| Forward-cause episode SSE削減 | 5.517%（必要10%） | FAIL |
| Persistent episode SSE削減 | -5.766%（悪化、必要5%） | FAIL |
| Persistent improved wells | 10/16 | PASS |
| Persistent improving folds | 4/5 | PASS |
| Control pooled RMSE delta | +0.093698 ft | FAIL |
| Control by-well delta p95 | +1.394368 ft | FAIL |

Technical gateはruntime projectionだけFAILし、残りの数値、support、nonnegative
weight、brute-force、forward/backward table identity、truth-late、readback SHA、
finite coverage、RSS契約はPASSした。exp439で不可能だったfailure edgeも、
`v_eff=0.0264 ft²`、variance inflation `0.01139375 ft²`として正常に通過した。

Mechanismは6項目中2項目PASSだった。forward-cause episode SSEは
`10,827,465.786`から`10,230,142.572`へ5.517%改善したが、事前固定した10%に届かなかった。
persistent episode SSEは`13,363,710.665`から`14,134,214.141`へ5.766%悪化した。
4/5 foldsと10/16 wellsは改善した一方、fold 0が
`896,156.141`から`3,491,884.152`へ大きく悪化し、pooled判定を反転させた。

matched controlでも親RMSE `3.428436 ft`に対してcandidate `3.522134 ft`となり、
許容`+0.02 ft`を超えた。by-well p95も許容`+0.25 ft`に対して
`+1.394368 ft`だった。

## 解釈

exp443の主要な表現契約は成立した。連続台形平均を0.35-ft格子上でほぼ誤差なく保存し、
exp209のone-step grid mean biasをほぼ完全に除去できる。一方、9.67百万edgeで有効に
なった追加分散は平均`0.003905 ft²`で、persistentの一部には効いてもfold 0と
matched controlの安全性を損ねた。したがって、position積分の平均biasだけを除けば
rate lagが安全に改善するという仮説は支持されない。

さらにfull 773-well投影は約34.84時間で、Stage 1上限8.5時間を大幅に超える。
これはpackageやsolver failureではなく、実データ上で完走した科学候補のnegative
evidenceである。

## 再現性

- prediction logical SHA:
  `ef0f61b8c2adc42a52bf1e6c50c1b69f296b774754480a03f8ae11562b0643d7`
- moment audit logical SHA:
  `d0c6cba874437fcf8115b5b144b3d9445c13981e04d97afa7d25d3b222f110ae`
- rate readout logical SHA:
  `3585ca26e5bfc14af73200649da00bd345607c5c5d48a2ffadd34f50a36b339d`
- Kaggle metrics SHA:
  `a1a773a2f967c1ac48e4d041e2df1251e469b991a86e8cdbd963cdcb3e806bfe`
- gate report SHA:
  `38d2d8032f5a52bc419b33f233ceff7f2b3f1ea3d909e64423d6de5143df3ac9`

初回runはdeterministic anchorではない。ただしruntimeとmechanismの双方が明確に
FAILしたため、同一設定rerunによるanchor確立は行わない。

## 結論

mean-preserving trapezoidal lattice仮説はStage 0で棄却する。grid、support、
variance floor、`sig_p`、rate、emission、gate、blend、selectorを同じfixed32で
救済せず、Stage 1、inference、submissionへ進めない。原因分解を続ける場合は、
保存済みvariance audit / predictionだけを使う0-HMMの独立readoutを別実験・別承認で
行う。
