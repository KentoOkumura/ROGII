# 設計

## 1. 科学差分

exp209のrate kernelを`K_parent`とする。各行・source rateについて

```text
K = 0.99 * K_parent + 0.01 * K_broad
mu_broad = r_source - (1-0.998)*r_source*delta_MD
sigma_broad = 0.02
```

とする。`K_broad`は全rate binのVoronoi区間へGaussian CDFを積分する。
finite support外massは捨て、端で再正規化しない。branchは両方向対称で、
current GR、posterior innovation、truthから方向を決めない。

positionは親どおりdestination rateで進める。state、GR emission、prior、
forward/backward、posterior mean/stdも変更しない。

## 2. 固定理由と独立性

`0.01`は通常kernelを99%残すrare branch、`0.02`はnominal 0.005 rate binの
4 bin幅として事前固定する。grid探索はしない。

exp411はGR innovation triggerの将来方向一致`0.225397`、positive fold`0/5`だった。
したがってtriggerを再利用せず、HMM尤度に対称候補を比較させる。

compact実装候補とcontract testは2026-07-29のユーザー明示依頼により先行作成した。
当初はexp441がtechnical/control-safeかつ方向正・量不足の場合だけ実行する
保守的な先行条件を置いたが、2026-07-30のユーザー判断により撤回する。

exp441は親kernel全体をexact OUへ置換する一方、exp442はexp209 local kernelを
99%残し、尤度が必要とする場合だけ使える1% broad branchを追加する。科学差分と
通常区間への影響が異なるため、exp442を独立したdefensive mixture仮説として
評価する。`0.01` / `0.02`、fixed32、gateはexp441結果を見る前の固定値を維持し、
exp441のFAILをpositive evidence、gate変更、same-OOF rescueには使わない。

## 3. Stage 0

fixed32の1候補×32 wells。local/broad/mixture kernel、posterior branch
responsibility、prediction、diagnosticをfreeze後、truth-late評価する。

2026-07-30のユーザー依頼により、正規train Notebook採用、Kaggle private CPU
package、Stage 0実行を承認済みとする。保存exp209 controlを使い、parent rerunは0。
LightGBM config / trained fold / booster / fitted model / PF / Beam / GPUは全て0。

Technical:

- local branch parent parity、mixture分解、broad in-support mass誤差`<=1e-12`。
- posterior normalization、brute-force responsibility差`<=1e-6`。
- finite coverage 1.0、pre-freeze truth read 0。
- full換算`<=30,600 sec`、RSS`<=25 GB`。

Mechanism:

- non-adjacent posterior edge mass`>=0.001`。
- jump edgeのfuture-rate方向一致`>=0.60`、positive`>=4/5 folds`。
- forward-cause SSE`>=10%`、persistent SSE`>=5%`削減。
- persistent改善`>=10/16 wells`、`>=4/5 folds`。
- control pooled`<=+0.02 ft`、p95`<=+0.25 ft`。

FAIL時はweight、sigma、trigger、emission、grid、gateを救済しない。

future-rate方向はexp411と同じ、各transitionの前32行physical interval rate中央値と
次32行中央値の差とする。jump方向はnon-adjacent broad posterior edgeの
responsibility加重rate差の符号とし、方向一致率も同edge massで加重する。
positive foldは加重一致率が厳密に`0.50`を上回るfoldとする。

## 4. Stage 1と再現性

全PASS・別承認時のみ同じ1候補を773 wellsで実行する。direct RMSE gain
`>=0.05 ft`、4/5 folds、固定scope、by-well tailをAND判定する。

mixtureはsamplingせず厳密周辺化するためRNGなし。固定well/row/state/branch順で
kernel、responsibility、prediction、metrics SHAを記録する。Stage 0/1 HMM runsは
`32/773`、parent rerun 0、ML/PF/Beam/GPUは0。

初回Stage 0はmechanism preflightであり、CV / promotion evidenceでも
deterministic anchorでもない。Stage 1、inference、submissionは別承認まで禁止する。
