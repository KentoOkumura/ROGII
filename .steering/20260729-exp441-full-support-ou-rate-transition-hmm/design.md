# 設計

## 1. 科学差分

exp209の状態`(TVT_t,r_t)`は維持する。現在のrate transitionは各source binから
`i-1/i/i+1`だけへ移るEuler型kernelである。これを連続OU過程

```text
kappa = -log(0.998)
E[r_t | r_{t-1}=r_i] = exp(-kappa*h) * r_i
Var[r_t | r_i] = sig_r^2 * (1-exp(-2*kappa*h)) / (2*kappa)
h = delta_MD
```

へ置換する。`kappa -> 0`ではvarianceを`sig_r^2*h`とする。
Gaussian conditionalを全41 binのVoronoi区間へCDF積分する。有限support外tailは
親のrate端と同様に捨て、端のrowを再正規化しない。

rate更新後のposition transitionは親の
`r_destination*h-delta_Z`と5点Gaussianをそのまま使う。したがって、
「rateを更新してからTVTを更新する」順序は変えず、rateの到達速度だけを変える。

## 2. 固定条件

- position grid step `0.35 ft`、band `100 ft`、`sig_p=0.02`。
- per-well zero-centered 41 rate grid、minimum span `0.10`。
- `momentum=0.998`、`sig_r=0.002`、initial rate sigma `0.01`。
- Gaussian typewell GR emission、lambda `1.0`、prefix calibration。
- forward/backward、posterior mean/std、float32 message。
- rate/position端のparent truncation semantics。
- 保存済みexp209 predictionをcontrolとし、control HMMは再実行しない。

## 3. Stage 0

exp411 fixed32のpersistent 16 / matched-control 16を使う。candidate 1本を
32 wellsで実行し、kernel、prediction、diagnostic、SHAをfreezeした後だけ
truth、role、fold、episode/causeをjoinする。

Technical AND gate:

- 32 wells / 156,088 suffix rows / 5 folds、finite coverage 1.0。
- analytic in-support mass、interior OU mean/variance誤差`<=1e-12`。
- posterior normalization、brute-force小規模HMM差`<=1e-6`。
- position kernel parity`<=1e-12`、pre-freeze truth read 0。
- full換算`<=30,600 sec`、peak RSS`<=25 GB`。

Mechanism AND gate:

- zero-directed under-response SSE shareを絶対5 points以上削減。
- forward-cause episode SSEを10%以上削減。
- persistent episode SSEを5%以上削減。
- persistent改善`>=10/16 wells`かつ`>=4/5 folds`。
- matched-control pooled delta`<=+0.02 ft`、by-well p95`<=+0.25 ft`。

一つでもFAILならOU式、`sig_r`、momentum、support、emission、grid、gateを
同じ実験で救済せず閉じる。

## 4. Stage 1

Stage 0全PASSと別承認後だけ同じ1候補を773 wellsで実行する。direct RMSE gain
`>=0.05 ft`、4/5 folds、forward/persistent改善、near/mid/1000+、
hidden-like 2面、by-well p95/worstをAND判定する。PASSしてもinference、
submission、blend、selectorは別設計・別承認とする。

## 5. 再現性と実行量

- RNGなし。well/row/state/reduction順固定。
- kernel float64、messageは親どおりfloat32。
- input/kernel/prediction/diagnostic/metrics SHAを保存する。
- Stage 0 / Stage 1 candidate HMM runs=`32 / 773`、parent rerun 0。
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPUは全て0。
- 初回runはdeterministic anchorとせず、独立rerun SHA一致後だけ再判定する。

## 6. 実装フェーズ

2026-07-29のユーザー依頼により、次だけを実装する。

- compact self-contained Jupytext train/inference候補。
- exact OU全support CDF kernel、exp209 position kernel、float32 message。
- analytic mass/moment、position parity、小規模dense brute-force contract。
- 全32 kernel/prediction/diagnostic SHA freeze後のtruth-late readout。
- exp408 parent row ledgerによるzero-directed under-response比較。
- 専用test、Jupytext変換、構文、Ruff、strict experiment validation。

正規Notebook採用、Kaggle package/push/runはこの実装承認に含めない。

## 7. Stage 0結果

2026-07-30の別承認後、正規train Notebook採用とKaggle private CPU
Stage 0 version 1を実行した。32 wells / 156,088 rowsを1,582.080秒、
peak RSS 1.123249 GBで完走した。

- technical: 16 / 17 PASS。full 773-well換算38,217.120秒が
  30,600秒上限を超えた。
- mechanism: 2 / 7 PASS。matched-control pooled / by-well p95だけPASS。
- under-response share削減0.022974、forward episode SSE削減-0.001635、
  persistent episode SSE削減-0.016743、改善8 / 16 wells・1 / 5 folds。

設計どおり`stage0_fail_closed`とし、OU parameter、support、emission、
grid、gateを救済せず、Stage 1、inference、submissionへ進まない。
