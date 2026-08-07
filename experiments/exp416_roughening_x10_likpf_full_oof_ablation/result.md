# exp416_roughening_x10_likpf_full_oof_ablation 結果

## 状態

Kaggle CPU 4 shardとstrict merge/readout version 2が完了。
train-side scientific / technical gateはFAILし、roughening x10を棄却してbranchを閉じた。

## 仮説

resampling roughening 10倍がparticle diversityとcorrect basinの再捕捉を改善し、
exp072 likelihood-PFの全OOF RMSEとpersistent-offset SSEを安全に下げる。

## 親と変更点

- 親: exp072
- treatment: position / rate rougheningを両方10倍
- control: 保存済みexp072 `likpf_mean`
- planned PF: 1 variant ×773 wells、500 particles ×128 seeds
- control rerun / model / booster / GPU: 0
- metric: RMSE、fold/scope/well-tail/persistent-offset AND gate

## 結果

compact self-contained trainとcontract testsを実装し、4 CPU shardで
773 wells / 3,783,989 rows / 98,944 seed-well trajectories /
49,472,000 particle startsを完走した。4 shardのscientific contract SHAは一致し、
control rerun / LightGBM / HMM / Beam / GPUはすべて0。

strict merge version 1は、exp209 cacheに存在しない
`likpf_mean_exp209_reconstructed`をpreflightで要求したため、shard merge・評価前に
停止した。exp410と同じ
`float32(hmm_mean_tvt - hmm_minus_likpf_mean)`復元へ修正し、同一kernel
`kentookumura/exp416-rough-x10-merge`のversion 2で評価を完了した。

主要結果:

| scope | roughening x10 RMSE | exp072 control RMSE | improvement |
| --- | ---: | ---: | ---: |
| overall | 13.617718 | 11.594894 | -2.022823 ft |
| raw GR observed | 13.435950 | 11.657445 | -1.778505 ft |
| raw GR missing | 14.000727 | 11.459183 | -2.541544 ft |
| MD since 1000+ | 14.902648 | 12.702987 | -2.199661 ft |
| hidden-like spatial | 15.783010 | 13.643821 | -2.139189 ft |
| hidden-like typewell-purged | 15.634950 | 13.506814 | -2.128136 ft |

- 5 foldsすべて悪化。fold別regressionは`+0.583911`から`+2.625954 ft`
- by-well RMSE delta p95は`+14.104742 ft`
- worst-well regressionは`+41.050361 ft`
- within-10ft率は`0.772793 -> 0.695594`
- 一方、事前固定16 persistent-offset episodesのSSEは
  `113,224,053.56 -> 85,257,299.90`で`24.700364%`改善

scientific AND gateは、persistent episode SSE以外の主要条件を満たさずFAILした。
technical gateも、exp209 reconstructed controlのrow parity最大差
`0.000471875 ft`が事前上限`0.00001 ft`を超えてFAILした。これはprobe未実行とは
別の判定である。行数、773 wells、fold 0--4、finite coverage、fallbackなし、
saved exp072 RMSE parity、truth-freeze、実行量、runtime、memoryはPASSした。

## 解釈

exp410のtarget-late sentinel改善は全OOFへ一般化しなかった。roughening x10は
overall biasを`-1.099425 -> -0.104186 ft`へ寄せたが、RMSE、within-10ft、
全fold、全stress scope、well-tailを大幅に悪化させた。局所的なpersistent-offset
回復と全体性能には強いtrade-offがあり、global roughening倍率の増加をprediction候補、
inference候補、deterministic anchorとして採用しない。

## 次

事前登録どおりroughening倍率、position/rate別、process noise、ESS、GR sigma、
seed/particle、well/row gate、same-OOF rescueを探索せずexp416を閉じる。
probe / inference / submissionは行わない。原因分解が必要なら、保存済みprediction /
well audit / by-well metricsだけを使う0-PF readoutを別承認で行う。
