# exp241_adaptive_likelihood_pf_trajectory_containment_audit 結果

## 状態

完了・train-side不採用。Kaggle CPU auditのshard 0/2/3、574/773 wells（74.3%）、
2,813,393 rowsを集計した。ユーザー判断によりshard 1は実行せず、全4 shardのstrict mergeを
主張しない部分監査として閉じた。

## 結果

| candidate | pooled RMSE | 1000_plus RMSE |
|---|---:|---:|
| saved exp072 `likpf_mean` | 11.323679 | 12.406563 |
| paired regenerated T=1 | 13.235174 | 14.444841 |
| gated T=2 | 13.223203 | 14.432237 |

T=2はpaired T=1よりoverallで`-0.011971`、1000_plusで`-0.012604`改善し、
shard 0/2/3のすべてで同じ向きだった。ただし保存済みexp072に対してはT=1/T=2とも
約+1.9悪く、exp232の大きな悪化は主にreplay parityの問題で、T=2 gate固有の悪化ではない。

hidden-likeはspatial `+0.012390`、typewell-purged `+0.011157`と僅かに悪化し、
worst-well delta最大は`+1.666805`だった。

first-gate event後のmean absolute path divergenceは8 rowsの`0.127775 ft`から、
1024 rowsの`1.399960 ft`、endの`3.123176 ft`へ増加した。terminal divergence平均は
`5.430025 ft`、event内max平均は`10.475701 ft`で、軌跡が再収束するcontainment仮説は
支持されない。一方、end cumulative RMSE delta平均は`-0.016838`、悪化event率は
`49.66%`で、path divergenceが系統的な誤差悪化を生む証拠もない。

overall/1000_plus guardだけが通過し、hidden-like、worst-well、late divergenceは不通過。
trajectory containmentを不支持、direct robust likelihoodを不採用とする。追加grid、
raw-test inference、submitは行わない。

## 根拠生成物

- shard 0/2/3の`candidate_metrics.csv`
- shard 0/2/3の`distance_bucket_metrics.csv`、`hidden_like_metrics.csv`、`by_well.csv`
- shard 0/2/3の`event_horizon_summary.csv`、`summary.json`、`metrics.json`

## 判定

paired T=1 control に対する gated T=2 のoverall / 1000_plus / hidden-like / worst-wellと、
event後horizon別のRMSE delta、path divergence、ESS、resamplingを評価した。

containment を支持するには、late horizonでdivergenceが増え続けず、overall、1000_plus、
worst-wellのいずれにも正の回帰を残さない必要がある。guard不通過なら robust likelihood の
temperature再grid、mixture、process-noise変更、raw-test inference、submitへ進まない。
今回はhidden-like、worst-well、late divergenceが不通過だったため不採用とした。
