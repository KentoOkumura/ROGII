# exp355 結果

## 状態

Kaggle CPU Stage 0 version 1の平均改善を根拠にユーザーがworst-well gateを
overrideし、Stage 1 version 2を完了した。技術gateはPASSしたが、科学gateは
8件中5件PASS、hidden-like 2面とworst-well guardの3件FAILでbranchを閉じる。

## 仮説

exp226 geometryの相対rate変化をexp209 rate-prior meanへ単独で移植できるかを、
HMM前のidentifiability readoutで判定する。

## 判定予約

Stage 0はrate-change/path gain、4/5 folds、1000+、hidden-like 2面、worst guardを
すべて要求する。PASS後もStage 1は別承認で、exp209比0.05 ft以上を要求する。

## Stage 0結果

- rows / wells: `3,783,989 / 773`
- runtime: `460.872765 sec`
- segment rate-change RMSE: `0.018237982 -> 0.016710597`
  （gain `8.374744%`、4/5 folds改善）
- cumulative path RMSE: `49.493155005 -> 46.977325325`
  （gain `2.515829680 ft`、5/5 folds改善）
- 1000+ delta: `-2.828214214 ft`
- hidden-like spatial / typewell-purged delta:
  `-1.830574021 / -2.056030635 ft`
- by-well: 439改善 / 334悪化、median `-1.941905 ft`、
  p95 `+20.585463 ft`
- worst: `071d7b45`, `+69.017669 ft`
- fallback well / segment: `0 / 0`
- 13 raw SHAと3 gzip decompressed SHAは全一致

## 解釈

exp226 geometryの相対rate変化にはpooled・fold・long-tail・hidden-likeで一貫した
平均signalがある。しかし悪化tailが非常に大きく、固定scheduleを全wellへ適用する
Stage 1は安全でない。worst guardは`+0.25 ft`に対して`+69.018 ft`であり、閾値近傍の
偶然ではないため、同じOOFでclip、gate、区間数、rate scaleを救済しない。

## Stage 1実行契約

- candidate: `exp226_geometry_rate_prior_mean_residual_hmm` 1件
- exact-HMM: 773 well-runs
- model config / trained fold / booster / parent-control rerun: すべて0
- Stage 0 schedule / ledger logical SHAをhard guard
- suffix truth、saved exp209、fixed LikPF 50:50、hidden-like roleはprediction freeze後に結合
- parameter rescue、inference、submissionは対象外

## Stage 1結果

- kernel version / id_no: `2 / 128366148`
- runtime: `18,161.789478 sec`（`5.045 h`）
- rows / wells / HMM runs: `3,783,989 / 773 / 773`
- direct RMSE: `11.938287235 -> 11.291976616`
  （`0.646310619 ft`、`5.4138%`改善、5/5 folds改善）
- fold改善量: `1.818538 / 0.383520 / 0.523301 / 0.320850 / 0.420460 ft`
- fixed LikPF 50:50: `10.269696317 -> 10.053143746`
  （`0.216552571 ft`改善、4/5 folds。fold 4は`0.238414 ft`悪化）
- 1000+ delta: `-0.730160080 ft`
- hidden-like spatial / typewell-purged delta:
  `+0.414943459 / +0.371719953 ft`
- well別: 360改善 / 413悪化、paired delta中央値`+0.012873 ft`、
  paired delta p95 `+5.663043 ft`
- candidate/parentのwell-RMSE分布p95差は`-2.541374 ft`で事前gateをPASSしたが、
  paired well deltaの上側tailは悪化した
- worst: `86454a6f`, `+52.743754 ft`
- fallback well / segment / prefix-rate fallback: `0 / 0 / 0`
- prediction logical SHA:
  `634303f022bced6685367094304da6182fee42815302344469b5919a36cd5e21`

## 最終解釈

row-weighted pooled RMSEと全foldが改善したため、exp226 geometryの相対rate変化には
exact-HMMでも有効な平均signalがある。ただし、well数では過半数の413/773が悪化し、
hidden-test-like 2面も悪化した。さらにStage 0のworst `071d7b45`とは別の
`86454a6f`がStage 1で大幅悪化しており、単一外れ値だけの問題ではない。

したがって「平均改善する候補」としては成功だが、「未知wellへ安全に移す候補」としては
不合格である。事前gateとfailure actionを維持し、同じOOFでのparameter、clip、blend、
selector救済は行わず、inferenceとsubmissionにも進まない。
