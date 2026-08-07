# exp417_scale5_seed_aggregation_promotion_audit 結果

## 状態

Kaggle private CPU version 1でStage A完了。technical gateはPASS、
scientific gateはFAILしたため、固定scale-5 seed aggregationを棄却して
`stage_a_failed_closed_no_rescue`で閉じた。inference / submissionは実行していない。

## 仮説

同じPF trajectory bankに固定temperature-5 likelihood weightingを適用すると、
128 seed算術平均より安全にRMSEとacross-seed offsetを減らせる。

## 設定

- artifact parent: exp404
- scientific control: exp072
- control / candidate: x1.0 arithmetic mean / fixed scale5 weighted mean
- Stage A PF / model / booster / GPU: 0
- metric: RMSE、fold/scope/blend/well-tail AND gate

## 結果

- kernel:
  `kentookumura/exp417-scale5-seed-agg-promotion-audit-train` version 1
  （id_no `128917131`）
- rows / wells / reporting folds: `3,783,989 / 773 / 5`
- runtime: `154.215 sec`
- arithmetic control RMSE: `11.594897884`
- fixed scale-5 candidate RMSE: `10.914522073`
- pooled gain: `0.680375810 ft`
- fold改善: `5 / 5`
- raw-GR observed gain: `0.744954600 ft`
- fixed exp209 HMM/LikPF 50:50 gain: `0.184786467 ft`
- raw-GR missing / high-missing / 1000+ / hidden-like 2面: 全て非悪化
- direct by-well: 改善`516`、悪化`257`
- by-well delta RMSE p95: `+2.941688483 ft`、上限`0.0 ft`でFAIL
- worst well: `70925e23`、`+25.311274575 ft`、上限`0.25 ft`でFAIL
- technical gate: PASS
- scientific gate: FAIL

Stage AのPF / parent PF / HMM / Beam / model config / trained fold / booster /
GPU実行はすべて0。入力SHA、同一500-particle ×128-seed x1.0 bank、
prediction freeze前truth / fold / role読取0、保存control parityを全て確認した。
取得したartifact manifest 7件はbytes / SHAとも一致した。

## 解釈

固定temperature-5重みはpooled、全fold、事前固定scope、固定50:50 blendの平均性能を
明確に改善した。一方、well単位では257/773 wellsが悪化し、上側tailとworst wellの
悪化が大きい。pooled gainだけではdirect PF候補の安全な一般化を示せないため、
事前登録AND gateどおり不採用とする。

## 次

temperature / scale / best seed / median / mode / medoid / selector /
well・row gateを同じOOFで救済探索せず、exp417を閉じる。raw-test inferenceと
submissionへ進まない。今回の結果だけでexp413の独立ML全面置換branchを再分類しない。
