# exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264

## 状態

- ルート: ensemble
- 状態: Kaggle CPU version 1完了 / scientific gate FAIL / branch close
- hard OOF RMSE: `8.787855710`
- parent fixed12: `8.652531956`（差分`+0.135323754 ft`）
- Public / Private LB: 未提出
- Kaggle kernel id_no: `128436686`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 候補元: `exp362_segment_local_donor_slope_exact_hmm`

## 仮説

exp209比で平均改善した一方、wellごとの得失が大きかったexp362の
`prefix_rate_exact_hmm`を既存HMMと置換せず13番目の候補として追加すれば、
dual selectorが有効区間だけを選択できると仮定した。

ここで評価した候補はlocal donor slopeではない。exp362では局所gradient採用が
`0 / 12,368 segments`で、全segmentがprefix-rate-onlyへ退化していたためである。

## 実行

- canonical kernel:
  `kentookumura/exp375-exp362-prefix-fixed13-selector-train`
- version / id_no: `1 / 128436686`
- runtime: `6978.658914 sec`
- 1 variant × 2 objectives × outer 5 × inner 4 = 40/40 CPU selector models
- parent/control再学習、GPU、downstream TVT、inference、submission: すべて0
- exp362 OOFは6列allowlistだけを読み、global key join後にexp263 selector foldへ
  repartitionした。source foldはprovenance-onlyでmodel featureには使っていない。

## 検証方針

- exp263 selector outer 5 × inner 4のwell-disjoint nested stackingを固定した。
- outer-train compact scoreはinner OOF、outer-validは4 inner model ensembleだけを使った。
- 保存済みcorrected exp264 fixed12 hard OOF `8.652531956`を主比較にした。
- pooled、fold、near、1000+、hidden-like 2面、by-well p95、worst wellを
  事前固定AND gateで判定した。
- exp362 allowlist、decompressed SHA、global key parity、truth-late join、
  source-fold feature禁止、native confidence finiteをtechnical guardにした。

## 結果

| 指標 | 値 |
| --- | ---: |
| fixed13 hard OOF RMSE | 8.787855710 |
| parent fixed12 hard OOF RMSE | 8.652531956 |
| pooled delta | +0.135323754 ft |
| 改善fold | 0 / 5 |
| near delta | +0.037147066 ft |
| 1000+ delta | +0.148630322 ft |
| hidden-like spatial delta | -0.080214151 ft |
| hidden-like typewell-purged delta | -0.071070951 ft |
| by-well delta p95 | +1.047744567 ft |
| worst well delta | +28.995116411 ft |
| 追加候補top1率 | 11.525879% |

technical、leakage、selector score guardはすべてPASSしたが、pooled、全5 folds、
near、1000+、well tailを悪化させたため、
`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`とした。

post-freeze oracle診断ではH512 `0.162676643 ft`、whole-well
`0.123213333 ft`のheadroomがあった。候補の補完性はあるが、現selectorでは
安全に局在化できない。

## 所見

- score calibrationとleakage guardは通過しており、実装失敗ではなくscientific
  negative resultとして信頼できる。
- 追加候補usageは十分だったが全5 foldsで親を悪化させ、使用率だけでは安全性を
  説明できなかった。
- hidden-like 2面の改善だけを根拠にpooled・tail・worst-well gateをoverrideしない。
- oracle headroomは候補補完性の証拠に限定し、selector採用根拠にしない。

## 生成物

必要な小型metrics / manifestだけを
`kaggle/output/train_v1/artifacts/`へ取得した。

- `exp375_summary.json`
- `exp375_scientific_gate.json`
- `exp375_fixed13_vs_fixed12_scope_metrics.csv`
- `exp375_fixed13_candidate_usage.csv`
- `exp375_fixed13_vs_fixed12_by_well.csv`
- `exp375_postfreeze_addone_novelty.{csv,json}`
- `nested_selector_metrics.{csv,json}`
- `nested_selector_model_manifest.json`
- `nested_compact_manifest.json`
- `reproducibility_manifest.json`

49,191,857-row score parquetやmodel本体を含むoutput全体は取得していない。

## 解釈

追加候補は5/5 foldsで使われたが、parent比較は5/5 foldsで悪化した。
worst `b19b0395`では追加候補top1率が`0.258042%`しかないため、直接の誤選択だけでなく、
selector再学習による既存候補rerankingの不安定性が疑われる。

## 次

- 同一OOFでのweight / threshold / domain / gate救済は行わない。
- current-test候補生成、downstream TVT、inference、submissionへ進まない。
- 原因追跡が必要な場合だけ、exp371 / exp373 / exp375を横断する0-boosterの
  incumbent-reranking診断を別承認で検討する。

詳細は`result.md`と`SESSION_NOTES.md`を参照する。
