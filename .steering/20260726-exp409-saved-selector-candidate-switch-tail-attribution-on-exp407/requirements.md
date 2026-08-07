# 要件

## 依頼

`KAGGLE_DIRECTION.md`の
`saved_selector_candidate_switch_tail_attribution_on_exp407`を実装する。
exp407のinverse-RMSE weightingで悪化したtailを、corrected exp264 Stage B v5から
exp407 Stage B v1へのhard-selected candidate遷移として保存OOFだけで原因分解する。

## 制約

- Route: `ml_model`
- 新規model、booster、prediction、PF/HMM/Beam、inference、submissionは0。
- 親exp264とexp407のcandidate-score OOFをSHA固定し、候補値、候補順、
  11候補hard selection domain、foldを変更しない。
- `pred_abs_error`による両surfaceの選択、distance bucket、hidden-like role、
  transition inventoryをtruth-freeにfreezeし、freeze SHAを保存してから
  `actual_abs_error`だけを読む。
- exp407を救済または再分類しない。weight、threshold、候補、clip、
  exponentのgridを作らない。
- worst wellはexp407で確定済みの`52f1e77a`に固定し、再選択しない。
- 初回Notebook実行はKaggleを正とする。ローカルNotebook実行は行わない。
- 親corrected Stage B v5 OOFをKaggleで利用可能なprivate inputへ置く操作と
  Kaggle実行は別承認とする。
- `docs/06_reproducibility.md`に従い、input、freeze、truth ledger、
  row attribution、summaryのSHAを記録する。

## 受け入れ基準

- Jupytext percent形式のcompact self-contained train候補があり、
  input check、truth-free freeze、truth join、集計、gate、生成物保存を
  Notebookセルで追える。
- candidate-long 12候補の順序、base key、candidate value、fold、
  actual-error parityをfail-closedで検証する。
- additiveな`exp407 SSE - parent SSE`をtransition、fold、distance、
  hidden-like、well別に保存する。
- 1000+、hidden-like spatial、hidden-like typewell-purgedの各scopeで、
  同じ正のexcess-SSE rank-1遷移が4/5 foldsに再現し、固定worst wellでも
  rank-1かを閾値gridなしで判定する。
- synthetic testでtruth早期read拒否、candidate order、tie break、
  key/value parity、SSE集計、4/5 gateを検証する。
- `execution.run_approved=false`を維持し、Kaggle package/push/runを行わない。
- strict experiment validation、Jupytext round-trip、py_compile、
  Ruff F821、関連pytestがPASSする。
