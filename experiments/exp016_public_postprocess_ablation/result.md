# exp016_public_postprocess_ablation 結果

## 仮説

Public notebook style の後処理候補を `exp013` raw OOF 上で比較すれば、
raw LightGBM no-GR 基準 より信頼できる後処理候補を特定できる。

## 設定

- 親: `exp013_model_diversity_or_postprocess`
- 検証: exp013 OOF の same-OOF 比較、leave-one-original-fold-out selection audit、stable well-hash holdout selection audit
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| Raw clean CV | 13.549257 |
| Best same-OOF | 13.501824 (`exp013_bucket_shrink`) |
| Best fixed same-OOF | 13.515133 (`alpha_tau_250_a020_115`) |
| Leave-one-original-fold-out selection | 13.551561 |
| Well-hash holdout selection | 13.515133 |
| Public LB | - |
| Private LB | - |

## 解釈

Same-OOF では `exp013_bucket_shrink` が最良だが、これは exp013 の同一
OOF fit alpha に由来するため、clean CV としては扱わない。固定候補では
`alpha_tau_250_a020_115` が same-OOF 13.515133 まで改善したが、original
fold 外 candidate selection は 13.551561 で raw 13.549257 よりわずかに悪い。

したがって clean CV 基準は raw `lightgbm_no_gr` 13.549257 のまま維持する。
Public LB 基準としては exp013 の postprocess 提出 12.271 を引き続き分けて扱う。

## 次

Public postprocess の単純な固定候補は採用せず、次は DWT/DTW route または
candidate quality/routing の検証に進む。
