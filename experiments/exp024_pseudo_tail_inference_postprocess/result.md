# exp024_pseudo_tail_inference_postprocess 結果

## 状態

Kaggle inference / submit 完了。`submission.csv` は submit-check PASS。

## 出力

- Kaggle kernel: `kentookumura/exp024-pseudo-tail-inference` version 1
- Training variant: `pseudo_tail_3_cutoffs_distance_balanced`
- Postprocess: `raw`
- Parent CV: 12.942938
- Public LB: 12.166 (`ref=53408921`)
- Final train wells: 773
- Final train rows: 242,843
- Submission rows: 14,151

## Sanity

- `submission.csv` は `sample_submission.csv` と header / row count が一致。
- duplicate ID なし。
- NaN / Inf / empty values なし。
- Prediction range: 11592.563603 - 12235.564650。

## 解釈

exp024 は exp023 best recipe の raw inference candidate を正常に生成し、Public LB 12.166 で従来 基準 `exp013` 12.271 を更新した。`exp021` は CV 改善に対して Public LB 12.523 と悪化したが、exp024 は CV/LB の向きが一致したため、pseudo-tail + distance-balanced training を現時点の主軸にする。

次は exp024 raw candidate を壊さない範囲で、postprocess 監査または sequence diversity を小さく比較する。
