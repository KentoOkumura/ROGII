# exp017_deterministic_dtw_addonly 結果

## 結論

Kaggle full CV では `dtw_dwt_no_gr` が raw 基準 より悪化したため、DTW/DWT add-only features は採用しない。

## 評価

- Raw 基準: `exp012/exp013 lightgbm_no_gr` CV 13.549257
- Selected variant before CV: `dtw_dwt_no_gr`
- Raw 基準 `control_lightgbm_no_gr`: CV 13.549257
- DTW/DWT add-only `dtw_dwt_no_gr`: CV 13.949718
- DTW/DWT bucket postprocess: CV 13.910963
- Public LB: なし

## 解釈

DTW/DWT alignment quality features は、add-only では raw LightGBM no-GR 基準を上回らなかった。`postprocess_metrics.csv` の same-OOF bucket shrink でも 13.910963 までで、raw 13.549257 との差は大きい。次に使うなら、直接特徴追加ではなく candidate routing / confidence / selective fallback に限定して再設計する。
