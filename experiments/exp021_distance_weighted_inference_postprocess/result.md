# exp021_distance_weighted_inference_postprocess 結果

## Status

- 状態: Kaggle train / inference 完了
- Kaggle train: `kentookumura/exp021-dw-post-train` v2
- Kaggle inference: `kentookumura/exp021-dw-post-infer` v1
- submit-check: PASS
- 提出: 完了、ref `53406803`

## Evaluation

- Parent reference: `exp020 near_down_far_up_lightgbm` CV 13.470015
- Weighted raw CV: 13.470015
- Weighted + distance bucket shrink CV: 13.415799
- Public LB: 12.523
- Submission: `data/external/kaggle-output/exp021_distance_weighted_inference_postprocess/inference/submission.csv`
- Submission SHA256: `f0e1289b28453b558978ebc48986fa4fd3a85d1ba05299e4455fca0b4a00611f`

## Interpretation

`weighted_distance_bucket_shrink` は weighted raw から -0.054216 改善し、exp020 parent 13.470015 と exp014 held-out postprocess reference 13.535596 をどちらも上回った。near rows の悪化は bucket shrink で大きく補正され、rows 0-49 は 3.576164 から 0.963697、rows 50-249 は 4.078820 から 3.572572 に改善した。

Public LB は 12.523 で、現行 基準 `exp013` Public LB 12.271 より悪化した。CV では `exp021` が clean 基準を更新したが、visible public wells では exp013 の distance bucket shrink の方が強い。Public LB 基準は exp013 のまま維持し、exp021 は clean CV 改善候補として private 期待と次の uncertainty shrink の材料に使う。
