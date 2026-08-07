# exp022_distance_uncertainty_shrink 結果

## Status

- 状態: Kaggle train 完了
- Kaggle train: `kentookumura/exp022-uncertainty-shrink-train` v1
- Kaggle inference: 未実行
- 提出: 未実行

## Evaluation

- Parent reference: `exp021 weighted_distance_bucket_shrink` CV 13.415799
- Parent weighted raw: 13.470015
- Weighted raw: 13.470015
- Weighted + distance bucket shrink: 13.415799
- Weighted + uncertainty shrink conservative: 13.555887
- Weighted + uncertainty shrink medium: 13.747715
- Weighted + uncertainty shrink aggressive: 13.935058
- Public LB 基準: `exp013` 12.271
- exp021 Public LB: 12.523

## Interpretation

固定 uncertainty shrink 3 候補はいずれも全体 CV で `weighted_distance_bucket_shrink` を上回れなかった。near rows では aggressive / medium / conservative が rows 0-49 をわずかに改善し、conservative は rows 50-249 も微改善したが、rows 250+ で residual を縮めすぎて全体を悪化させた。

Best は親実験と同じ `weighted_distance_bucket_shrink` 13.415799 のため、exp022 は inference / submit に進めない。単純な fixed uncertainty shrink は打ち切り、再開する場合は original-fold 外 selection または abs-error model の held-out 監査に限定する。
