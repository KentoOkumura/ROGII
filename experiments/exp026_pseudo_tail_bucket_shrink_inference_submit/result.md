# exp026_pseudo_tail_bucket_shrink_inference_submit 結果

## 状態

完了。Kaggle inference が完了し、submit-check とコンペ提出も完了した。

## 要約

exp025 で選択した固定 `exp014_bucket_shrink_params` を、exp024 の pseudo-tail 推論処理に適用した。

Kaggle inference v1 は 14,151 行の `submission.csv` を生成し、`data/raw/sample_submission.csv` に対する submit-check は PASS。予測範囲は 11590.725143 から 12237.368348、平均は 11907.302608。exp024 の生 pseudo-tail submission と比べた予測差分は min -1.838460、max 1.803698、mean 0.102789、abs mean 0.438886、RMSE 0.611885。

## 判断

提出 `ref=53411137` の Public LB は 12.102。exp024 の生 Public LB 12.166 から 0.064 改善し、この時点の Public LB 基準になった。
