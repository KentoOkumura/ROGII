# exp025_pseudo_tail_postprocess_cv_audit 結果

## 状態

完了。

## 要約

`pseudo_tail_3_cutoffs_distance_balanced` の fold-safe OOF 予測を再生成し、pseudo-tail 基準予測に対する後処理候補を監査した。提出は生成していない。

生の pseudo-tail CV は 12.942938。固定 `exp014_bucket_shrink_params` を適用すると 12.870780 まで改善し、original-fold と well-hash の固定候補 holdout でも毎回選択された。holdout 内で bucket alpha を学習する案も original-fold CV 12.887830、well-hash CV 12.879401 まで改善したが、固定 exp014 パラメータの方が強く単純だった。

## 判断

次は exp024 の生 pseudo-tail 推論処理に `exp014_bucket_shrink_params` を適用する推論実験へ進む。同一 OOF で alpha を学習した結果は診断値としてのみ扱う。
