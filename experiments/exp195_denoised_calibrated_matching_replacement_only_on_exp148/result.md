# exp195_denoised_calibrated_matching_replacement_only_on_exp148 結果

## 状態

Kaggle train v1 完了。train-side CV が exp148 / exp190 から大きく悪化したため、推論化・提出はしない。

## 仮説

exp190 add-only では DCM block と exp145 learned likelihood confidence block が競合した可能性がある。`learned_likelihood_confidence` の `ll_*` 54列を外し、同じ候補信頼度役割を DCM block へ置き換えることで、exp148 anchor を改善できるか確認する。

## 評価設計

- `denoised_calibrated_matching_replacement_only`: base 196 features、`projection_correction`、`u_disagreement`、`denoised_calibrated_matching` を使う。
- `learned_likelihood_confidence`: active model feature list から完全に除外する。
- `exp148_fulltrain_control`: 再学習しない。保存済み exp148 metrics を historical baseline として参照する。
- GroupKFold 5 folds、well group、metric は RMSE。
- GPU runtime、3 LightGBM configs、5 folds、15 boosters。

## 結果

Kaggle train v1 は `kentookumura/exp195-dcm-replace-exp148-train` version 1 として完了。ログに summary JSON と pooled metrics が出力されたため、Kaggle output archive は取得していない。

| model | RMSE TVT | delta vs exp148 | delta vs exp190 add-only | prediction SHA256 |
| --- | ---: | ---: | ---: | --- |
| lgb0 | 9.612543035441323 | +1.0127571760624328 | +1.0108647599829599 | `f455fcf18496820e5da96c27bea9cf680c2b8a86fafd134d075a528ef5c7dd53` |
| lgb1 | 9.405030561019409 | +0.8410594397897402 | +0.8654060808851156 | `a871ccfa10f14f37b7000f6bfc4f045b9e8136d90a38cf53d7c1081c2cc34f3e` |
| lgb2 | 9.388749748145075 | +0.8789300293510003 | +0.8486761866381229 | `65c3c269a223bd6328fd53d2665766e5daba7ada6711128fff1d27e47c6e0f98` |
| lgb_mean | 9.409612610766938 | +0.9083314288711186 | +0.9060164512821132 | `cab0cd29913e8cae39c807f3588e170a93b3590f7ad90f0601dc048af624fb76` |

補足:

- rows: 3,783,989
- wells: 773
- features: 377
- feature join coverage: pass、dropped rows 0、dropped wells 0
- elapsed seconds: 15,193.865

## 解釈

replacement-only DCM は不採用。exp190 add-only では `lgb1` 単体に小改善があったが、`learned_likelihood_confidence` 54列を外すと全 config が大きく悪化した。DCM block は exp145 learned likelihood confidence の代替にはならず、exp148 の `ll_*` block が主要 signal を持っている可能性が高い。

current-test feature generation、inference port、submit は行わない。DCM 系を続ける場合も、`ll_*` block 置換ではなく、exp190 add-only の feature importance / error bucket を読んで一部の DCM quality signal だけを小さく使う方向に限定する。
