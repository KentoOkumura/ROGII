# exp035_public_sel15_pf_meta_inference_port 結果

## 仮説

`exp034` の selected `ridge_meta_residual_shrink0p75_clip60p0` を 見えない test well 用推論 に移植すれば、exp026-style pseudo-tail 基準 と public sel15 PF/Beam 診断を組み合わせた補正が 見えない test wells でも改善する可能性がある。

## 設定

- 親: `exp034_public_sel15_pf_meta_stack`
- port 対象: `ridge_meta_residual_shrink0p75_clip60p0`
- 基準予測: exp026-style pseudo-tail 基準
- 補正: `0.75 * clip(ridge_meta_residual, -60, 60)`
- training source: exp029 `public_sel15_pf_oof_features.csv.gz`
- 補助 CV: exp034 original-fold 14.313668、well-hash 14.172010

## 結果

| メトリック | 値 |
| --- | --- |
| rows | 14,151 |
| output SHA256 | `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815` |
| submit-check | PASS |
| changed rows / wells | 0 / 0 |
| diff RMSE vs exp026 基準 sample output | 0.000000 |
| 予測範囲 | 11587.038593 - 12240.016066 |
| code submit ref | `53452712` |
| Public LB | 13.738 |

## 解釈

Kaggle inference version 1 は完了し、exp026-style 基準 model と Ridge meta residual model は notebook 内で fit された。public sample は全 3 wells が 見えている train wells で 物理処理を使うため、見えない test well 用 meta 処理は sample output では発火せず、exp027/031/033 と同一 SHA になった。

提出 ref `53452712` は Public LB 13.738 で完了した。exp027 基準 8.781 から +4.957 悪化したため、この 見えない test well 用 meta 処理は採用しない。

exp034 の train well の途中以降を隠した疑似 test 評価条件では selected candidate が original-fold 14.313668 / well-hash 14.172010 と強かったが、見えない test well 評価の LB には転移しなかった。exp031 の hold blend、exp033 の residual 補正、exp035 の meta stack のいずれも public sel15 の見えない test well 用処理改変は exp027 基準を下回ったため、公開 sel15 replay は exp027 のまま固定する。

## 次

exp027 基準 8.781 を維持する。public sel15 PF/Beam の疑似 test 生成物に基づく見えない test well 用処理補正の追加チューニングには進まない。
