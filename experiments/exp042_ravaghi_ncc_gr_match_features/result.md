# exp042_ravaghi_ncc_gr_match_features 結果

Kaggle train version 1 が完了。推論と提出は実行していない。

## 要約

| 項目 | 値 |
|---|---:|
| 行数 | 1,782,279 |
| wells | 773 |
| 最良の報告用比較基準 | `pf090_hold010` 15.089532 |
| Public PF selector | 15.172636 |
| Same-surface ML control | `exp026_regenerated_bucket_shrink` 16.483627 original / 16.429613 well-hash |
| Base geometry bucket-shrink | 19.089409 original / 18.959318 well-hash |
| 全体で最も強い特徴候補 | `base_plus_public_beam_aggregate_bucket_shrink` |
| 全体で最も強い特徴 CV | 16.123567 original / 16.132100 well-hash |
| 最良 NCC bucket 候補 | `base_plus_ncc_disagreement_bucket_shrink` |
| 最良 NCC bucket CV | 17.730920 original / 17.703975 well-hash |
| 最良 NCC raw 候補 | `base_plus_ncc_disagreement_raw` |
| 最良 NCC raw CV | 17.571472 original / 17.540442 well-hash |

## 解釈

Ravaghi 風 NCC / GR match 特徴は、弱い `base_geometry_bucket_shrink` 単体 LGBM 比較基準に対しては有効だった。最良 NCC bucket 候補は base から original-fold で -1.358489、well-hash で -1.255343 改善した。

ただし、ML route の同一 surface 比較としても、最良 NCC bucket 候補は `exp026_regenerated_bucket_shrink` より +1.247292 / +1.274362 悪い。raw NCC variants は bucket-shrink variants より少し強いが、同一 surface ML control には届かない。全体で最も強い `base_plus_public_beam_aggregate_bucket_shrink` は `exp026_regenerated_bucket_shrink` を -0.360060 / -0.297513 上回るが、これは NCC/GR match の効果ではなく public beam aggregate 系の候補である。

PF/Beam route との比較でも、`base_plus_ncc_disagreement_bucket_shrink` は `public_pf_selector` より +2.558283 / +2.531339 悪く、`pf090_hold010` より +2.641388 / +2.614443 悪い。exp041 の exact beam 食い違い特徴にも届かない。

## 判断

この実験は推論側へ移植せず、提出もしない。NCC/GR match 特徴は ML route の同一 surface control も PF/Beam route の direct controls も超えないため、直接置き換え候補ではなく、特徴量群比較の診断入力としてだけ扱う。

## 次

特徴量群を横並びで比較する必要がある場合だけ、`ravaghi_feature_family_ablation_matrix` へ進む。この結果だけを根拠に GR 位置合わせ実験を広げない。
