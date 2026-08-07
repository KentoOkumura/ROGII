# exp203_heatmap_mdn_candidates_into_selector_features 結果

Kaggle train v1 完了。

## 結論

exp202 heatmap MDN topK を既存 selector の add-only feature として渡す案は、exp158 continuity baseline からは改善したが、直接の親である exp184 heatmap path feature selector には届かなかった。feature signal はあるものの、exp203 単体は inference / submit には進めない。

| 指標 | 値 |
| --- | ---: |
| best Viterbi RMSE | 10.665741318 |
| MAE | 6.350286735 |
| within10 | 0.797977743 |
| oracle label accuracy | 0.271556551 |
| path switches | 12,807 |
| path switches / 1000 rows | 3.384523581 |
| rows / wells | 3,783,989 / 773 |

## 比較

| 比較対象 | RMSE | exp203 差分 |
| --- | ---: | ---: |
| exp158 continuity | 10.789163253 | -0.123421935 |
| exp184 heatmap path feature | 10.560650325 | +0.105090994 |
| likpf_mean | 11.594897672 | -0.929156354 |

## 解釈

`hmdn_` feature は有効で、exp158 の既存 continuity selector からは改善した。一方、exp184 の `hmpf_` heatmap path feature 追加済み selector より RMSE が悪く、path switch も exp184 の 5,713 / 1.509782 per 1000 rows から 12,807 / 3.384524 per 1000 rows へ増えた。

今回の実装は heatmap MDN path を選択候補にしていないため、exp202 の大きな oracle headroomを直接回収する設計ではなかった。意図していた方向は、`pred_top*_tvt` を selector の selectable candidate として追加する次実験で検証する。

## 判断

train-side feature signal は確認できたが、exp184 を更新しないため no submit。次は `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158` で heatmap MDN topK path を候補集合に追加する。

## 生成物

- Kernel: `kentookumura/exp203-hmdn-selector-train` v1
- status: `KernelWorkerStatus.COMPLETE`
- model manifest: `exp203_heatmap_mdn_candidates_into_selector_features_model_manifest.json`
- feature count: 298
- `hmdn_` generated feature count: 75
- predictions decompressed SHA: `65d1b1c9120f61e8200a723560608344036bd8d4e1e0d426efe5728b505d1cc5`
