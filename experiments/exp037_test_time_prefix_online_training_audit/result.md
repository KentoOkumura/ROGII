# exp037_test_time_prefix_online_training_audit 結果

## 状態

完了。Kaggle train version 1 が完了し、生成物を取得済み。

## 要約

exp026 の固定 bucket-shrink 比較基準の上に、見えている `TVT_input` prefix だけから作った小さな重みの test-time online training rows を追加する案を監査した。

同一 OOF の集計では小さな改善が出た。

- `online_weight_0_20`: 12.844383、比較基準から -0.026396
- `online_weight_0_05`: 12.855228、比較基準から -0.015552
- 比較基準 `exp026_bucket_shrink_control`: 12.870780
- `online_weight_0_10`: 12.909600、比較基準から +0.038820
- 生 pseudo-tail: 12.942938、比較基準から +0.072158

しかし fold 外の候補選択では online 候補を支持しなかった。

- leave-one-original-fold-out selection: 12.999364、比較基準から +0.128584
- well-hash holdout selection: 12.970333、比較基準から +0.099553

## 解釈

online rows は同一 OOF の集計では特に weight 0.20 で改善するが、original-fold や安定した well-hash holdout にはきれいに転移しなかった。prefix への過適合、または fold 固有の適応に見える。主催者が test-time online training を許容するか未確認なので、ルール面のリスクも残る。

## 判断

prefix online training は推論側へ移植しない。自前ルートの選択手法は `exp026_bucket_shrink_control` のまま維持する。
