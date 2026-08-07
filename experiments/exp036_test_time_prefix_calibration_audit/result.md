# exp036_test_time_prefix_calibration_audit 結果

## 状態

完了。Kaggle train version 1 が完了し、生成物を取得済み。

## 要約

exp026 の固定 bucket-shrink 比較基準の上に、test-time prefix calibration を加える案を監査した。比較したのは、生 pseudo-tail 予測、exp026 比較基準、prefix bias、prefix error slope、prefix global residual shrink、prefix distance-bucket shrink、prefix near-continuity decay。

最良は exp026 比較基準のままだった。

- 比較基準 `exp026_bucket_shrink_control`: 12.870780
- `prefix_near_continuity_decay`: 12.916015、比較基準から +0.045236
- 生 pseudo-tail: 12.942938、比較基準から +0.072158
- `prefix_distance_bucket_shrink`: 13.119153、比較基準から +0.248373
- `prefix_global_residual_shrink`: 13.119682、比較基準から +0.248902
- `prefix_bias_add`: 15.551284、比較基準から +2.680505
- `prefix_error_slope`: 19.540902、比較基準から +6.670123

fold 外の候補選択監査でも、すべての holdout で比較基準が選ばれた。

- leave-one-original-fold-out selection: 12.870780
- well-hash holdout selection: 12.870780

## 解釈

見えている prefix から作った補正残差は、本来の見えない tail に安定して転移しなかった。bias と error-slope 補正は見えている prefix に過適合し、近距離 rows を大きく壊した。residual-alpha 系候補は 50-249 rows では比較基準より良いが、250+ rows と全体を悪化させた。near-continuity decay は 250-999 rows でごく小さい改善があるものの、0-249 rows を壊すため、この実装では使えない。

## 判断

prefix calibration は推論側へ移植しない。自前ルートの通常 CV 基準は exp026 固定 bucket-shrink、全体の Public LB 基準は exp027 のまま維持する。
