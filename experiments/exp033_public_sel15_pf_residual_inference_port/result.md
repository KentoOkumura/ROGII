# exp033_public_sel15_pf_residual_inference_port 結果

## 要約

Kaggle inference version 1 が完了した。public sample の 3 wells はすべて見えている train wells で、変更なしの物理モデル処理を使うため、出力は exp027 / exp031 と完全一致した。

## 選択候補

- 候補: `ridge_residual_shrink0p5_clip20p0`
- 根拠: `exp032_public_sel15_pf_residual_補正`
- 推論式: `tvt_selector + 0.5 * clip(ridge_residual, -20, 20)`
- 適用範囲: 見えない test well のみ

## 検証

- ローカル構造検証: PASS
- Kaggle 用 notebook syntax check: PASS
- Kaggle inference: 完了、version 1
- Submit-check: PASS
- Public LB: 14.961、ref `53444678`

## 出力監査

- 行数: 14,151
- 欠損: 0
- 重複 ID: 0
- 予測範囲: 11587.038593 - 12240.016066
- 予測平均: 11903.630073
- 元 selector submission からの変更行数: 0
- 変更 well 数: 0
- 差分 RMSE: 0.000000
- SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- Code submit ref: `53444678`
- 残差学習の元生成物行数: 1,782,279
- 残差学習の抽出行数: 473,950

## 解釈

code-submit による見えない test well 用処理の評価は失敗だった。Public LB 14.961 は exp027 8.781 より +6.180、exp031 8.956 より +6.005 悪い。exp032 の見えない test 風データ上の残差改善は 見えない test well 評価の LB に転移しなかったため、この処理は採用せず、追加チューニングもしない。public sel15 の基準は exp027 のまま維持する。
