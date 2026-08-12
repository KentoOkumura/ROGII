# タスクリスト

## 未実行

- Kaggle train notebook を prepare / push する場合は、config と bootstrap の整合を確認する。
- Kaggle train 完了後、logs / cell output から CV、fold なしの audit metrics、生成物 path、SHA を `SESSION_NOTES.md` / `result.md` に記録する。
- positive の場合だけ、bucket guard、worst-well regression、raw-test parity、posterior trajectory の物理妥当性を追加確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements / design を作成した。
- `exp173_beam_topk_path_posterior_audit` を作成した。
- Beam top-K path/cost audit module と config を実装した。
- Kaggle push 前の計算規模を config / SESSION_NOTES に記録する方針を決めた。
