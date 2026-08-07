# タスクリスト

## TODO

- audit-only inferenceを許可するかユーザー判断を確認する。
- hard top1ではなくwell-risk / confidence gate付きdirect auditを別実験にするか確認する。

## 進行中

- なし。

## ブロック中

- safety guard不通過のため通常inferenceとdirect採用を停止。

## 完了

- 再現性設計を`design.md`に記入。
- exp245 steeringと実験ディレクトリを作成。
- parity-safe selector contextとexp226診断再生成を実装。
- train/inference notebookをKaggle packageへ変換。
- metadata、bootstrap manifest、静的検証を確認。
- Kaggle CPU selector train v1を完了。
- 143 context、missing/nonfinite 0、20 model SHAを監査。
- selector safety guard不通過とworst-well +38.016697を記録。
