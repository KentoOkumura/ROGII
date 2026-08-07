# 要件

## 依頼

`exp023` の best pseudo-tail recipe で Kaggle inference を実行し、`submission.csv` を生成する。

## 制約

- final fit は公式 train wells のみを使う。
- test well では既知 `TVT_input` prefix だけを使う。
- 未監査 postprocess は submit 候補にしない。初回 inference は raw pseudo-tail prediction とする。
- submit はユーザーが別途明示するまで行わない。

## 受け入れ基準

- exp024 inference notebook が Kaggle 用に生成できる。
- Kaggle inference output に `submission.csv` が生成される。
- `submission.csv` が sample submission と互換で、欠損・重複・行数不一致がない。
