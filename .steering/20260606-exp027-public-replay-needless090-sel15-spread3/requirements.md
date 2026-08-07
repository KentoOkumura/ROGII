# 要件

## 依頼

`exp027_public_replay_needless090_sel15_spread3` を実装し、公開 notebook `needless090/lb8-781-rogii-sel15-spread3` を Kaggle 上で無改造 replay できる状態にする。

## 制約

- replay output が確認できるまで、自前 best `exp026` と blend しない。
- 公開 notebook のロジックは変更しない。Kaggle push 用 bootstrap cell の追加だけ許容する。
- CPU / internet off / external source なしの metadata 前提を維持する。
- CV は作らず、Kaggle inference output、submit-check、LB、runtime、dependency review を記録する。

## 受け入れ基準

- `experiments/exp027_public_replay_needless090_sel15_spread3/` が標準構造で作成される。
- inference notebook が archived public notebook body を持つ。
- `task validate-exp` が通る。
- `task prepare-kaggle-notebooks ... --notebook inference --run-on-push --strict` が通る。
- 次に Kaggle push / output / submit-check / submit 記録へ進める。
