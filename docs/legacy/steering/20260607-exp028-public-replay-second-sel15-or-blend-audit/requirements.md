# 要件

## 依頼

`exp028_public_replay_second_sel15_or_blend_audit` を実装し、公開 notebook `needless090/lb-8-860-rogii-sel15-256seeds` を Kaggle 上で無改造 replay できる状態にする。実験名に blend audit を含むが、まずは2本目 replay を優先し、blend は replay output / submit-check / LB が揃ってから判断する。

## 制約

- Route: `pf_beam`
- 公開 notebook のロジックは変更しない。Kaggle push 用 bootstrap cell の追加だけ許容する。
- CPU / internet off / external source なしの metadata 前提を維持する。
- CV は作らず、Kaggle inference output、submit-check、LB、runtime、dependency review、exp027 との差分を記録する。
- exp027 と exp028 の blend submission は、この実験の replay output と LB が未確認の間は作らない。

## 受け入れ基準

- `experiments/exp028_public_replay_second_sel15_or_blend_audit/` が標準構造で作成される。
- inference notebook が archived public notebook body を持つ。
- `uv run python scripts/validate_experiment.py --experiment exp028_public_replay_second_sel15_or_blend_audit` が通る。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp028_public_replay_second_sel15_or_blend_audit --notebook inference --run-on-push --strict ...` が通る。
- 次に Kaggle push / output / submit-check / exp027 差分比較 / submit 記録へ進める。
