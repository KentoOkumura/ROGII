# {{ EXPERIMENT_NAME }}

## 状態概要

- ルート: ml_model
- 状態: planned
- 作成日: {{ TODAY }}
- 親実験: -
- 仮説要約: TODO
- 変更点要約: TODO
- リスク: TODO
- 次: TODO

## 正の記録

- 数値: [`metrics.json`](metrics.json)
- 結果、実行証拠、ユーザー判断: [`result.md`](result.md)
- 実行中の作業ログ: [`SESSION_NOTES.md`](SESSION_NOTES.md)
- 実装前の要件と設計: 対応する `.steering/` 文書

## 実行入口

- 学習 notebook: `{{ EXPERIMENT_NAME }}_train.ipynb`
- 推論 notebook: `{{ EXPERIMENT_NAME }}_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP={{ EXPERIMENT_NAME }}`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
