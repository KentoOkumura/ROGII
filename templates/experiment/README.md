# {{ EXPERIMENT_NAME }}

## 状態

- ルート: ml_model
- 状態: planned
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: {{ TODAY }}
- 親実験: -

## 仮説

TODO

## 変更点

- TODO

## 検証方針

- Fold:
- Group:
- Stratification:
- Leakage Check:

## 実行入口

- 学習 notebook: `{{ EXPERIMENT_NAME }}_train.ipynb`
- 推論 notebook: `{{ EXPERIMENT_NAME }}_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP={{ EXPERIMENT_NAME }}`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- TODO

### 悪かった点

- TODO

### リスク / 注意

- TODO

## 次

- TODO

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
