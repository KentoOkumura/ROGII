# 要件

## 依頼

Kaggle 実験テンプレート用の初期リポジトリ構成を作成する。

## 制約

- データは `data/raw/` を正式な置き場にする。
- 既存の `notebooks/` ディレクトリは維持する。
- 生成データ、モデル成果物、提出物はデフォルトでは Git に含めない。
- `experiments/` 配下に、実験ごとのコードと設定をまとめる。
- エージェントの計画記録は `.steering/` 配下に保持する。

## 受け入れ基準

- リポジトリ全体のガイドファイルが存在する。
- `experiments/exp001_baseline/` に config、train、inference、notes、result、metrics ファイルが含まれている。
- `templates/experiment/` をコピーして今後の実験を作成できる。
- サマリー文書と提出履歴文書が存在する。
