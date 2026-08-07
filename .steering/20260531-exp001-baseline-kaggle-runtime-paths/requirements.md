# 要件

## 依頼

ノートブック実行をローカルではなく Kaggle 環境で行う前提に統一する。

## 制約

- Kaggle Notebook の `/kaggle/input` を公式データ入力として優先する。
- Kaggle Notebook の `/kaggle/working` を metrics、artifacts、submission の出力先にする。
- ローカル notebook 実行は通常ワークフローから外し、必要な場合だけ明示 override で smoke debug する。

## 受け入れ基準

- `exp001_baseline` と experiment template の `settings.py` が Kaggle runtime を検出して Kaggle path を優先する。
- train / inference notebook がデフォルトで Kaggle runtime を要求する。
- 手順書とタスク説明が Kaggle prepare/push/status を通常経路として示す。
- `validate_experiment.py` と `prepare_kaggle_notebooks.py --strict` が通る。`task` が使える環境では同等の Taskfile コマンドでも通る。
