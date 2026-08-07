# 設計

## アプローチ

Kaggle Notebook 実行を正とし、runtime 検出、入力パス、出力パス、実行手順をそろえる。

- `settings.py` に Kaggle runtime 検出と明示ローカル override を追加する。
- Kaggle runtime では `/kaggle/input/<competition>` を train/test/sample の入力元にし、見つからない場合は fail fast する。
- Kaggle runtime では `/kaggle/working` を experiment/output root にし、`submission.csv` も project config からそこへ解決する。
- notebook setup セルで `paths.require_kaggle_runtime()` を呼び、通常のローカル実行を止める。
- `scripts/execute_experiment_notebook.py` は `--allow-local` がある場合だけ smoke debug を許可する。
- AGENTS / docs / task descriptions は Kaggle prepare/push/status を通常ワークフローとして記述する。

## 実験範囲

- 対象実験: `exp001_baseline` と `templates/experiment`
- 親実験: なし。既存 baseline の runtime/path 前提のみ変更する。
- 変更する変数: runtime path resolution、notebook setup、workflow docs
- 固定する変数: baseline 予測ロジック、CV split、metric、model params

## リスク

- リークリスク: 予測ロジックは変更しないため低い。Kaggle hidden sample を必ず使うことで local sample 固定の事故を避ける。
- CV/LB 不一致リスク: CV 数値は再計算しない。runtime path 変更が出力場所だけに閉じているか validate/prepare で確認する。
- ランタイム/メモリリスク: `/kaggle/working` に artifacts を出すため output サイズが増える可能性がある。exp001 の CSV artifacts は小さい。
