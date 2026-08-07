# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 旧 exp056 artifact ベース audit を無効化した。
- `public_notebook_replay_audit.py` を追加した。
- train notebook を strict public raw replay に差し替えた。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` を更新した。
- 旧 artifact audit script と未使用 helper を削除した。
- `scripts/prepare_kaggle_notebooks.py` を experiment-level GPU metadata 対応に更新した。
- `py_compile`、`ruff check`、`validate_experiment` を通した。
- Kaggle train package を生成した。
- Kaggle train strict replay version 3 を実行した。
- Kaggle logs/output を取得し、生成物を同期した。
- `metrics.json`、`result.md`、`README.md`、`SESSION_NOTES.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を結果で更新した。
- `public_notebook_replay_audit.py` に `run_public_replay_inference` を追加した。
- inference notebook の no-op guard を外し、selected Pixiux LGBM replay candidate で `submission.csv` を生成する構成に差し替えた。
- inference では hidden-specific branch、guarded overlap override、static visible override、pretrained booster、CatBoost、Ridge stack、final blend、projection を除外した。
- inference version 1 は train feature を再生成する設計だったため、手動停止した。
- train notebook で Pixiux fold LightGBM booster を保存し、inference notebook は saved booster + test-only feature generation を使う構成に修正した。
- 後続実験で再利用できる PF/Beam/likelihood-PF tracker feature frame の train/test csv.gz 保存を追加した。
