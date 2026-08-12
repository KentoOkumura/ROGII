# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `exp082_public_artifact_replay_followup_inference.ipynb` を jaemin archived source 版に差し替えた。
- `validate_experiment` と `prepare_kaggle_notebooks --notebook inference --strict` を実行した。
- Kaggle inference commit v1 を実行し、output を取得した。
- output 取得後に prediction SHA、submission SHA、sidecar diff、Kaggle kernel version を記録した。
- `submit-check` と `scripts/validate_submission.py` を通した。
- notebook version submit はユーザー側で実行済み。ref `53896556` は Public LB `7.602`、ref `53896658` は complete no public score。ref `53896594` は exp096 として再帰属した。
- exp082 の現 ensemble anchor は fle3n final ref `53885305` / Public LB `7.601` のままにした。
