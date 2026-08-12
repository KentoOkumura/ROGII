# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- exp161 の2段階 cache/train 構成を exp166 用にコピーした。
- config を `tail500` / `tail1000` replacement-only に変更した。
- push 前 booster 数を `SESSION_NOTES.md` に記録した。
- Jupytext で `.ipynb` を再生成し、古い exp161 content を消した。
- `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test`、`make validate-exp` を通した。
- Kaggle feature cache notebook を CPU / internet off で prepare/push した。
- feature cache 完了後、manifest の rows / wells / feature_count / SHA / elapsed を記録した。
- split train notebook (`lgb0`, `lgb1`, `lgb2`) を cache source 付きで prepare/push した。
- split train v1 の memory failure を確認し、variant ごとに必要な48列だけを読み込む構造へ修正した。
- split train v2 を完了し、variant/window 別 CV と exp148 差分を `result.md` / `metrics.json` / `experiment_summary.md` に記録した。
- exp166 は train-side rejected / no submit と判定した。
