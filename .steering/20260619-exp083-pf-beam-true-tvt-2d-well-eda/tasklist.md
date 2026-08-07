# タスクリスト

## TODO

- Kaggle train notebook を実行し、PNG / manifest / summary を取得する。
- 実行後に `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md` を更新する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements / design / tasklist を exp072 input cache 前提で記入した。
- `config.yaml` を `pf_beam` route、親 `exp072`、anchor `exp073` として設定した。
- `pf_beam_true_tvt_eda.py` を追加した。
- exp072 型の `target` delta と `*_d` 候補を TVT 空間へ戻す実装を追加した。
- notebook を train-side EDA / inference no-op に更新した。
- 合成 exp072 型 CSV で target 復元、well summary、metrics 出力を確認した。
- `ruff check` と `py_compile` を通した。
- `validate_experiment.py` を通した。
- Kaggle train package を `kentookumura/exp083-pfbeam-true-tvt-eda-train` として生成し、metadata の exp072 kernel source を確認した。
- Kaggle inference no-op package を生成した。
