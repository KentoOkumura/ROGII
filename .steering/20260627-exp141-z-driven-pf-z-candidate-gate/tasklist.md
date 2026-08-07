# タスクリスト

## TODO

- Kaggle train package を作成し、metadata と bootstrap manifest を確認する。
- Kaggle train を実行して `likpf_mean`、`pf_z`、low-frequency gate、oracle headroom を比較する。
- 結果取得後に `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。
- 改善した場合だけ、同じ exp141 内で raw-test-compatible inference port を設計する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- `config.yaml` に exp072 cache 入力、route、gate grid、SHA 記録方針を明記した。
- `z_driven_pf_z_candidate_gate.py` に train-side posthoc audit を実装した。
- train/inference notebook を diagnostic 用 orchestration に差し替えた。
- `SESSION_NOTES.md`、`README.md`、`result.md` の初期記録を更新した。
