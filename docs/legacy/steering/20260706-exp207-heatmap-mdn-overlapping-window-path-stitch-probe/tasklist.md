# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/` を template から作成した。
- `config.yaml` を pf_beam route、exp202 path artifact、exp099 candidate cache、CPU diagnostic に更新した。
- `heatmap_mdn_overlapping_window_path_stitch_probe.py` に target-free beam stitch、coverage、oracle readout、SHA 記録を実装した。
- train notebook source を Jupytext percent 形式で追加した。
- inference notebook source を diagnostic-only no-submit guard として追加した。
- py_compile、ruff F821/E501、Jupytext train/inference conversion + `--test`、`make validate-exp` を通した。
- 書き出しなしの local helper smoke を 2 wells で通した。
- Kaggle train v1 を実行し、notebook kernelspec metadata 不足による `No kernel name found` 失敗を確認した。
- train / inference percent source に Python 3 kernelspec header を追加した。
- Kaggle train v2 を同じ kernel id `kentookumura/exp207-hmdn-path-stitch-train` へ push し、COMPLETE を確認した。
- Kaggle output を `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/kaggle/output/train_v2` に取得した。
- `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を結果ベースで更新した。
- exp207 は sparse artifact 診断として閉じ、dense window regeneration は `heatmap_mdn_dense_stride_window_path_regeneration_probe` の別 backlog として追加した。
