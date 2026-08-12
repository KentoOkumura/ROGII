# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp212_heatmap_mdn_full_grid_path_generation_probe` を exp210 から作成した。
- exp210 の covered-row table を full-grid table generation に変更した。
- `coverage_flag` / `fallback_flag` / `fill_method` と full-grid contract metrics を追加した。
- 再現性設計を `design.md` に記入した。
- synthetic smoke で full-grid 補間・外挿、coverage/fallback 集計、contract metrics、oracle readout の基本動作を確認した。
- `py_compile`、`ruff --select F821,E501`、Jupytext conversion/test、`make validate-exp` を通した。
- Kaggle train package を strict mode で生成し、GPU false / internet false / exp208+exp099 kernel sources を確認した。
- Kaggle train v1 を実行し、kernel `kentookumura/exp212-hmdn-full-grid-path-generation-train` version 1 が `COMPLETE` になったことを確認した。
- Kaggle output を `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/kaggle/output/train_v1` に取得した。
- Full-grid contract metrics、oracle readout、SHA を `metrics.json`、`result.md`、`SESSION_NOTES.md` に記録した。
- full-grid contract は通ったが fallback unique row rate が `0.5699083691839485` のため、当時は exp204 系を guarded selector candidate follow-up に限定すると判断した。
- 後続 exp215 でも生成 path 自体の弱さが残ったため、heatmap 由来 path 生成 route は closed/rejected とした。
