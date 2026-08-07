# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp061 steering docs を作成した。
- exp059 から `experiments/exp061_seedbag_anchor_model_diff_distance_gate/` を作成した。
- `seedbag_distance_gate` postprocess を train audit に追加した。
- inference helper を `selected_postprocess: seedbag_distance_gate` に対応させた。
- config に `near_mid_a0p25_far0`、`near_mid_a0p50_far0`、`global_a0p25` の 3 profile を追加した。
- `python -m py_compile`、`uv run ruff check`、`validate_experiment.py` を通した。
- Kaggle train / inference package を作成した。
- `experiment_summary.md` に exp061 を追加した。
- Kaggle train v1 が完了し、output を同期した。
- selected candidate を `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0` に更新した。
- Kaggle inference package を再作成し、hidden branch output と submission を検証した。
- Code submission ref `53581056` が完了し、Public LB 11.826 を記録した。
