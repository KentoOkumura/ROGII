# 要件

## 依頼

`seedbag_anchor_model_diff_distance_gate` を実装する。

exp059 の PF/Beam-vs-exp052/054 model-diff raw branch は Public LB 11.878 まで改善したが、exp054 seed-bag pseudo-tail anchor の Public LB 11.856 には届かなかった。exp059 は遠距離 bucket `rows_2500_plus` が弱いため、exp054 を主 anchor として固定し、exp059 raw model-diff 予測との差分だけを距離 bucket 別に小さく混ぜる候補を作る。

## 制約

- Route: `ml_model`
- 親実験: `exp059_pf_model_diff_foldsafe_surface_shrink`
- seed-bag anchor: `exp054_pseudo_tail_seed_bagging_inference_submit`
- model-diff correction source: `exp059` の `lgbm_capacity_pf_model_diff_foldsafe_raw`
- `rows_2500_plus` の悪化を最優先で防ぐ。
- PF/Beam raw prediction を直接置換候補として使わない。
- fold-safe source prediction と well-level split を維持する。

## 受け入れ基準

- `experiments/exp061_seedbag_anchor_model_diff_distance_gate/` が作成され、`config.yaml` の `experiment.route` が `ml_model` である。
- train 監査に `seedbag_distance_gate` postprocess があり、少なくとも far alpha 0 の profile と小さい global alpha profile を比較できる。
- inference helper が config の `selected_postprocess: seedbag_distance_gate` と `selected_gate_profile` を使って、hidden branch で `exp054_source + alpha(distance) * (raw_model_diff - exp054_source)` を生成する。
- `python -m py_compile`、`ruff check`、`validate_experiment.py` が通る。
