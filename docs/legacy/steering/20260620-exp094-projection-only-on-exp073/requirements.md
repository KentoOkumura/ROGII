# 要件

## 依頼

`projection_only_on_exp073` を実装する。exp073 full replay deterministic ML anchor の保存済み OOF prediction に対して、参照元 public notebook `pilkwang/rogii-target-free-tvt-geosteering` の `TVT + Z - anchor` projection を後処理として移植し、再学習なしで有効性を監査する。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 入力 prediction は exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` の fold-out OOF に固定する。
- projection fit は `pred_tvt`, raw horizontal well の `MD/Z`, known prefix anchor `TVT_input/Z` のみを使い、評価 tail の `target_tvt` は scoring 以外に使わない。
- degree 3/4/5、blend beta 0.25/0.50/0.75、robust C 小さめを config grid として比較する。
- 全体 RMSE だけで採用せず、original GroupKFold 相当、well-hash fold、distance bucket、tail rank bucket、tail length bucket、near-continuity、prediction range、correction p95 を確認する。
- inference port は train-side guard 通過後に `inference.selected_variant` を固定するまで無効にする。
- 再現性: `docs/06_reproducibility.md` に従い、gzip は decompressed content SHA を主証拠として記録する。

## 受け入れ基準

- `experiments/exp094_projection_only_on_exp073/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook は設定確認、入力契約、projection grid 実行、出力 preview、metrics 保存をセル単位で追える。
- 補助スクリプトは exp073 OOF prediction と raw train well files から projection context を作り、variant metrics、fold metrics、bucket metrics、by-well metrics、best prediction、summary JSON を保存する。
- inference notebook は selected variant が null の間は submission を作らず、選択後だけ exp073 inference prediction に同じ projection を適用する。
- `make validate-exp EXP=exp094_projection_only_on_exp073` が通る。
- local full notebook 実行は行わず、必要な場合のみ明示 debug として扱う。
