# 要件

## 依頼

`projection_fadein_after_prefix` を実装する。`exp094_projection_only_on_exp073` の best projection correction をそのまま global beta で足すのではなく、known prefix 直後は beta 0 とし、`md_since` に応じて selected beta へ線形 fade-in する。

## 制約

- Route: `ml_model`
- 親実験: `exp094_projection_only_on_exp073`
- 入力 prediction は exp073 `gpu_repro_guard_dp_threads8` / `lgb_mean` の fold-out OOF に固定する。
- projection fit は exp094 と同じく `pred_tvt`, raw horizontal well の `MD/Z`, known prefix anchor `TVT_input/Z` のみを使い、評価 tail の `target_tvt` は scoring 以外に使わない。
- `md_since <= 250` は beta 0、`250-750` または `250-1000` で beta を 0 から selected beta へ線形 fade、以降 selected beta 固定にする。
- 候補は selected beta `0.50/0.75`、fade window `250-750/250-1000`、projection shape `degree4/c2` と `degree5/c1.5` に限定する。
- 全体 RMSE だけで採用せず、original fold、well-hash fold、distance 0-50 / 50-100 / 100-250、tail rank 0-99、tail length bucket、correction p95 を確認する。
- inference port は train-side guard 通過後に `inference.selected_variant` を固定するまで無効にする。
- 再現性: `docs/06_reproducibility.md` に従い、gzip は decompressed content SHA を主証拠として記録する。

## 受け入れ基準

- `experiments/exp096_projection_fadein_after_prefix/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook は設定確認、入力契約、fade-in projection grid 実行、出力 preview、metrics 保存をセル単位で追える。
- 補助スクリプトは row-wise `projection_beta_effective` と `projection_correction_applied` を保存し、variant metrics、fold metrics、bucket metrics、by-well metrics、best prediction、summary JSON を保存する。
- inference notebook は selected variant が null の間は submission を作らず、選択後だけ exp073 inference prediction に同じ fade-in projection を適用する。
- `.venv/bin/python -m py_compile experiments/exp096_projection_fadein_after_prefix/projection_fadein_after_prefix.py` が通る。
- `make validate-exp EXP=exp096_projection_fadein_after_prefix` が通る。
- local full notebook 実行は行わず、必要な場合のみ明示 debug として扱う。
