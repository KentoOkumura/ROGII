# 要件

## 依頼

`beam_topk_bimodal_gate_posthoc_audit` バックログを実装する。exp173 の保存済み Beam top-K path / cost / posterior diagnostics を使い、二峰性や低 cost gap がある row に限定して posterior / top2 / weighted mean へ置換する no-training posthoc gate を確認する。

## 制約

- Route: `pf_beam`
- 親実験: `exp173_beam_topk_path_posterior_audit`
- 追加学習、Beam 再生成、PF 再生成、GPU 使用、inference port、submission 作成はしない。
- 入力は exp173 Kaggle output の `topk_diagnostics.csv.gz`、`topk_paths.csv.gz`、`candidate_wide.csv.gz`、`candidate_metrics.csv` に限定する。
- Gate は `top1_top2_sep`、`top2_cost_gap_per_row`、`topk_entropy`、`topk_spread` の target-free 分位点と AND 条件だけで作る。
- true TVT は scoring、changed subset、bucket、worst-well regression の評価にだけ使い、gate threshold の決定には使わない。
- 再現性は `docs/06_reproducibility.md` に従い、gzip 生成物は decompressed SHA を主証拠として記録する。

## 受け入れ基準

- `experiments/exp177_beam_topk_bimodal_gate_posthoc_audit/` に config、監査 helper、Jupytext train/inference script、notebook、記録ファイルがある。
- `config.yaml` の `experiment.route` は `pf_beam`、status は実装済み pending train、kernel source は exp173 train output を参照する。
- train notebook は入力 artifact check、gate policy、実行コスト、run、metrics 表示をセルで追える。
- `py_compile`、ruff `F821,F401,E501`、Jupytext 変換/`--test`、`make validate-exp` が通る。
- Kaggle push 前の計算規模として、Beam regeneration 0、LightGBM config 0、fold 0、booster 0、control 再学習なしを `SESSION_NOTES.md` に記録する。
