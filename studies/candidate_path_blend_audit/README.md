# Candidate path blend audit

保存済み train-side OOF / pseudo-tail candidate を ID で揃え、単純平均と
well-outer-fold で学習した凸結合を比較する再現用 study です。

- 大きい入力と作業用 memmap は `/tmp/candidate_path_blend_audit_work` に置く。
- Git 管理するのはスクリプトと小さい集計 CSV / JSON だけ。
- 50/50、等重み multi-way は target-free な固定診断。
- 最適重みは同一 OOF 上の値を採用根拠にせず、5-fold の train wells で重みを求めて
  held-out wells に適用した `crossfit_*` を主に読む。

実行:

```bash
uv run python studies/candidate_path_blend_audit/run_blend_audit.py
uv run python studies/candidate_path_blend_audit/run_robustness_readout.py
```

主な出力:

- `candidate_metrics.csv`: 81候補パスと13 model/selector出力の単体RMSE。
- `pair_blends.csv`: 全4,371ペアの50/50、full-OOF診断、well cross-fit凸結合。
- `equal_multiway_blends.csv`: 24本shortlist、サイズ2–6の均等平均190,026組。
- `crossfit_triple_blends.csv`: 上位20本、全1,140 tripleのwell cross-fit凸結合。
- `blend_scope_metrics.csv` / `blend_fold_metrics.csv` / `blend_well_risk.csv`: 有力blendの距離帯、hidden-like、fold、well別risk。
- `non_stacking_scope_summary.json`: HMM+LGB系exp221/234/240とmodel outputsを主評価から外した78 pathのcurrent summary。固定`exp226 + w500`とraw-path cross-fitを分けて記録する。

結論と採用可否は[`docs/surveys/candidate_path_blend_audit_20260716.md`](../../docs/surveys/candidate_path_blend_audit_20260716.md)を参照する。
