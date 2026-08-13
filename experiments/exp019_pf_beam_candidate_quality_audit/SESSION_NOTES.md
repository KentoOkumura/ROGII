# exp019_pf_beam_candidate_quality_audit セッションノート

## 目的

バックログ先頭の `pf_beam_candidate_quality_audit` を実装する。`exp015` で
PF/beam add-only features が CV 14.442743 まで悪化したため、再学習せずに
direct candidate quality、scale/confidence/GR gap、hurt/help wells を診断する。

## 現在の状態

- 状態: 完了
- 親実験: `exp015_public_pf_beam_scale_selector_features`
- Raw CV anchor: `exp012/exp013 lightgbm_no_gr` 13.549257
- exp015 PF/beam feature CV: 14.442743

## コマンドログ

- 2026-06-05: `uv run python scripts/new_steering.py --experiment exp019_pf_beam_candidate_quality_audit` で steering docs を作成。
- 2026-06-05: `uv run python scripts/new_experiment.py --name exp019_pf_beam_candidate_quality_audit --source experiments/exp018_candidate_distribution_router` で診断実験を作成。
- 2026-06-05: notebook 名と `settings.py` を exp019 用に更新。
- 2026-06-05: `audit_pf_beam_candidate_quality.py` を実装。
- 2026-06-05: ローカルで smoke/full audit を開始したが、AGENTS.md の Kaggle 実行方針に反するため停止。ローカル結果は採用しない。
- 2026-06-05: Kaggle bundle に入るよう `pf_beam_baseline.py` と `exp015_source_config.yaml` を同梱し、train notebook entrypoint を exp019 用に修正。
- 2026-06-05: `runtime.kaggle.kernel_sources` を `kernel-metadata.json` に反映するよう `scripts/prepare_kaggle_notebooks.py` を更新。
- 2026-06-05: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp019_pf_beam_candidate_quality_audit --notebook train --title 'exp019 pf beam candidate quality audit train' --run-on-push --strict` で Kaggle train package を生成。
- 2026-06-05: `kaggle kernels push -p experiments/exp019_pf_beam_candidate_quality_audit/kaggle/train` の初回 push は exp015 source slug が無効だったため、`kentookumura/exp015-pf-beam-train` に修正。
- 2026-06-05: 修正版を version 2 として push。URL: https://www.kaggle.com/code/kentookumura/exp019-pf-beam-candidate-quality-audit-train
- 2026-06-05: `kaggle kernels status kentookumura/exp019-pf-beam-candidate-quality-audit-train` は Kaggle API 500。output 取得も成果物なし。監視は未完了。
- 2026-06-05: ユーザー確認で version 2 failed / version 1 running と判明。version 2 log は `argparse` が notebook kernel 引数 `-f ... --HistoryManager.hist_file=:memory:` を受けて `SystemExit: 2`。`parse_known_args()` に修正。
- 2026-06-05: `uv run python -m py_compile ...`、`uv run ruff check ...`、`uv run python scripts/validate_experiment.py --experiment exp019_pf_beam_candidate_quality_audit` が通過。
- 2026-06-05: 修正版を version 3 として push。URL: https://www.kaggle.com/code/kentookumura/exp019-pf-beam-candidate-quality-audit-train
- 2026-06-06: ユーザー確認で version 1 completed / version 3 job 不明と判明。current output log は `exp015_source_config.yaml` が bundle に入っておらず `FileNotFoundError`。`scripts/prepare_kaggle_notebooks.py` を top-level `.yaml/.yml` も bundle するよう修正。
- 2026-06-06: `exp015_source_config.yaml` が Kaggle train package と notebook bootstrap manifest に含まれることを確認。
- 2026-06-06: version 4 を push。URL: https://www.kaggle.com/code/kentookumura/exp019-pf-beam-candidate-quality-audit-train
- 2026-06-06: version 4 output を取得。Kaggle log は 100/773 から 700/773 wells まで進捗し、全 artifact と `metrics.json` を生成。実行時間は約 5,030 秒。
- 2026-06-06: 小さい artifact と log を `experiments/exp019_pf_beam_candidate_quality_audit/artifacts/` に保存し、`metrics.json` を更新。

## 変更点

- `exp015` の PF/beam feature builder を import し、train CSV から direct PF/beam candidates を再計算する。
- `exp013` raw LightGBM OOF と row 単位で比較する。
- `pf_mean`、`pf_best`、scale 別 `pf_s3/s5/s8/s12`、hold blend、last anchor、recent linear を比較する。
- 距離 bucket、best scale、confidence、GR missing、eval length、Z span、trajectory steepness 別に RMSE を出す。
- `exp015` の well metrics を結合し、PF feature model の top hurt/help wells を出す。

## Artifacts

- `artifacts/pf_beam_candidate_metrics.csv`
- `artifacts/pf_beam_well_deltas.csv`
- `artifacts/pf_beam_scale_diagnostics.csv`
- `artifacts/pf_beam_top_hurt_help.csv`
- `artifacts/pf_beam_candidate_quality_summary.json`
- `metrics.json`

## 結果

- Raw LightGBM no-GR: 13.549257
- Best direct candidate: `raw_lightgbm_no_gr` 13.549257
- Best PF-derived full-row candidate: `pf_hold_mean_blend` 19.142388、raw から +5.593130 悪化
- `pf_best`: 114.654448、raw から +101.105190 悪化
- `exp015` PF feature model mean well delta vs control: +0.648761
- PF feature model improved 310/773 wells but hurt more overall.
- PF direct candidates did not beat raw in confidence / best-scale / GR-missing / long-eval / high-Z-span / steep-trajectory groups.
- Non-PF controls beat raw near the hidden start: rows 0-49 `recent_linear` 0.796588 and `last_anchor` 0.960110 vs raw 3.231596; rows 50-249 `recent_linear` 3.615510 vs raw 3.829747.

## 次のアクション

1. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` に反映する。
2. PF/beam 再投入は止め、次は距離 bucket の raw residual bias/variance と near-row damping/training weight を診断する。
