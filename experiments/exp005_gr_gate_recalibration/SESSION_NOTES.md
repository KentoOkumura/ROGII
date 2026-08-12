# exp005_gr_gate_recalibration セッションノート

## 目的

`exp004_gr_gating` の Public LB 乖離を踏まえ、GR coverage gate を再設計する。exp002 all-GR を default とし、no-GR へ倒す条件を soft blend または strict coverage にすることで、visible/public 寄り well の悪化を抑えながら CV 改善が残るか検証する。

## 現在の状態

- 状態: Kaggle inference / submission 完了
- 親実験: `exp004_gr_gating`
- LB anchor: `exp002_drift_minimal` CV 14.124569 / public LB 12.533
- comparison: `exp004_gr_gating` CV 13.932968 / public LB 12.730
- selected candidate: `gate_low_gr_strict_hard`
- CV: 13.936732
- LB: public 12.579 (`ref=53249562`)

## コマンドログ

- 2026-06-01: `make new-steering EXP=exp005_gr_gate_recalibration EXTRA_ARGS="--title gr-gate-recalibration"` で steering docs を作成。
- 2026-06-01: `make new-exp EXP=exp005_gr_gate_recalibration SOURCE=experiments/exp004_gr_gating` で exp004 から実験を作成。
- 2026-06-01: notebook ファイル名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics、steering docs を exp005 用に更新。
- 2026-06-01: exp004 OOF control rows から hard gate 候補を再構成し、`gate_low_gr_strict_hard` の推定 CV 13.936732 と visible `000d7d20` guard を確認。
- 2026-06-01: `make validate-exp EXP=exp005_gr_gate_recalibration` が通過。
- 2026-06-01: `.venv/bin/ruff check experiments/exp005_gr_gate_recalibration/baseline.py` が通過。
- 2026-06-01: `make prepare-kaggle-notebooks EXP=exp005_gr_gate_recalibration EXTRA_ARGS="--notebook train --run-on-push --title 'exp005 gr gate recalibration train' --strict"` が通過。
- 2026-06-01: `make prepare-kaggle-notebooks EXP=exp005_gr_gate_recalibration EXTRA_ARGS="--notebook inference --run-on-push --title 'exp005 gr gate recalibration inference' --strict"` が通過。
- 2026-06-01: `python -m py_compile experiments/exp005_gr_gate_recalibration/baseline.py experiments/exp005_gr_gate_recalibration/settings.py` が通過。
- 2026-06-01: `.venv/bin/pytest` が通過。9 tests passed。
- 2026-06-01: `.venv/bin/python scripts/record_experiment.py --experiment exp005_gr_gate_recalibration --status scaffold_completed ...` で `metrics.json` と `experiment_summary.md` を更新。
- 2026-06-01: `KAGGLE_DIRECTION.md` のアイデアバックログから実装済みの GR coverage gate 見直しを外し、現在の重点に exp005 の Kaggle CV 待ちを追加。
- 2026-06-01: `make push-kaggle-train EXP=exp005_gr_gate_recalibration` は sandbox 内 DNS 制限で失敗後、承認済み escalated `kaggle kernels push -p experiments/exp005_gr_gate_recalibration/kaggle/train` で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp005-gr-gate-recalibration-train
- 2026-06-01: `kaggle kernels status kentookumura/exp005-gr-gate-recalibration-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp005-gr-gate-recalibration-train -p /tmp/kaggle-output/exp005_gr_gate_recalibration/train` で output と kernel log を取得。
- 2026-06-01: Kaggle output の `metrics.json`、`artifacts/ablation_metrics.csv`、`well_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、train log を `experiments/exp005_gr_gate_recalibration/` に反映。
- 2026-06-01: `make prepare-kaggle-notebooks EXP=exp005_gr_gate_recalibration EXTRA_ARGS="--notebook inference --run-on-push --title 'exp005 gr gate recalibration inference' --strict"` で inference notebook を最新 config から再生成。
- 2026-06-01: `kaggle kernels push -p experiments/exp005_gr_gate_recalibration/kaggle/inference` を承認済み escalated 実行で成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp005-gr-gate-recalibration-inference
- 2026-06-01: `kaggle kernels status kentookumura/exp005-gr-gate-recalibration-inference` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp005-gr-gate-recalibration-inference -p /tmp/kaggle-output/exp005_gr_gate_recalibration/inference` で `submission.csv`、`inference_well_summaries.csv`、kernel log を取得。
- 2026-06-01: `.agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp005_gr_gate_recalibration/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。
- 2026-06-01: `make submit-check EXP=exp005_gr_gate_recalibration SUBMISSION=/tmp/kaggle-output/exp005_gr_gate_recalibration/inference/submission.csv` は PASS。外部 `/tmp` パス表示で落ちていた `scripts/validate_submission.py` を display-only 修正した。
- 2026-06-01: visible duplicate well sanity を `artifacts/visible_submission_well_comparison.csv` に保存。aggregate visible RMSE は exp002 7.916353、exp004 7.948310、exp005 7.916353。
- 2026-06-01: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp005-gr-gate-recalibration-inference -v 1 -f submission.csv -m "exp005_gr_gate_recalibration gate_low_gr_strict_hard CV 13.936732"` を承認済み escalated 実行で成功。
- 2026-06-01: submission ref `53249562` は `SubmissionStatus.COMPLETE`、public LB 12.579。
- 2026-06-01: `.venv/bin/python scripts/record_submission.py --experiment exp005_gr_gate_recalibration --file /tmp/kaggle-output/exp005_gr_gate_recalibration/inference/submission.csv --cv 13.936732 --public-lb 12.579 ...` で `SUBMISSIONS.md` に v005 を記録。
- 2026-06-01: `.venv/bin/python scripts/record_experiment.py --experiment exp005_gr_gate_recalibration --status completed --cv 13.936732 --public-lb 12.579 ...` で `metrics.json` と `experiment_summary.md` を更新。

## 変更点

- `control_exp002_all`: exp002 と同じ all-GR residual model。
- `control_exp003_no_gr`: exp003 selected variant と同じ no-GR residual model。
- `control_exp004_low_gr_any_hard`: exp004 selected gate の再実行。`prefix_gr_missing_rate >= 0.35` または `eval_gr_missing_rate >= 0.40` で no-GR へ 100% gate。
- `gate_low_gr_any_soft_050`: exp004 と同じ gate 条件で no-GR weight を 0.5 に下げる。
- `gate_low_gr_strict_hard`: `prefix_gr_missing_rate >= 0.35` かつ `eval_gr_missing_rate >= 0.40` の場合だけ no-GR へ 100% gate。
- `gate_low_gr_strict_soft_050`: strict 条件で no-GR weight を 0.5 に下げる。

## 事前確認

exp004 OOF control rows から hard gate を再構成すると、`control_exp004_low_gr_any_hard` は CV 13.932968、`gate_low_gr_strict_hard` は推定 CV 13.936732。strict hard はわずかに CV が悪いが、visible `000d7d20` の `prefix_gr_missing_rate` が 0.306519 で threshold 0.35 未満のため no-GR routing から外れる。

## 結果

| Variant | CV | exp002 差分 | Gate Weight | Gated Wells | Eval Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `control_exp003_no_gr` | 13.882944 | -0.241625 | - | - | - |
| `control_exp004_low_gr_any_hard` | 13.932968 | -0.191601 | 1.0 | 297 | 1,506,134 |
| `gate_low_gr_strict_hard` | 13.936732 | -0.187837 | 1.0 | 214 | 1,115,013 |
| `gate_low_gr_any_soft_050` | 13.998291 | -0.126278 | 0.5 | 297 | 1,506,134 |
| `gate_low_gr_strict_soft_050` | 14.007188 | -0.117381 | 0.5 | 214 | 1,115,013 |
| `control_exp002_all` | 14.124569 | 0.000000 | - | - | - |

`gate_low_gr_strict_hard` は selected として CV 13.936732。exp004 selected gate の再現 `control_exp004_low_gr_any_hard` より 0.003764 悪いが、visible `000d7d20` は gate weight 1.0 から 0.0 へ戻り、public-visible guard の狙いは満たした。

## Inference / Submission

| Well | exp004 Gate | exp005 Gate | exp002 RMSE | exp004 RMSE | exp005 RMSE | exp005 - exp004 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `000d7d20` | 1.0 | 0.0 | 7.789073 | 7.908222 | 7.789073 | -0.119149 |
| `00bbac68` | 0.0 | 0.0 | 9.393623 | 9.393623 | 9.393623 | 0.000000 |
| `00e12e8b` | 0.0 | 0.0 | 5.356808 | 5.356808 | 5.356808 | 0.000000 |

Aggregate visible RMSE: exp002 7.916353、exp004 7.948310、exp005 7.916353。

Public LB は 12.579。exp004 12.730 より改善し、exp003 12.852 も上回ったが、exp002 12.533 には届かない。

## 次のアクション

1. LB anchor は exp002 のまま維持する。
2. 次は exp002 / exp005 の OOF hard well 解析へ進み、GR gate ではなく failure pattern に基づく改善候補を作る。
