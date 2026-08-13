# exp015_public_pf_beam_scale_selector_features セッションノート

## 目的

`exp013_model_diversity_or_postprocess` を親に、`exp012/exp013` の raw `lightgbm_no_gr` anchor へ fold-safe な PF/beam scale selector features を追加し、Kaggle full CV で比較する。

## 現在の状態

- 状態: 完了
- 親実験: `exp013_model_diversity_or_postprocess`
- raw model anchor: `exp012/exp013 lightgbm_no_gr` CV 13.549257
- selected variant before CV: `pf_beam_no_gr`
- CV: `pf_beam_no_gr` 14.442743
- best variant by CV: `control_lightgbm_no_gr` 13.549257
- LB: 未提出

## コマンドログ

- 2026-06-04: `uv run python scripts/new_steering.py --experiment exp015_public_pf_beam_scale_selector_features` で steering docs を作成。
- 2026-06-04: `docs/legacy/steering/20260604-exp015-public-pf-beam-scale-selector-features/{requirements.md,design.md,tasklist.md}` に仮説、設計、タスクを記入。
- 2026-06-04: `uv run python scripts/new_experiment.py --name exp015_public_pf_beam_scale_selector_features --source experiments/exp013_model_diversity_or_postprocess` で exp013 から実験を作成。
- 2026-06-04: train / inference notebook を exp015 名にリネームし、`settings.py` の `EXPERIMENT_NAME` を更新。
- 2026-06-04: `config.yaml` を exp015 用に置換し、`control_lightgbm_no_gr` と `pf_beam_no_gr` の 2 variants を定義。
- 2026-06-04: `baseline.py` に deterministic PF/beam feature generator を追加。
- 2026-06-04: `uv run python scripts/validate_experiment.py --experiment exp015_public_pf_beam_scale_selector_features` が通過。
- 2026-06-04: `uv run python -m py_compile experiments/exp015_public_pf_beam_scale_selector_features/baseline.py experiments/exp015_public_pf_beam_scale_selector_features/settings.py` が通過。
- 2026-06-04: `uv run ruff check experiments/exp015_public_pf_beam_scale_selector_features/baseline.py experiments/exp015_public_pf_beam_scale_selector_features/settings.py` が通過。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp015_public_pf_beam_scale_selector_features --notebook train --strict` が通過。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp015_public_pf_beam_scale_selector_features --notebook inference --strict` が通過。
- 2026-06-04: feature sanity check で control 25 features、PF/beam 82 features、`pf_beam_*` 57 features を確認。
- 2026-06-04: `uv run pytest` が通過。9 tests passed。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp015_public_pf_beam_scale_selector_features --notebook train --run-on-push --title "exp015 pf beam train" --strict` で train package を再生成。
- 2026-06-04: `kaggle kernels push -p experiments/exp015_public_pf_beam_scale_selector_features/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp015-pf-beam-train
- 2026-06-04: `kaggle kernels status kentookumura/exp015-pf-beam-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-04: `kaggle kernels output kentookumura/exp015-pf-beam-train -p /tmp/kaggle-output/exp015_public_pf_beam_scale_selector_features/train` で output と kernel log を取得。
- 2026-06-04: Kaggle output の `metrics.json`、小さい `artifacts/*.csv`、train log を `experiments/exp015_public_pf_beam_scale_selector_features/` に反映。`row_oof_predictions.csv` は 1.1GB のため実験ディレクトリには常設しない。
- 2026-06-05: `/tmp/kaggle-output/exp015_public_pf_beam_scale_selector_features/train/artifacts/row_oof_predictions.csv` は後続入力として使っていないため削除。必要なら `kaggle kernels output kentookumura/exp015-pf-beam-train -p /tmp/kaggle-output/exp015_public_pf_beam_scale_selector_features/train` で再取得する。

## 変更点

- `pf_beam_*` features:
  - scale-specific candidate path score / shift / slope / DTW cost
  - best / second score、confidence、entropy、hold weight
  - candidate path mean/std/range と recent slope path との差
  - eval length / Z span selector
- feature generation は `MD`、`Z`、`GR`、known `TVT_input` prefix、paired typewell GR のみを使う。

## 結果

| Variant | Feature set | CV | mean fold RMSE |
| --- | --- | ---: | ---: |
| `control_lightgbm_no_gr` | `no_gr_signal` | 13.549257 | 13.521370 |
| `pf_beam_no_gr` | `no_gr_signal_plus_pf_beam` | 14.442743 | 14.401690 |

PF/beam add-only features は control から +0.893486 悪化したため採用しない。

## 次のアクション

1. `backlog/KAGGLE_DIRECTION.md` から実装済み `public_pf_beam_scale_selector_features` を削除する。
2. 次の最優先を `public_postprocess_ablation` に上げる。
3. PF/beam は再投入する場合でも、add-only ではなく candidate quality audit / feature pruning / router に限定する。
