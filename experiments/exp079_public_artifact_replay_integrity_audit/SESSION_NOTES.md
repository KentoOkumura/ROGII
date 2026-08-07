# exp079_public_artifact_replay_integrity_audit セッションノート

## 目的

公開 notebook route の候補を直接 submit する前に、外部生成物依存、static visible CSV、code competition rerun 互換、branch output、既存 anchor との差分を監査する。

## 現在の状態

- Route: pf_beam
- 状態: Kaggle audit completed
- CV: なし
- LB: なし
- Submit: なし

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp079_public_artifact_replay_integrity_audit
uv run python scripts/new_experiment.py --name exp079_public_artifact_replay_integrity_audit
```

実装後の検証:

```bash
uv run python scripts/validate_experiment.py --experiment exp079_public_artifact_replay_integrity_audit
uv run ruff check experiments/exp079_public_artifact_replay_integrity_audit/public_artifact_integrity_audit.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp079_public_artifact_replay_integrity_audit --notebook train --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp079_public_artifact_replay_integrity_audit --notebook inference --run-on-push --strict
uv run pytest tests/test_kaggle_notebooks.py
uv run python scripts/update_experiment_summary.py
```

- `validate_experiment`: pass.
- `ruff check`: pass.
- local smoke: `/tmp/exp079-smoke-artifacts` に audit summary / CSV / JSONL / README を保存し、Kaggle source 不在により想定通り `blocked_missing_required_sources`。
- `prepare_kaggle_notebooks` train / inference: pass。`kernel-metadata.json` に Pilkwang / ridge-sp kernel sources と Pilkwang / fleongg / Ravaghi dataset sources が入ることを確認。
- `pytest tests/test_kaggle_notebooks.py`: 3 passed.
- `update_experiment_summary.py`: 81 experiments.

Kaggle 実行:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp079_public_artifact_replay_integrity_audit --notebook train --kernel-id kentookumura/exp079-public-artifact-audit-train --title "exp079 public artifact audit train" --run-on-push --strict
kaggle kernels push -p experiments/exp079_public_artifact_replay_integrity_audit/kaggle/train
kaggle kernels pull kentookumura/exp079-public-artifact-audit-train -p /tmp/kaggle-pull/exp079-public-artifact-audit-train -m
timeout 180 kaggle kernels logs -f --interval 10 kentookumura/exp079-public-artifact-audit-train
kaggle kernels output kentookumura/exp079-public-artifact-audit-train -p /tmp/kaggle-output/exp079_public_artifact_replay_integrity_audit/train_v4
```

- Initial push with `kentookumura/exp079-public-artifact-replay-integrity-audit-train` failed with `SaveKernel` 400 because the slug was too long / title resolution was rejected. Canonical kernel id was shortened to `kentookumura/exp079-public-artifact-audit-train`.
- v1 failed because `sample_submission.csv` resolved to `/kaggle/working/data/raw/sample_submission.csv`.
- v2 completed but returned `blocked_missing_required_sources` because Kaggle CLI v2 mounted sources under `/kaggle/input/notebooks/...` and `/kaggle/input/datasets/...`, not `/kaggle/input/<slug>`.
- v3 completed with `audit_completed`, 28 candidate files, 2 notebook inspections, 120 pairwise distances.
- v4 is the正 result. It fixes duplicate `submission.csv` pairwise labels by using `source_name::label`.
- v4 log summary: `audit_completed`, missing required sources 0, candidate files 28, notebook inspections 2, pairwise distances 136.
- v4 output: `/tmp/kaggle-output/exp079_public_artifact_replay_integrity_audit/train_v4`.

## 変更点

- `.steering/20260618-exp079-public-artifact-replay-integrity-audit/` を作成し、要件、設計、タスクを記入した。
- `config.yaml` に Pilkwang / ridge-sp / fle3n-SP45-Koolbox placeholder と required input slug、branch file、anchor submission path を記入した。
- `public_artifact_integrity_audit.py` を追加し、次を保存できるようにした。
  - required input slug の存在
  - input file inventory / SHA
  - gzip decompressed content SHA
  - notebook metadata / input ref / risk pattern
  - candidate submission の sample 互換性、予測範囲、SHA
  - candidate-vs-anchor と branch 間の pairwise distance
- train / inference notebook を no-submit audit entrypoint に置き換えた。

## 再現性メモ

- seed policy: `no_rng_used`
- stochastic components: なし
- CPU/GPU runtime: CPU only。GPU なし。
- Kaggle kernel id / version: `kentookumura/exp079-public-artifact-audit-train` v4
- input SHA: `artifacts/exp079_public_artifact_replay_integrity_audit_summary.json` に保存済み
- feature schema SHA: 対象外
- feature content SHA: 対象外
- model manifest / model SHA: 対象外
- prediction SHA: candidate CSV の SHA として `exp079_public_artifact_replay_integrity_audit_submission_summary.csv` に保存済み
- submission SHA: candidate CSV の SHA として `exp079_public_artifact_replay_integrity_audit_submission_summary.csv` に保存済み
- rerun check: v3 / v4 とも `audit_completed`。v4 を正とする。

## 結果

- Missing required sources: 0
- Notebook inspections: 2
- Candidate files: 28
- Valid submission CSVs: 17
- Pairwise distances: 136
- Pilkwang final `submission.csv` SHA: `53986e96fdf30b311a3298bea51849e5ec5088aa007f8d36ceab001ffa76e07f`
- Pilkwang final is identical to `submission_projected_ridge_pf_pretrained_lgbm_base.csv` and `submission_projected_ridge_pf_pretrained_lgbm_w0.55.csv`.
- Pilkwang final vs projected ridge/PF projection RMSE: 1.299277767.
- Pilkwang final vs pretrained LGBM branch RMSE: 1.588006160.
- Pilkwang final vs model-package-only RMSE: 17.318521442.
- Pilkwang final vs ridge-sp final RMSE: 2.020019968.
- model package tiny gate vs base RMSE: gmax 0.003 = 0.005973822, 0.005 = 0.009956370, 0.010 = 0.019912740.
- Pilkwang notebook risk hits: `exact_match_or_override=38`, `mentions_sample_submission=26`, `writes_submission_csv=3`.
- ridge-sp notebook risk hits: `writes_submission_csv=1`, `reads_submission_csv=2`, `mentions_public_or_visible=1`.

## 次のアクション

1. `pilkwang_branch_decomposition` を実行し、final / projected ridge-PF / pretrained LGBM / model-package-only / gated candidates の寄与を整理する。
2. 提出するなら full final からではなく、branch diff と risk guard を読んで 1-2 candidate に絞る。
3. fle3n / SP45 / Koolbox 系の exact source slug を固定して追加監査する。
