# exp081_pilkwang_branch_decomposition セッションノート

## 目的

`exp079_public_artifact_replay_integrity_audit` v4 の保存済み output から、Pilkwang final と branch candidate の寄与を分解し、提出候補を絞る。

## 現在の状態

- Route: pf_beam
- 状態: decomposition_completed
- CV: なし
- LB: なし
- Submit: なし

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp081_pilkwang_branch_decomposition
uv run python scripts/new_experiment.py --name exp081_pilkwang_branch_decomposition
uv run python experiments/exp081_pilkwang_branch_decomposition/pilkwang_branch_decomposition.py
uv run ruff check experiments/exp081_pilkwang_branch_decomposition/pilkwang_branch_decomposition.py
uv run python scripts/validate_experiment.py --experiment exp081_pilkwang_branch_decomposition
uv run ruff format --check experiments/exp081_pilkwang_branch_decomposition/pilkwang_branch_decomposition.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp081_pilkwang_branch_decomposition --notebook train --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp081_pilkwang_branch_decomposition --notebook inference --strict
uv run pytest tests/test_kaggle_notebooks.py
```

- `validate_experiment`: pass.
- `ruff check`: pass.
- `ruff format --check`: pass.
- `prepare_kaggle_notebooks` train / inference: pass.
- `pytest tests/test_kaggle_notebooks.py`: 3 passed.

## 変更点

- `docs/legacy/steering/20260619-exp081-pilkwang-branch-decomposition/` を作成し、要件、設計、タスクを記入した。
- `config.yaml` に exp079 v4 output path、Pilkwang branch roles、anchor labels、candidate policy を記入した。
- `pilkwang_branch_decomposition.py` を追加した。
- train / inference notebook を audit entrypoint に更新した。
- artifacts に branch summary、role summary、anchor comparison、candidate decisions、summary JSON を保存した。

## 再現性メモ

- seed policy: no_rng_used
- stochastic components: なし
- CPU/GPU runtime: CPU only。GPU なし。
- Kaggle kernel id / version: 未実行。ローカル二次解析。
- input SHA: summary `1b3b6c1ca580bbf8888eefe6b0d9c4f628d5b21a2b348dd5cd085a1254a6dd3b`、submission summary `a61819e47b1f38f0ea1ae7c860971d2c7e0f847d1894b6cf653d7e26b5222401`、pairwise `a77d61295112851931b05a90540c6d89fa630434f5a14abc6dbd0cb8dc6c043d`
- feature schema / content SHA: 対象外
- model manifest / model SHA: 対象外
- prediction SHA / submission SHA: exp079 の candidate SHA を `exp081_pilkwang_branch_decomposition_branch_summary.csv` と decision table に再掲。
- rerun check: script rerun で同一 summary を再生成。

## 結果

- Candidate count: 16
- Valid candidate count: 16
- Shortlist count: 6
- Submit candidate count: 2
- Submit candidate rank 1: `submission_projected_ridge_pf_projection_d4_b075_raw.csv`。vs final RMSE 1.4422981136、vs ridge-sp RMSE 1.1301896874。
- Submit candidate rank 2: `submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv`。vs final RMSE 0.1443641963、vs ridge-sp RMSE 1.9410101494。
- final / base は同一 prediction。`w0.55` も pairwise では final と RMSE 0。
- model package tiny gate は final から RMSE 0.005974 / 0.009956 / 0.019913 の微小差。
- pretrained LGBM 単独は vs final RMSE 1.588006、vs ridge-sp RMSE 3.205317。
- model-package-only は vs final RMSE 17.318521 で reject。
- exp027 / exp073 / exp063 anchor との pairwise は exp079 v4 に保存されておらず、`missing_pairwise` と記録した。
- row-level diff は候補 CSV 本体がローカル output にないため未実施。

## 次のアクション

1. submit するなら rank 1 / rank 2 の候補 CSV 本体を再取得し、submit-check と row-level guard を行う。
2. SP45 / fle3n / Koolbox 系の exact source slug を固定し、公開 notebook followup audit へ進む。
