# exp074_compact_tracker_surface_lgbm_candidate_audit セッションノート

## 現在の状態

- status: `kaggle_train_inference_completed_pending_optional_lb`
- route: `ml_model`
- parent: `exp070_gpu_reproducibility_guard_for_exp063`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- purpose: exp070 の 65-feature compact tracker surface を、exp063 full replay reproducibility ではなく LB 候補として監査する。

## 実装内容

- `.steering/20260616-exp074-compact-tracker-surface-lgbm-candidate-audit/` を作成。
- exp070 を source として exp074 を作成。
- notebook 名を exp074 に正規化。
- `exp063_reproducibility_guard.py` を `compact_tracker_surface_audit.py` にリネーム。
- 出力 prefix を `compact_tracker_surface_audit` に変更。
- `config.yaml` を candidate audit 目的に更新。
- train は exp063 compact tracker train artifact を固定入力として LightGBM を再学習する。
- inference は exp074 train の saved boosters を読み、raw test から compact tracker features を再生成して `submission.csv` を作る。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp074_compact_tracker_surface_lgbm_candidate_audit
uv run python scripts/new_experiment.py --name exp074_compact_tracker_surface_lgbm_candidate_audit --source experiments/exp070_gpu_reproducibility_guard_for_exp063
```

実装後の検証:

```bash
uv run python -m py_compile experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/compact_tracker_surface_audit.py experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/public_notebook_replay_audit.py experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/settings.py
uv run ruff check experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/compact_tracker_surface_audit.py experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/public_notebook_replay_audit.py experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/settings.py
uv run python -m json.tool experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/exp074_compact_tracker_surface_lgbm_candidate_audit_train.ipynb
uv run python -m json.tool experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/exp074_compact_tracker_surface_lgbm_candidate_audit_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp074_compact_tracker_surface_lgbm_candidate_audit
```

Kaggle train:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp074_compact_tracker_surface_lgbm_candidate_audit --notebook train --kernel-id kentookumura/exp074-compact-tracker-lgbm-audit-train --title "exp074 compact tracker lgbm audit train" --run-on-push --strict
kaggle kernels push -p experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/kaggle/train
kaggle kernels pull kentookumura/exp074-compact-tracker-lgbm-audit-train -p /tmp/kaggle-pull/exp074-compact-tracker-lgbm-audit-train-v1 -m
kaggle kernels logs kentookumura/exp074-compact-tracker-lgbm-audit-train
kaggle kernels output kentookumura/exp074-compact-tracker-lgbm-audit-train -p /tmp/kaggle-output/exp074_compact_tracker_surface_lgbm_candidate_audit/train_v1
```

Kaggle inference:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp074_compact_tracker_surface_lgbm_candidate_audit --notebook inference --kernel-id kentookumura/exp074-compact-tracker-lgbm-audit-infer --title "exp074 compact tracker lgbm audit infer" --run-on-push --strict
kaggle kernels push -p experiments/exp074_compact_tracker_surface_lgbm_candidate_audit/kaggle/inference
kaggle kernels pull kentookumura/exp074-compact-tracker-lgbm-audit-infer -p /tmp/kaggle-pull/exp074-compact-tracker-lgbm-audit-infer-v1 -m
kaggle kernels logs kentookumura/exp074-compact-tracker-lgbm-audit-infer
kaggle kernels output kentookumura/exp074-compact-tracker-lgbm-audit-infer -p /tmp/kaggle-output/exp074_compact_tracker_surface_lgbm_candidate_audit/infer_v1
```

## 結果

- 静的検証:
  - `py_compile`: PASS
  - `ruff check`: PASS
  - train notebook JSON validation: PASS
  - inference notebook JSON validation: PASS
  - `validate_experiment.py`: PASS
- Kaggle train v1:
  - kernel: `kentookumura/exp074-compact-tracker-lgbm-audit-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp074-compact-tracker-lgbm-audit-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp074-compact-tracker-lgbm-audit-train-v1`
  - status: complete
  - output: `/tmp/kaggle-output/exp074_compact_tracker_surface_lgbm_candidate_audit/train_v1`
  - rows / wells / features: 3,783,989 / 773 / 65
  - elapsed: 6,925.739 sec
  - `lgb_mean` CV: 9.73150619943287
  - feature source SHA: `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`
  - OOF prediction SHA: `09ccb9edd59cd50057da0ee7738229749996219708f36e6c45f870d0efd026a5`
  - model manifest SHA: `e379b078b4fdfaceb39c25fcc8246cab221ab16038e3dac4f8c2b74360197ece`
- Kaggle inference v1:
  - kernel: `kentookumura/exp074-compact-tracker-lgbm-audit-infer`
  - URL: `https://www.kaggle.com/code/kentookumura/exp074-compact-tracker-lgbm-audit-infer`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp074-compact-tracker-lgbm-audit-infer-v1`
  - status: complete
  - output: `/tmp/kaggle-output/exp074_compact_tracker_surface_lgbm_candidate_audit/infer_v1`
  - raw-test regenerated feature SHA: `723d2d29bd4701f05fc7ee7337a6911368dcc4dec651237679b739260a74e5d7`
  - feature generation elapsed: 93.103 sec
  - total inference elapsed: 122.933 sec
  - rows / fallback: 14,151 / 0
  - prediction SHA: `0a4a5c4010217624f8eb73f191ecbbedeaefba369db40542c171c078d5c84a9f`
  - submission SHA: `22f9eb3710ccec7741ce8006bee02ed69ed25829439114411cdba0038dcde0bc`
- Submit check:
  - command: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp074_compact_tracker_surface_lgbm_candidate_audit/infer_v1/submission.csv`
  - result: PASS

## 次のアクション

1. 必要なら inference v1 から Kaggle code submission を行い、Public LB ref を `submissions/SUBMISSIONS.md` に記録する。
2. Public LB を取らない場合は、exp074 を compact surface candidate evidence として保持し、次の戦略候補へ進む。
