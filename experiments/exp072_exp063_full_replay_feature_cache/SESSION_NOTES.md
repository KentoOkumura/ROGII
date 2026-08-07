# exp072_exp063_full_replay_feature_cache セッションノート

## 現在の状態

- status: `completed`
- route: `pf_beam`
- parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- Kaggle train feature cache: v2 `COMPLETE`
- Kaggle inference: 対象外

## 実装内容

- exp063 から実験フォルダをコピーして exp072 を作成。
- `.steering/20260614-exp072-exp063-full-replay-feature-cache/` を作成。
- `feature_cache.py` を追加。
  - exp063 の `public_notebook_replay_audit.py` を再利用。
  - raw train files から `pixiux_likpf_public_replay` full train features を生成。
  - 期待 feature count は 196。
  - LightGBM / CatBoost / Ridge / prediction / submission は実行しない。
  - test features は作らない。各後続 experiment の inference notebook 内で current raw test から再生成する。
- train notebook を CPU-only train feature cache generation に差し替え。
- inference notebook は no-op policy check に差し替え。
- `config.yaml` を CPU-only feature cache 用に更新。
- 静的検証:
  - `py_compile`: PASS
  - `ruff check`: PASS
  - train notebook JSON validation: PASS
  - inference notebook JSON validation: PASS
  - `validate_experiment.py`: PASS
- Kaggle train package:
  - generated at `experiments/exp072_exp063_full_replay_feature_cache/kaggle/train`
  - kernel id: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel sources: none
  - bootstrap includes `feature_cache.py`, `public_notebook_replay_audit.py`, `settings.py`, `config.yaml`, `project.yml`, `src/`
- Kaggle train feature cache version 1 pushed successfully:
  - URL: `https://www.kaggle.com/code/kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp072-exp063-full-replay-feature-cache-train-v1`
  - initial status: `KernelWorkerStatus.RUNNING`
  - metadata: `enable_gpu=false`
- Kaggle train feature cache version 1 completed:
  - rows / wells / features: 3,783,989 / 773 / 196
  - train feature SHA256: `86d4777ddf44134cc8e1c7ce4eebf56cc1537ce6baf2e39f75c5c65cf26335ae`
  - elapsed seconds: 16,714.392
  - feature generation seconds: 14,415.986
  - generated train features: `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
  - generated schema: `exp063_full_replay_feature_cache_feature_schema.csv`
  - generated summary: `exp063_full_replay_feature_cache_summary.json`
  - local `kaggle kernels output` download was incomplete for the large train gzip, but logs confirm the file exists in Kaggle output. Downstream exp073 uses the Kaggle kernel source directly.
- Deterministic patch:
  - `public_notebook_replay_audit.py` now derives stable seeds with SHA256.
  - `run_pf_ancc()` uses `stable_seed("pf_ancc", wid)`.
  - `run_pf_z()` uses `stable_seed("pf_z", wid)`.
  - likelihood PF uses `stable_seed("likpf", split, wid)` as `seed_base`.
  - JIT warmup no longer uses `np.random.randn` for `_beam_jit`.
- Kaggle train feature cache version 2 pushed and completed:
  - rows / wells / features: 3,783,989 / 773 / 196
  - train feature SHA256: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - previous v1 SHA256: `86d4777ddf44134cc8e1c7ce4eebf56cc1537ce6baf2e39f75c5c65cf26335ae`
  - elapsed seconds: 17,728.972
  - feature generation seconds: 15,380.262

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp072_exp063_full_replay_feature_cache --title "exp063 full replay feature cache"
uv run python scripts/new_experiment.py --name exp072_exp063_full_replay_feature_cache --source experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
uv run python -m py_compile experiments/exp072_exp063_full_replay_feature_cache/feature_cache.py experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py experiments/exp072_exp063_full_replay_feature_cache/settings.py
uv run ruff check experiments/exp072_exp063_full_replay_feature_cache/feature_cache.py experiments/exp072_exp063_full_replay_feature_cache/public_notebook_replay_audit.py experiments/exp072_exp063_full_replay_feature_cache/settings.py
uv run python -m json.tool experiments/exp072_exp063_full_replay_feature_cache/exp072_exp063_full_replay_feature_cache_train.ipynb
uv run python -m json.tool experiments/exp072_exp063_full_replay_feature_cache/exp072_exp063_full_replay_feature_cache_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp072_exp063_full_replay_feature_cache
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp072_exp063_full_replay_feature_cache --notebook train --kernel-id kentookumura/exp072-exp063-full-replay-feature-cache-train --title "exp072 exp063 full replay feature cache train" --run-on-push --strict
kaggle kernels push -p experiments/exp072_exp063_full_replay_feature_cache/kaggle/train
kaggle kernels pull kentookumura/exp072-exp063-full-replay-feature-cache-train -p /tmp/kaggle-pull/exp072-exp063-full-replay-feature-cache-train-latest -m
kaggle kernels logs kentookumura/exp072-exp063-full-replay-feature-cache-train
```

## 次のアクション

1. 後続実験は v2 deterministic train feature cache を kernel source として読み、test features は各 inference notebook 内で同じ stable seed policy で raw test から再生成する。
