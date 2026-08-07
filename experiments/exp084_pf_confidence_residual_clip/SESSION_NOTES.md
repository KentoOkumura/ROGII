# exp084_pf_confidence_residual_clip セッションノート

## 現在の状態

- status: `submitted_noop_not_adopted_public_lb_8_746`
- route: `ml_model`
- parent: `exp077_full_replay_postprocess_guard`
- source model experiment: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- postprocess policy: `pf_confidence_residual_clip_q995`
- exp077 local config/package: `longtail_likpf_tiny_gate_w006` に復旧済み

## 実装内容

- `.steering/20260620-exp084-pf-confidence-residual-clip/` を作成。
- `experiments/exp084_pf_confidence_residual_clip/` を exp077 から作成。
- exp077 に実装済みだった `pf_confidence_residual_clip_q995` inference policy を、独立 exp として実行できるように分離。
- `config.yaml` の `inference.postprocess_policy` を `pf_confidence_residual_clip_q995` に設定。
- exp073 selected OOF `target_delta` q0.995 に由来する `residual_clip_limit=66.5908203125` を固定。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp084_pf_confidence_residual_clip
uv run python scripts/new_experiment.py --name exp084_pf_confidence_residual_clip --source experiments/exp077_full_replay_postprocess_guard
uv run python -m py_compile experiments/exp084_pf_confidence_residual_clip/exp063_full_replay_reproducibility_guard.py experiments/exp084_pf_confidence_residual_clip/public_notebook_replay_audit.py experiments/exp084_pf_confidence_residual_clip/settings.py
uv run python -m json.tool experiments/exp084_pf_confidence_residual_clip/exp084_pf_confidence_residual_clip_inference.ipynb
uv run ruff check experiments/exp084_pf_confidence_residual_clip/exp063_full_replay_reproducibility_guard.py experiments/exp084_pf_confidence_residual_clip/public_notebook_replay_audit.py experiments/exp084_pf_confidence_residual_clip/settings.py
uv run python scripts/validate_experiment.py --experiment exp084_pf_confidence_residual_clip
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp084_pf_confidence_residual_clip --notebook inference --kernel-id kentookumura/exp084-pf-confidence-residual-clip-infer --title "exp084 pf confidence residual clip infer" --run-on-push --strict
kaggle kernels push -p experiments/exp084_pf_confidence_residual_clip/kaggle/inference
kaggle kernels pull kentookumura/exp084-pf-confidence-residual-clip-infer -p /tmp/kaggle-pull/exp084-pf-confidence-residual-clip-infer-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp084-pf-confidence-residual-clip-infer
kaggle kernels output kentookumura/exp084-pf-confidence-residual-clip-infer -p /tmp/kaggle-output/exp084-pf-confidence-residual-clip-infer-v1
python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp084-pf-confidence-residual-clip-infer-v1/submission.csv --sample data/raw/sample_submission.csv
```

## 結果

- Kaggle inference v1: COMPLETE
- URL: https://www.kaggle.com/code/kentookumura/exp084-pf-confidence-residual-clip-infer
- output path: `/tmp/kaggle-output/exp084-pf-confidence-residual-clip-infer-v1`
- selected: `gpu_repro_guard_dp_threads8` / `lgb_mean`
- model manifest SHA256: `af31c1835d5e592f684dfb4d91f0d638d20a221495754e2ca1e3e734edd90b33`
- policy: `pf_confidence_residual_clip_q995`
- adjusted rows: `0`
- residual clip limit: `66.5908203125`
- pf std p75 / beam abs std p75: `1.1387114524841309` / `4.414724826812744`
- dynamic limit min / median / max: `22.19693946838379` / `66.5908203125` / `66.5908203125`
- rows: `14,151`
- prediction range min / max / mean / std: `11593.671875` / `12241.693359375` / `11905.6562581432` / `279.3037397081`
- test feature raw SHA256: `2468123c910965bcd0d16d0fbadaeb34a3645e613dd1380b45074ff43d00fb44`
- test feature content SHA256: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
- prediction SHA256: `2e47e986c013acfafaa01c652d47649778db5616e40dc3130e4e12dede7b7502`
- prediction content SHA256: `a66acb16ff40a25062598eef21579495486a8752076ed74ca8dc765ab6f2816f`
- submission SHA256: `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`
- summary SHA256: `dcdc50b349ba342b1a3ab7001c7f37d4fb9d77a9b5f6d1e77b102c85fb329dbe`
- metrics SHA256: `96184f8c8a09c95589f90f301f9515db939693ce102164bb41703825624e999d`
- submit-check: PASS

## 解釈

public sample では `pf_confidence_residual_clip_q995` が no-op だった。local submission SHA は exp073 deterministic output と同じ `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b` なので、以降の LB は policy 改善ではなく no-op 出力の再提出スコアとして扱う。

## Submission

- User reported submission completed.
- `kaggle competitions submissions rogii-wellbore-geology-prediction` latest observed:
  - ref `53854846`, date `2026-06-19 16:14:17.167000`, status `COMPLETE`, Public LB `8.746`
  - ref `53854829`, date `2026-06-19 16:13:41.720000`, status `COMPLETE`, Public LB `8.746`
- Both rows have blank descriptions. Attribution to exp084 is based on timing after exp084 inference output.
- Public LB `8.746` is worse than exp077 `8.611`, so exp084 is not adopted.

## 次のアクション

1. `pf_confidence_residual_clip` backlog は実装・提出済みとして閉じる。
2. 次に続ける場合は、clip policy を直接広げるより、`prefix_backtest_tvt_confidence` または `pf_beam_disagreement_error_map` で confidence calibration を先に行う。
