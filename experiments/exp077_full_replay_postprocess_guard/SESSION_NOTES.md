# exp077_full_replay_postprocess_guard セッションノート

## 現在の状態

- status: `ml_route_postprocessed_anchor`
- route: `ml_model`
- requested backlog name: `exp073_full_replay_postprocess_guard`
- actual experiment name: `exp077_full_replay_postprocess_guard`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle train: v1 `COMPLETE`
- Kaggle inference: v1 `COMPLETE`
- latest observed submission: ref `53809333` `COMPLETE`, Public LB `8.611`

## 実装内容

- `.steering/20260618-exp077-full-replay-postprocess-guard/` を作成。
- `experiments/exp077_full_replay_postprocess_guard/` を exp073 から作成。
- `config.yaml` を exp077 の後処理監査用に更新。
- train notebook を postprocess audit 中心に更新。
- `run_postprocess_guard()` を追加。
  - exp073 OOF predictions を読む。
  - exp072 full replay feature cache が見つかる場合は必要列だけ id join する。
  - baseline、residual clip、tail-start fade、flat-prefix hold blend、PF confidence clip、PF-vs-ML tiny gate、long-tail tiny gate を比較する。
  - policy ごとの RMSE、MAE、prediction SHA、distance bucket metrics、policy predictions を保存する。
- LightGBM feature importance 出力を追加。
  - `_fit_one_mode()` は fold/model ごとの gain/split importance を保存する。
  - `run_saved_model_feature_importance_audit()` は exp073 saved booster manifest から fold/model importance を復元する。
  - `write_feature_importance_outputs()` は fold/model 平均 importance CSV と matplotlib plot を保存する。
  - train notebook は `exp063_full_replay_repro_guard_feature_importance_mean_top40.png` を表示する。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp077_full_replay_postprocess_guard
uv run python scripts/new_experiment.py --name exp077_full_replay_postprocess_guard --source experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay
```

静的確認:

```bash
uv run python -m py_compile experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py
uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/exp077_full_replay_postprocess_guard_train.ipynb
uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/exp077_full_replay_postprocess_guard_inference.ipynb
uv run ruff check experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py
uv run python scripts/validate_experiment.py --experiment exp077_full_replay_postprocess_guard
```

## 結果

- Kaggle train v1 は完了済み。
- exp073 の実験番号は再利用せず、最新 `exp076` の次として `exp077` を使った。
- 静的検証:
  - `uv run python -m py_compile experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py`: pass
  - `uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/exp077_full_replay_postprocess_guard_train.ipynb`: pass
  - `uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/exp077_full_replay_postprocess_guard_inference.ipynb`: pass
  - `uv run ruff check experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py`: pass
  - `uv run python scripts/validate_experiment.py --experiment exp077_full_replay_postprocess_guard`: pass
- Kaggle package:
  - train: `experiments/exp077_full_replay_postprocess_guard/kaggle/train`
  - inference: `experiments/exp077_full_replay_postprocess_guard/kaggle/inference`
  - train kernel id: `kentookumura/exp077-full-replay-postprocess-guard-train`
  - inference kernel id: `kentookumura/exp077-full-replay-postprocess-guard-infer`
- Kaggle train v1:
  - `kaggle kernels push -p experiments/exp077_full_replay_postprocess_guard/kaggle/train`: `Kernel version 1 successfully pushed`.
  - URL: https://www.kaggle.com/code/kentookumura/exp077-full-replay-postprocess-guard-train
  - `kaggle kernels pull kentookumura/exp077-full-replay-postprocess-guard-train -p /tmp/kaggle-pull/exp077-full-replay-postprocess-guard-train-v1 -m`: pass.
  - Initial normal logs and multiple `logs -f` polling windows were empty while status was `RUNNING`; treated as Kaggle CLI log lag / notebook startup, not failure.
  - Later `logs -f` returned completion logs.
  - `kaggle kernels status kentookumura/exp077-full-replay-postprocess-guard-train`: `KernelWorkerStatus.COMPLETE`.
  - `kaggle kernels output kentookumura/exp077-full-replay-postprocess-guard-train -p /tmp/kaggle-output/exp077-full-replay-postprocess-guard-train-v1`: pass.
  - output path: `/tmp/kaggle-output/exp077-full-replay-postprocess-guard-train-v1`
  - rows / wells: 3,783,989 / 773
  - prediction source SHA: `986e26c5c6617ade714623d44433e9beacdb2b1027d46c4a4e70825bc`
  - feature source SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - exp073 model manifest SHA: `af31c1835d5e592f684dfb4d91f0d638d20a221495754e2ca1e3e734edd90b33`
  - best policy: `longtail_likpf_tiny_gate_w006`
  - best OOF RMSE: `9.470514771712411`
  - exp073 baseline OOF RMSE in this audit: `9.526374749390682`
  - delta vs baseline: `-0.055859977678271`
  - best policy prediction SHA: `9813e6ba2e008f87c37ca0185fb754e17435a9a70c9a5f559ccd7c9a3dce3d24`
  - decompressed policy predictions content SHA: `807109e68b8af252e06a00bc9a443d3b1c0639791919b8ba3c415c3a1e4c2292`
  - summary SHA: `a840c740a9d422b83ab7be4fd50fa31e8acc8fb3939883aafe893a70c7bcef95`
  - metrics SHA: `3c504885271ab91d203887350174bff1a268dbc4a168edf77f85f2a047988c0a`
  - feature importance plot SHA: `d206af0b5f14cfdbcc815f6234c5508c830aef7a5a0b0effb281ee075d72837f`
  - top gain importance features: `likpf_mean_d`, `tvt_dense50_d`, `tvt_densew_d`, `pf_ancc_delta`, `tvt_dense_d`

## 次のアクション

## Inference / submit

- Fixed policy `longtail_likpf_tiny_gate_w006` を inference に port。
- `env UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp077_full_replay_postprocess_guard --notebook inference --kernel-id kentookumura/exp077-full-replay-postprocess-guard-infer --title "exp077 full replay postprocess guard infer" --run-on-push --strict`: pass.
- `kaggle kernels push -p experiments/exp077_full_replay_postprocess_guard/kaggle/inference`: `Kernel version 1 successfully pushed`.
- URL: https://www.kaggle.com/code/kentookumura/exp077-full-replay-postprocess-guard-infer
- `kaggle kernels pull kentookumura/exp077-full-replay-postprocess-guard-infer -p /tmp/kaggle-pull/exp077-full-replay-postprocess-guard-infer-v1 -m`: pass.
- `kaggle kernels status kentookumura/exp077-full-replay-postprocess-guard-infer`: `KernelWorkerStatus.COMPLETE`.
- `kaggle kernels output kentookumura/exp077-full-replay-postprocess-guard-infer -p /tmp/kaggle-output/exp077-full-replay-postprocess-guard-infer-v1`: pass.
- output path: `/tmp/kaggle-output/exp077-full-replay-postprocess-guard-infer-v1`
- selected: `gpu_repro_guard_dp_threads8` / `lgb_mean`
- postprocess policy: `longtail_likpf_tiny_gate_w006`
- postprocess adjusted rows: 14,151
- rows / wells / features: 14,151 / 3 / 196
- fallback rows: 0
- test feature raw SHA: `e9f49c8d0abb5c4e20a6210ccf493adaa3a863c82ba378e977b0397c093cec13`
- test feature decompressed content SHA: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
- test feature generation: raw test PF/Beam/likelihood-PF replay was regenerated on Kaggle (`source_kind=raw_test_regenerated_exp063_public_replay`); feature elapsed `94.059` sec for 3 wells / 14,151 rows.
- prediction SHA: `02e4f25f665c1610db80a68bd5f77f62ec347f702126f26828ac7312f2a7ddfd`
- prediction decompressed content SHA: `6286ea39a876b345e37055e8e8fa5ccc7f7c8bd8e71021ff0f3960d0fa67a8bf`
- submission SHA: `ccf17704959274d9e38f6eb8a7fe3c55a19128a8f24ba1a3d555f6af73bc8538`
- prediction range min / max / mean / std: 11594.5224609375 / 12241.671875 / 11905.889577012005 / 279.3515639361514
- `python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp077-full-replay-postprocess-guard-infer-v1/submission.csv --sample data/raw/sample_submission.csv`: PASS
- User reported submission completed manually.
- `kaggle competitions submissions rogii-wellbore-geology-prediction` latest observed:
  - ref `53809333`, date `2026-06-18 13:23:00.397000`, status `COMPLETE`, Public LB `8.611`.
  - nearby refs `53807892` and `53807896` completed with Public LB `8.489`, but they are exp075 duplicate submissions, not exp077.
  - exp077 improves over the exp073 deterministic raw ML anchor Public LB `8.780`, so it is promoted to the ML route submitted/postprocessed anchor.

## 次のアクション

1. 後続の ML route LB 比較では exp077 Public LB `8.611` を submitted/postprocessed anchor として使う。
2. exp073 は raw deterministic anchor として保持し、後処理や特徴量追加の素の比較基準に使う。
3. 後処理を再訪する場合は、OOF best 単独ではなく visible-tail / hidden-transfer guard を先に置く。

## 2026-06-20 pf_confidence_residual_clip 実装

- `pf_confidence_residual_clip_q995` を fixed inference postprocess policy として実行できるようにした。
- train audit 側の `pf_confidence_residual_clip_q995` 候補も同じ helper を使うように整理した。
- `apply_fixed_postprocess_policy()` は `pf_confidence_residual_clip` / `pf_confidence_residual_clip_q995` を受け付ける。
- `run_saved_model_inference()` と inference notebook は `inference.postprocess_params` を受け取る。
- `config.yaml` の `inference.postprocess_params.pf_confidence_residual_clip.residual_clip_limit` に、exp073 selected OOF `target_delta` の q0.995 = `66.5908203125` を固定値として追加した。
- `pf_std_p75` / `beam_std_abs_p75` は null の場合、target-free に current inference frame から計算する。
- 既存の submitted anchor を変えないため、`inference.postprocess_policy` の default は `longtail_likpf_tiny_gate_w006` のまま維持した。`pf_confidence_residual_clip_q995` を実行する場合は config の policy を切り替える。
- 既存 Kaggle inference v1 output を使った smoke:
  - policy: `pf_confidence_residual_clip_q995`
  - adjusted rows: `0`
  - dynamic limit min / median / max: `22.19693946838379` / `66.5908203125` / `66.5908203125`
  - adjusted delta range: `-22.164161682128906` - `18.152917861938477`

検証:

```bash
uv run python -m py_compile experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py
uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/exp077_full_replay_postprocess_guard_inference.ipynb
uv run ruff check experiments/exp077_full_replay_postprocess_guard/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/settings.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp077_full_replay_postprocess_guard --notebook inference --kernel-id kentookumura/exp077-full-replay-postprocess-guard-infer --title "exp077 full replay postprocess guard infer" --run-on-push --strict
uv run python -m py_compile experiments/exp077_full_replay_postprocess_guard/kaggle/inference/exp063_full_replay_reproducibility_guard.py experiments/exp077_full_replay_postprocess_guard/kaggle/inference/public_notebook_replay_audit.py experiments/exp077_full_replay_postprocess_guard/kaggle/inference/settings.py
uv run python -m json.tool experiments/exp077_full_replay_postprocess_guard/kaggle/inference/exp077_full_replay_postprocess_guard_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp077_full_replay_postprocess_guard
```

結果:

- すべて pass。

## 2026-06-20 exp084 への分離

- `pf_confidence_residual_clip` は `KAGGLE_DIRECTION.md` の独立 backlog 項目だったため、exp077 の variant として実行するのではなく `exp084_pf_confidence_residual_clip` に分離した。
- exp077 の source config と local Kaggle inference package config は、submitted anchor policy の `longtail_likpf_tiny_gate_w006` に戻した。
- 注意: 分離前に exp077 inference kernel version 2 を `pf_confidence_residual_clip_q995` config で push してしまった。以後の記録では exp077 submitted anchor は Kaggle inference v1 / submission ref `53809333` / Public LB `8.611` として扱い、`pf_confidence_residual_clip_q995` の検証結果は exp084 に集約する。
