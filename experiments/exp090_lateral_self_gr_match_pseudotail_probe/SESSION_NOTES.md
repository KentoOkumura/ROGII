# exp090_lateral_self_gr_match_pseudotail_probe セッションノート

## 現在の状態

- status: `train_completed_weak_positive`
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- blocked: none
- 2026-06-20 user clarification: exp090 は現在の解釈どおり exp073 系 ML 追加特徴 probe として進める。より良い PF/Beam を作る目的は別 backlog に分離する。

## 実装内容

- `.steering/20260620-exp090-lateral-self-gr-match-pseudotail-probe/` を作成し、requirements / design / tasklist を記入。
- `experiments/exp090_lateral_self_gr_match_pseudotail_probe/` を exp089 から作成。
- `settings.py` の experiment name を exp090 に更新。
- `config.yaml` を lateral self-GR match probe 用に更新。
- 補助実装を `lateral_self_gr_match_pseudotail_probe.py` に整理。
  - exp072 deterministic 196-feature train cache を読む。
  - target は exp073 と同じ `TVT - last_known_tvt` のままにする。
  - raw train horizontal well の `GR` と finite prefix `TVT_input` だけから、同一 well 内の prefix/eval GR match summary を作る。
  - multi-scale half window 8/15/25 の NCC score、matched prefix TVT offset、z-normalized L2、score gap、scale disagreement、prefix/eval missingness context を保存する。
  - `control_exp073_base196`、`self_gr_core`、`self_gr_core_multiscale`、`self_gr_core_context` を比較する。
  - fold/pooled metrics、well metrics、distance/tail buckets、OOF predictions、feature schema、feature importance、model manifest を保存する。
- train notebook を exp090 用の4セクション構成に更新。
- inference notebook は selected variant 未設定なら停止する guard notebook のままにした。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp090_lateral_self_gr_match_pseudotail_probe
uv run python scripts/new_experiment.py --name exp090_lateral_self_gr_match_pseudotail_probe --source experiments/exp089_pf_beam_disagreement_sample_weight
```

## 次のアクション

1. 静的検証を通す。
2. Kaggle train package を作成し、bootstrap manifest の config / 補助 `.py` SHA を確認する。
3. Kaggle train を実行して、variant 別 pooled RMSE、worst-well、distance/tail bucket、feature importance を確認する。
4. 改善候補が出た場合だけ inference port と raw-test self-GR feature parity を設計する。

## 検証

- `uv run python -m py_compile experiments/exp090_lateral_self_gr_match_pseudotail_probe/lateral_self_gr_match_pseudotail_probe.py experiments/exp090_lateral_self_gr_match_pseudotail_probe/public_notebook_replay_audit.py experiments/exp090_lateral_self_gr_match_pseudotail_probe/settings.py`: PASS
- `python3 -m json.tool experiments/exp090_lateral_self_gr_match_pseudotail_probe/exp090_lateral_self_gr_match_pseudotail_probe_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp090_lateral_self_gr_match_pseudotail_probe/exp090_lateral_self_gr_match_pseudotail_probe_inference.ipynb`: PASS
- `uv run ruff check experiments/exp090_lateral_self_gr_match_pseudotail_probe/lateral_self_gr_match_pseudotail_probe.py experiments/exp090_lateral_self_gr_match_pseudotail_probe/public_notebook_replay_audit.py experiments/exp090_lateral_self_gr_match_pseudotail_probe/settings.py`: PASS
- `uv run ruff format --check experiments/exp090_lateral_self_gr_match_pseudotail_probe/lateral_self_gr_match_pseudotail_probe.py experiments/exp090_lateral_self_gr_match_pseudotail_probe/settings.py`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp090_lateral_self_gr_match_pseudotail_probe`: PASS
- synthetic frame による `build_self_gr_match_features()` smoke test: PASS、10 rows / 26 columns、dynamic half-window feature groups OK。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp090_lateral_self_gr_match_pseudotail_probe --notebook train --kernel-id kentookumura/exp090-lateral-self-gr-match-pseudotail-probe-train --title "exp090 lateral self gr match pseudotail probe train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp090_lateral_self_gr_match_pseudotail_probe/kaggle/train`
- generated kernel id: `kentookumura/exp090-lateral-self-gr-match-pseudotail-probe-train`
- generated metadata: GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes `config.yaml` SHA `693390ceb5a1a5a1547664eeb7bf44a5c36da638eb14a9f194d02401d185a8fa` and `lateral_self_gr_match_pseudotail_probe.py` SHA `f30c024d17b43cf55189109022b98099c72d6ccabe133b33a2c0a1407a8fc6c8`。
- `uv run python scripts/update_experiment_summary.py`: PASS、91 experiments。

## Kaggle train v1

- canonical kernel id: `kentookumura/exp090-self-gr-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp090-self-gr-train`
- note: initial long slug `kentookumura/exp090-lateral-self-gr-match-pseudotail-probe-train` failed with `SaveKernel` 400, so package was regenerated with shorter slug/title.
- push: `Kernel version 1 successfully pushed`
- existence check: `kaggle kernels pull kentookumura/exp090-self-gr-train -p /tmp/kaggle-pull/exp090-self-gr-train-v1 -m`: PASS
- status after user completion notice: `KernelWorkerStatus.COMPLETE`
- logs: fetched; notebook completed and wrote result display / nbconvert output.
- output: `/tmp/kaggle-output/exp090_lateral_self_gr_match_pseudotail_probe/train_v1`
- copied small generated files to `experiments/exp090_lateral_self_gr_match_pseudotail_probe/artifacts/`
- full `kaggle kernels output` was interrupted after metrics/model/manifest files were available; downloaded `predictions.csv.gz` was empty, so gzip/content SHA are not used as evidence.

### Result

| variant | lgb_mean CV | control 差分 | feature 数 |
| --- | ---: | ---: | ---: |
| `self_gr_core_multiscale` | 9.516732864806912 | -0.009557442830422147 | 210 |
| `control_exp073_base196` | 9.526290307637334 | 0.000000000000000 | 196 |
| `self_gr_core` | 9.541383726295855 | +0.01509341865852143 | 201 |
| `self_gr_core_context` | 9.599141986796033 | +0.07285167915869906 | 208 |

- selected train-side variant: `self_gr_core_multiscale`
- source feature SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- best OOF prediction SHA: `6ffd17d023d1b0db0d85e4782d5b5cc75effb094635d030721b095a1616fc3d9`
- model manifest model count: 60
- well deltas: 402 improved / 371 worsened
- distance bucket deltas for best vs control: `000_050` -0.00299、`050_100` +0.00075、`100_250` +0.00325、`250_500` -0.00118、`500_1000` -0.00525、`1000_plus` -0.01060
- interpretation: weak positive but not enough to inference-port directly; well-level risk remains large.
