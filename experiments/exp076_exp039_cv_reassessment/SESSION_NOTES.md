# exp076_exp039_cv_reassessment セッションノート

## 現在の状態

- status: submitted_not_adopted
- route: `ml_model`
- parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- reference branch: `exp039_ravaghi_single_lgbm_inference_submit`
- requested backlog name: `exp073_exp039_cv_reassessment`
- actual experiment name: `exp076_exp039_cv_reassessment`

## 仮説

破棄済み `exp068` の意図を、対象だけ `exp063` から deterministic anchor の `exp073` に差し替える。`exp039` CV surface で exp073 full replay LightGBM family を再評価し、旧 exp039 branch との比較材料にする。

## 実装メモ

- 実験番号は既存最大 `exp075` の次として `exp076` にした。
- train notebook:
  - exp039 CV surface と exp072/exp073 full replay train cache を `id` join する。
  - exp072/exp073 full replay train cache の feature schema が exp073 と同じ 196 features であることを schema CSV で検証する。
  - exp073 cache を学習 row set の正にし、exp039 CV surface は well_id 単位の fold 付与に使う。exp039-only rows と exact id overlap のない exp073 rows は join stats に記録する。
  - exp073 cache 側の well に exp039 CV surface が欠ける場合は、CV を付与できないため学習前に停止する。
  - exp073 cache の `target` と exp039 surface 由来の `target_tvt - last_known_tvt` が exact id overlap subset で一致することを検証し、exp039 target で上書きしない。
  - `leave_one_original_fold_out` と `well_hash_holdout` で LightGBM を再学習評価する。
  - train 側の再現性境界は LightGBM。固定 feature cache content SHA、model SHA、OOF prediction SHA、manifest を保存する。
- inference notebook:
  - exp076 train output の saved boosters を読む。
  - raw test から exp073 full replay PF/Beam/likelihood-PF features を再生成する。
  - inference 側の再現性境界は PF/Beam。exp073 で確立済みの stable per-well seed policy と generated feature content SHA で担保し、2 回生成はしない。
  - static public-sample prediction artifact は使わない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp076_exp039_cv_reassessment
uv run python scripts/new_experiment.py --name exp076_exp039_cv_reassessment --source experiments/exp068_equivalent_pixiux_inference_port
```

実装後の静的確認:

```bash
uv run python -m py_compile experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py
uv run python -m json.tool experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_train.ipynb
uv run python -m json.tool experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_inference.ipynb
uv run ruff check experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py
uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment
```

Kaggle train package:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook train --kernel-id kentookumura/exp076-exp039-cv-reassessment-train --title "exp076 exp039 cv reassessment train" --run-on-push --strict
kaggle kernels push -p experiments/exp076_exp039_cv_reassessment/kaggle/train
kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-train -m
```

Kaggle inference package:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook inference --kernel-id kentookumura/exp076-exp039-cv-reassessment-infer --title "exp076 exp039 cv reassessment infer" --run-on-push --strict
```

## 結果

- 実装済み。
- 静的検証:
  - `uv run python -m py_compile experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
  - `uv run python -m json.tool experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_train.ipynb`: pass
  - `uv run python -m json.tool experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_inference.ipynb`: pass
  - `uv run ruff check experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
  - `uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment`: pass
- サブエージェントレビュー後の修正:
  - exp073 と同じ特徴量であることを、feature count だけでなく schema 順序込みで検証する guard を追加した。
  - exp039 CV surface の row が exp073 cache に欠落していないことを検証する guard を追加した。
  - exp073 cache target と exp039 surface 由来 target delta の一致 guard を追加し、target を exp039 側で上書きしないようにした。
  - 修正後に `py_compile` / `ruff check` / `validate_experiment` を再実行し、train / inference Kaggle package を再生成した。
- Kaggle package:
  - train: `experiments/exp076_exp039_cv_reassessment/kaggle/train`
  - inference: `experiments/exp076_exp039_cv_reassessment/kaggle/inference`
  - train metadata uses `kentookumura/exp029-sel15-pf-oof-train` and `kentookumura/exp072-exp063-full-replay-feature-cache-train`.
  - inference metadata uses `kentookumura/exp076-exp039-cv-reassessment-train`.
- Kaggle train:
  - GPU enabled metadata (`enable_gpu: true`) で train package を再生成した。
  - `kaggle kernels push -p experiments/exp076_exp039_cv_reassessment/kaggle/train`: `Kernel version 1 successfully pushed`.
  - URL: https://www.kaggle.com/code/kentookumura/exp076-exp039-cv-reassessment-train
  - `kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-train -m`: pass.
  - push 直後の通常 logs と 180 秒 polling は空。ユーザー指示により追加監視は行わない。
- Kaggle train v1 failure:
  - `kaggle kernels logs kentookumura/exp076-exp039-cv-reassessment-train`: failed at `build_exp039_cv_exp073_frame`.
  - error: `exp039 CV surface rows are missing from the exp073 full replay cache: dropped_exp039_rows=316`.
  - sample missing ids: `404c4384_1698`, `404c4384_1699`, `404c4384_1700`, `fba7683c_1338`...
  - interpretation: exp039 と exp073 の target 差分ではなく、exp039 CV surface 側に exp073 full replay cache には存在しない row があることが原因。
  - fix: exp073 cache row set を正にし、exp039-only rows は `dropped_exp039_rows` / `dropped_exp039_row_sample` として記録して除外する。逆に exp073 cache rows に exp039 fold が欠ける場合は停止する。
  - verification after fix:
    - `uv run python -m py_compile experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
    - `uv run ruff check experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
    - `uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment`: pass
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook train --kernel-id kentookumura/exp076-exp039-cv-reassessment-train --title "exp076 exp039 cv reassessment train" --run-on-push --strict`: pass
  - `kaggle kernels push -p experiments/exp076_exp039_cv_reassessment/kaggle/train`: `Kernel version 2 successfully pushed`.
  - `kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-train-v2 -m`: pass.
  - version 2 URL: https://www.kaggle.com/code/kentookumura/exp076-exp039-cv-reassessment-train
- Kaggle train v2 failure:
  - `kaggle kernels logs kentookumura/exp076-exp039-cv-reassessment-train`: failed at the reverse row-coverage guard.
  - error: `exp073 full replay cache rows are missing from the exp039 CV surface ... dropped_cache_rows=2002026`.
  - interpretation: exp039 CV surface is a sparse evaluation surface, while exp073 full replay cache contains a larger train row set. Exact `id` join cannot preserve the exp073 row set.
  - fix: assign exp039 `fold` by `well_id` to the full exp073 cache. Keep exp073 cache `target` as the supervised target. Use exact id overlap only for target-consistency audit.
  - added stats: `overlap_rows_for_target_audit`, `exp073_cache_rows_without_exact_exp039_id`, `fold_assignment=exp039_fold_by_well_id`.
  - verification after fix:
    - `uv run python -m py_compile experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
    - `uv run ruff check experiments/exp076_exp039_cv_reassessment/exp073_exp039_cv_reassessment.py experiments/exp076_exp039_cv_reassessment/settings.py`: pass
    - `uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment`: pass
    - synthetic `build_exp039_cv_exp073_frame` smoke: pass, preserving full cache rows while assigning fold by well.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook train --kernel-id kentookumura/exp076-exp039-cv-reassessment-train --title "exp076 exp039 cv reassessment train" --run-on-push --strict`: pass
  - `kaggle kernels push -p experiments/exp076_exp039_cv_reassessment/kaggle/train`: `Kernel version 3 successfully pushed`.
  - `kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-train-v3 -m`: pass.
  - version 3 URL: https://www.kaggle.com/code/kentookumura/exp076-exp039-cv-reassessment-train
- Kaggle train v3 completed:
  - `kaggle kernels logs kentookumura/exp076-exp039-cv-reassessment-train`: completed; summary and model manifest saved.
  - `kaggle kernels output kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-output/exp076-train-v3-complete`: pass.
  - `kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-train -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-train-v3-complete -m`: pass.
  - elapsed seconds: 14184.838.
  - join stats:
    - exp039 rows: 1,782,279
    - exp073 cache rows / training rows: 3,783,989
    - exact id overlap rows for target audit: 1,781,963
    - exp039-only rows: 316
    - exp073 cache rows without exact exp039 id: 2,002,026
    - fold assignment: `exp039_fold_by_well_id`
    - target diff max / mean on overlap: 0.0 / 0.0
  - pooled CV:
    - `leave_one_original_fold_out/lgb_mean`: 9.696040173882945
    - `well_hash_holdout/lgb_mean`: 9.553554167040426
    - `leave_one_original_fold_out/lgb0/lgb1/lgb2`: 9.966644975271121 / 9.651043444227914 / 9.705470296611832
    - `well_hash_holdout/lgb0/lgb1/lgb2`: 9.799962732101141 / 9.55120942405678 / 9.539948110569153
  - SHA:
    - exp039 surface content SHA: `b96046cd452abca92ed7200188e3b628745a3f5ff5ccb70679da1dc97a79b5a3`
    - feature cache raw SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
    - feature cache content SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
    - model manifest SHA: `d68aac6d5ec5a34f8ead30a0b119daace4823e4f1207428201ba8e47b54db37f`
    - OOF predictions decompressed content SHA: `2fe04bd2980505ee4fd7dbd0acd61b307e429d45a6baa71a2f11edc3effb21ef`
    - selected `leave_one_original_fold_out/lgb_mean` prediction SHA: `afcf0bd9325b7ee7060d299285dfa8aeaf035405902839deb47e2ece8d04c783`
    - selected `well_hash_holdout/lgb_mean` prediction SHA: `9ee31704447a57f2262a689d4b66300b3eb21f14b1de1d0217342c068f0972c2`
  - interpretation: exp039 CV surface reassessmentでは `leave_one_original_fold_out/lgb_mean` が exp073 native CV 9.526374749 より悪い。評価面が違うため anchor 更新根拠にはしない。
- Kaggle inference は v1 実行中。
- Kaggle inference v1:
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook inference --kernel-id kentookumura/exp076-exp039-cv-reassessment-infer --title "exp076 exp039 cv reassessment infer" --run-on-push --strict`: pass
  - `uv run python -m json.tool experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_inference.ipynb`: pass
  - `kaggle kernels push -p experiments/exp076_exp039_cv_reassessment/kaggle/inference`: `Kernel version 1 successfully pushed`.
  - URL: https://www.kaggle.com/code/kentookumura/exp076-exp039-cv-reassessment-infer
  - `kaggle kernels pull kentookumura/exp076-exp039-cv-reassessment-infer -p /tmp/kaggle-pull/exp076-exp039-cv-reassessment-infer-v1 -m`: pass.
  - train source: `kentookumura/exp076-exp039-cv-reassessment-train`
  - inference mode: `gpu_repro_guard_dp_threads8__leave_one_original_fold_out` / `lgb_mean`
  - raw-test PF/Beam regeneration: enabled.
  - `kaggle kernels logs kentookumura/exp076-exp039-cv-reassessment-infer`: completed.
  - `kaggle kernels output kentookumura/exp076-exp039-cv-reassessment-infer -p /tmp/kaggle-output/exp076-infer-v1-complete`: pass.
  - inference elapsed seconds: 128.305.
  - test features:
    - rows / wells / columns: 14,151 / 3 / 198
    - raw SHA: `67afd7343622c6f26128229209b9408e5fbcb36d0da2760c07a1247fa7a01025`
    - decompressed content SHA: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
  - inference metrics:
    - loaded model count: 15
    - test / submission / predicted rows: 14,151 / 14,151 / 14,151
    - fallback rows: 0
    - prediction min / max / mean / std: 11594.158203125 / 12241.3212890625 / 11905.878221255343 / 279.259413014171
    - prediction SHA: `bfb21114e7deb98e5880ebd0a1f0b33dfb129531dd4e8fa0908e4c65e01e4938`
    - test prediction CSV raw/content SHA: `b128d798d91ec021848ec169e6a6104e621f2d316036cbcc3b1f05e6f5cab43d` / `acb16905dc1cfda2ab8acd9fc9eeb279ca3749d75bbe9c70f1ff4b159dcec481`
    - submission SHA: `6afd2296208449ef372e4aef49c41de7636aadc266e09fd5ee41a2a4d36623c1`
  - `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp076-infer-v1-complete/submission.csv --sample data/raw/sample_submission.csv`: PASS.
- Kaggle submission:
  - `kaggle competitions submissions rogii-wellbore-geology-prediction`: latest matching submission is ref `53757190`.
  - date: `2026-06-17 00:00:20.047000`
  - status: `SubmissionStatus.COMPLETE`
  - Public LB: 8.799
  - Private LB: not shown.
  - interpretation: exp027 8.781 and nearby exp073 submissions 8.780 より悪化。採用しない。

## 2026-06-18 notebook orchestration refactor

- User request: test policy 2 from the notebook readability discussion, i.e. move only the experiment orchestration into notebook cells while leaving heavy helper implementations in `.py`.
- Experiment-number decision:
  - no new experiment number was created.
  - reason: this changes only notebook structure/readability and Kaggle push packaging, not the hypothesis, feature set, model family, CV surface, inference policy, or submission candidate.
- Updated source notebooks:
  - `exp076_exp039_cv_reassessment_train.ipynb`
    - expanded `run_exp073_on_exp039_cv(...)` into visible cells for source checks, frame construction, audit/mode selection, LightGBM fold fitting via `_fit_one_mode`, metrics/model manifest writing, and summary writing.
    - kept low-level loading, feature joining, split-code generation, and fold training helpers in `exp073_exp039_cv_reassessment.py`.
  - `exp076_exp039_cv_reassessment_inference.ipynb`
    - expanded `run_saved_model_inference(...)` into visible cells for model manifest selection, test feature regeneration/loading, saved booster prediction, submission assembly, metrics, SHA, and summary writing.
    - kept PF/Beam raw-test feature generation helpers in `public_notebook_replay_audit.py` and `generate_exp063_tracker_test_frame(...)`.
- Verification:
  - notebook code-cell AST parse: pass.
  - `uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment`: pass.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook train --kernel-id kentookumura/exp076-exp039-cv-reassessment-train --title "exp076 exp039 cv reassessment train" --run-on-push --strict`: pass.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook inference --kernel-id kentookumura/exp076-exp039-cv-reassessment-infer --title "exp076 exp039 cv reassessment infer" --run-on-push --strict`: pass.
- Not run:
  - no local notebook execution.
  - no Kaggle push or submission.

## 2026-06-18 marimo to ipynb trial

- User request: try a marimo -> ipynb workflow on exp076.
- Experiment-number decision:
  - no new experiment number was created.
  - reason: this is a notebook authoring/export workflow trial only; it does not change the experiment hypothesis, features, model family, CV, inference policy, or submission candidate.
- Commands / workflow:
  - `uv run --with marimo marimo --help`: pass; temporary marimo version used by generated files was `0.23.9`.
  - `uv run --with marimo marimo -q -y convert experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_train.ipynb -o experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_train_marimo.py`: pass.
  - `uv run --with marimo marimo -q -y convert experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_inference.ipynb -o experiments/exp076_exp039_cv_reassessment/exp076_exp039_cv_reassessment_inference_marimo.py`: pass.
  - `uv run --with marimo marimo check ..._train_marimo.py`: pass.
  - `uv run --with marimo marimo check ..._inference_marimo.py`: pass.
  - `uv run --with marimo marimo export ipynb ..._train_marimo.py -o ..._train_from_marimo.ipynb --sort=topological -f`: pass.
  - `uv run --with marimo marimo export ipynb ..._inference_marimo.py -o ..._inference_from_marimo.ipynb --sort=topological -f`: pass.
- Implementation notes:
  - Exported notebooks initially contained a first code cell `import marimo as mo`; this was removed from the official Kaggle-facing ipynb because Kaggle runtime should not require marimo.
  - marimo metadata was stripped from official ipynb after export.
  - marimo source files were moved under `marimo_sources/` so `prepare_kaggle_notebooks.py` does not include them in the Kaggle support ZIP. That script only packages experiment-root `.py` / YAML files.
  - Official notebooks now come from the marimo export flow:
    - `exp076_exp039_cv_reassessment_train.ipynb`
    - `exp076_exp039_cv_reassessment_inference.ipynb`
  - marimo source files retained:
    - `marimo_sources/exp076_exp039_cv_reassessment_train_marimo.py`
    - `marimo_sources/exp076_exp039_cv_reassessment_inference_marimo.py`
- Verification:
  - official notebook code-cell AST parse: pass.
  - `rg -n "marimo|mo\\."` on official and Kaggle-prepared train/inference notebooks: no matches.
  - `uv run python scripts/validate_experiment.py --experiment exp076_exp039_cv_reassessment`: pass.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook train --kernel-id kentookumura/exp076-exp039-cv-reassessment-train --title "exp076 exp039 cv reassessment train" --run-on-push --strict`: pass.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp076_exp039_cv_reassessment --notebook inference --kernel-id kentookumura/exp076-exp039-cv-reassessment-infer --title "exp076 exp039 cv reassessment infer" --run-on-push --strict`: pass.
- Not run:
  - no local notebook execution.
  - no Kaggle push or submission.

## 次のアクション

1. exp076 は完了。採用せず、比較材料として保持する。
2. 次は exp073 後処理 / public artifact replay integrity / long-tail gate の優先候補へ戻る。
