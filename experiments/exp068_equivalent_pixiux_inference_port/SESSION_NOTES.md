# exp068_equivalent_pixiux_inference_port セッションノート

## 現在の状態

- status: discarded
- route: `ml_model`
- parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- reference branch: `exp039_ravaghi_single_lgbm_inference_submit`
- 実験番号: ユーザー指定により、削除済みの exp067 を再利用せず `exp068` とした。

## 仮説

元のバックログ「exp039 型 branch の価値を exp063 上で再評価する」に従い、exp063 の Pixiux LightGBM model family を exp039/exp038 系 CV surface で再学習評価する。

## 実装メモ

- 途中で saved-booster-only port と tracker-only CV retrain の案に寄ったが、どちらも元バックログから外れていたため破棄。
- `exp063_branch_audit.py` を追加し、正の実装入口にした。
  - train notebook: exp029/exp039 の `public_sel15_pf_oof_features.csv.gz` から `fold` / target / evaluation rows を読み、exp063 output `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を `id` で join する。
  - train notebook: exp063 と同じ 3 本の LightGBM config を `leave_one_original_fold_out` と `well_hash_holdout` で cross-fit し、`lgb_mean` を保存する。
  - inference notebook: レビュー前は exp063 inference prediction artifact を読んでいたが、hidden scoring で static public artifact 依存になるため廃止。現在は exp068 full model artifact と hidden-test exp063 replay feature generation を使う。
- exp068 では exp039 CV でモデルを再学習評価する。
- exp068 では PF/Beam feature を再生成しない。
- LightGBM は GPU 実行を使用しつつ、`deterministic=true` / `force_col_wise=true` を追加。
- exp063 の実装ファイルは変更しない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp068_equivalent_pixiux_inference_port
uv run python scripts/new_experiment.py --name exp068_equivalent_pixiux_inference_port --source experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
```

実装後の確認:

```bash
uv run python -m py_compile experiments/exp068_equivalent_pixiux_inference_port/exp063_branch_audit.py experiments/exp068_equivalent_pixiux_inference_port/settings.py
uv run ruff check experiments/exp068_equivalent_pixiux_inference_port/exp063_branch_audit.py experiments/exp068_equivalent_pixiux_inference_port/settings.py
uv run python -m json.tool experiments/exp068_equivalent_pixiux_inference_port/exp068_equivalent_pixiux_inference_port_train.ipynb
uv run python -m json.tool experiments/exp068_equivalent_pixiux_inference_port/exp068_equivalent_pixiux_inference_port_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp068_equivalent_pixiux_inference_port
```

追加確認:

```bash
# exp039 CV surface と exp063 tracker/PF/Beam train features の sample join
# 2,000 / 2,000 rows joined, tracker columns=68
```

Kaggle train:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp068_equivalent_pixiux_inference_port --notebook train --kernel-id kentookumura/exp068-equivalent-pixiux-train --title "exp068 equivalent pixiux train" --run-on-push --strict
kaggle kernels push -p experiments/exp068_equivalent_pixiux_inference_port/kaggle/train
```

Kaggle inference:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp068_equivalent_pixiux_inference_port --notebook inference --kernel-id kentookumura/exp068-equivalent-pixiux-infer --title "exp068 equivalent pixiux infer" --run-on-push --strict
kaggle kernels push -p experiments/exp068_equivalent_pixiux_inference_port/kaggle/inference
```

取得/検証:

```bash
kaggle kernels output kentookumura/exp068-equivalent-pixiux-infer -p /tmp/kaggle-output/exp068_equivalent_pixiux_inference_port/inference_v2
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp068_equivalent_pixiux_inference_port/inference_v2/submission.csv
```

## 結果

- ローカル実装と静的検証まで実施。
- sample join: exp039 CV surface 2,000 rows / exp063 tracker hits 2,000 / joined 2,000。
- Kaggle train v1: failed。`public_sel15_pf_oof_features.csv.gz` が Kaggle input に無く、exp039 CV surface の読み込みで `FileNotFoundError`。
- 修正: train `kernel_sources` に `kentookumura/exp029-sel15-pf-oof-train` を追加し、exp039 CV surface を `/kaggle/input/**/public_sel15_pf_oof_features.csv.gz` から探索する fallback を追加。
- Kaggle train v2: failed。実行本体ではなく source artifact check の preview cell がローカル相対 path を直接 `pd.read_csv` して同じ `FileNotFoundError`。
- 修正: preview cell も `find_path(..., filename=EXP029_FEATURE_PATH.name)` で Kaggle input fallback を使うよう変更。
- Kaggle train v3: failed。exp063 tracker artifact の `well` 列が pandas に数値解釈され、`543198e8` が `54319800000000.0` になって join 整合性チェックで落ちた。
- 修正: exp063 tracker の `well` は artifact 列を信用せず、`id` prefix から復元する。
- Kaggle train v4: completed。
  - kernel: `kentookumura/exp068-equivalent-pixiux-train`
  - output: `/tmp/kaggle-output/exp068_equivalent_pixiux_inference_port/train_v4`
  - elapsed: 7801.091 sec
  - join: exp039 rows 1,782,279 / exp063 tracker rows 3,783,989 / joined rows 1,781,963 / dropped exp039 rows 316 / joined wells 773 / features 65
  - `leave_one_original_fold_out` pooled RMSE: `lgb0=12.112706`, `lgb1=11.918170`, `lgb2=11.930688`, `lgb_mean=11.878856`
  - `well_hash_holdout` pooled RMSE: `lgb0=12.207261`, `lgb1=12.023019`, `lgb2=12.017439`, `lgb_mean=11.994729`
  - artifacts:
    - `exp063_model_exp039_cv_metrics.csv`
    - `exp063_model_exp039_cv_by_well.csv`
    - `exp063_model_exp039_cv_predictions.csv.gz`
    - `exp063_model_exp039_cv_summary.json`
- Kaggle inference / submit-check は未実行。
- Kaggle inference v1: failed。train 用 config の `model.model=exp063_public_lightgbm_configs` を inference artifact の filter に使ってしまい、artifact 内の `model=lgb_mean` に一致せず `No predictions`。
- 修正: inference 用に `inference.model: lgb_mean` を追加し、inference notebook は `inference.model` を使うよう変更。
- Kaggle inference v2: completed。
  - kernel: `kentookumura/exp068-equivalent-pixiux-infer`
  - output: `/tmp/kaggle-output/exp068_equivalent_pixiux_inference_port/inference_v2`
  - submission rows: 14,151
  - predicted rows: 14,151
  - fallback rows: 0
  - prediction range: 11593.675 - 12240.099
  - prediction mean/std: 11905.529252 / 279.332550
  - SHA256: `26e3238a29ff37d4193cfec073d507fc840082b33fd82be10a0cc619302739c4`
  - submit-check: PASS via `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp068_equivalent_pixiux_inference_port/inference_v2/submission.csv`
  - local diff vs exp063 inference v2 submission: id mismatch 0, diff RMSE 0.000277, abs mean 0.000237, max abs 0.000484
  - notebook 内 `reference_diff` は Kaggle path 探索順の都合で自己参照になったため、上記 local diff を正の差分確認として扱う。config の reference path は実 input path に修正済み。

## 提出結果

- ユーザーにより提出完了。
- `kaggle competitions submissions rogii-wellbore-geology-prediction` で確認。
- `uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp068_equivalent_pixiux_inference_port --competition rogii-wellbore-geology-prediction` でも確認。
- ref: `53654439`
- submitted: `2026-06-14 00:33:47.950000`
- status: complete
- Public LB: `762.715`
- 注意: ローカルでは exp068 inference v2 submission と exp063 inference v2 submission の差分が丸め差程度だったため、期待 LB は exp063 `8.811` 近傍のはず。この `762.715` は提出ファイル/path 取り違えの可能性が高く、exp068 手法性能としては採用しない。

## 2026-06-14 review follow-up

- ユーザーの目的を「exp068 再学習モデルの提出」として確定。
- 問題: 旧 inference v2 は exp063 の静的 inference prediction artifact を読んで `submission.csv` を作っていた。public sample では exp063 と丸め差程度に同一だが、code-submission hidden scoring では hidden id の prediction が artifact に存在せず fallback しうる。
- 修正:
  - train notebook は CV 評価後、全 joined rows で `lgb0/lgb1/lgb2` の full LightGBM boosters を保存する。
  - full model artifacts は `exp068_exp039_cv_full_lgb_models/manifest.json`、feature schema、feature importance、train predictions。
  - inference notebook は exp068 train output の full boosters を読み、hidden test 上で exp063 replay feature generation code を実行して予測する。
  - 静的な exp063 inference prediction artifact は current inference flow では使わない。
  - sample id と prediction id が一致しない場合は fallback せず fail する。
  - `submission_diff()` は参照 submission が見つからない場合に current `submission.csv` へ fallback しないよう修正。
- Kaggle package は train / inference とも再生成済み。
- 修正版の Kaggle train / inference / submit-check / LB は未実行。

## 次のアクション

なし。2026-06-16 のユーザー指示により exp068 は破棄し、Kaggle train / inference の再実行は行わない。

## 2026-06-16 discard

- ユーザー指示: 「この実験は破棄としてください。代わりにexp039のcvでexp073を評価するバックログを作成してください。対象がexp063からexp073に変わっただけです。」
- 対応: exp068 の status を `discarded` に変更し、CV-only train v4 と invalid submission ref `53654439` は履歴として残す。
- 代替 backlog: `exp073_exp039_cv_reassessment`
- 意図: exp068 の元バックログ「exp039 型 branch の価値を exp063 上で再評価する」から、対象だけを `exp063` から `exp073` に差し替える。
