# exp013_model_diversity_or_postprocess セッションノート

## 目的

`exp012_single_catboost_lightgbm_residual` の `lightgbm_no_gr` を anchor とし、CV を崩さない保守的 postprocess と小さな HGB/LightGBM diversity 候補を同一 OOF split で比較する。

## 現在の状態

- 状態: inference / submit 完了
- 親実験: `exp012_single_catboost_lightgbm_residual`
- CV anchor: `exp012_single_catboost_lightgbm_residual` lightgbm_no_gr 13.549257
- Public LB anchor: `exp013_model_diversity_or_postprocess` distance_bucket_shrink 12.271
- selected variant: `lightgbm_no_gr`
- selected postprocess: `distance_bucket_shrink`
- raw CV: 13.549257
- selected postprocess OOF-fit score: 13.501824
- Public LB: 12.271 (`ref=53363702`)
- Review note: `distance_bucket_shrink_fit` の alpha は全 OOF の正解残差で fit し、同じ OOF rows で score している。13.501824 は nested / held-out CV ではないため、clean CV anchor は raw `lightgbm_no_gr` 13.549257 として扱う。提出済み postprocess は Public LB 12.271 を更新しており、形式上の再提出は不要。

## コマンドログ

- 2026-06-04: `uv run python scripts/new_steering.py --experiment exp013_model_diversity_or_postprocess` で steering docs を作成。
- 2026-06-04: `docs/legacy/steering/20260604-exp013-model-diversity-or-postprocess/{requirements.md,design.md,tasklist.md}` に仮説、設計、タスクを記入。
- 2026-06-04: `uv run python scripts/new_experiment.py --name exp013_model_diversity_or_postprocess --source experiments/exp012_single_catboost_lightgbm_residual` で exp012 から実験を作成。
- 2026-06-04: train / inference notebook を exp013 名にリネームし、`settings.py` の `EXPERIMENT_NAME` を更新。
- 2026-06-04: `config.yaml` を exp013 用に置換し、`control_hgb_no_gr` と `lightgbm_no_gr` の 2 variants、postprocess candidates、OOF 保存設定を定義。
- 2026-06-04: `baseline.py` に `smooth_prediction`、`distance_bucket_alphas`、`postprocess_predictions` を追加。
- 2026-06-04: train notebook に `row_oof_predictions.csv`、`postprocess_metrics.csv`、`postprocess_distance_bucket_summary.csv`、`postprocess_selected_params.json` の出力を追加。
- 2026-06-04: inference notebook に `postprocess.selected_method` による最終予測変換を追加。
- 2026-06-04: `uv run python scripts/validate_experiment.py --experiment exp013_model_diversity_or_postprocess` が通過。
- 2026-06-04: `uv run ruff check experiments/exp013_model_diversity_or_postprocess/baseline.py experiments/exp013_model_diversity_or_postprocess/settings.py` が通過。
- 2026-06-04: `uv run python -m py_compile experiments/exp013_model_diversity_or_postprocess/baseline.py experiments/exp013_model_diversity_or_postprocess/settings.py` が通過。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp013_model_diversity_or_postprocess --notebook train --strict` が通過。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp013_model_diversity_or_postprocess --notebook inference --strict` が通過。
- 2026-06-04: `uv run pytest` が通過。9 tests passed。
- 2026-06-04: `git status --short` は `fatal: not a git repository` で失敗。このワークスペースでは git repository として認識されなかった。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp013_model_diversity_or_postprocess --notebook train --run-on-push --title "exp013 model diversity or postprocess train" --strict` で train package を再生成。
- 2026-06-04: `kaggle kernels push -p experiments/exp013_model_diversity_or_postprocess/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp013-model-diversity-or-postprocess-train
- 2026-06-04: `kaggle kernels status kentookumura/exp013-model-diversity-or-postprocess-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-04: `kaggle kernels output kentookumura/exp013-model-diversity-or-postprocess-train -p /tmp/kaggle-output/exp013_model_diversity_or_postprocess/train` で output を取得。
- 2026-06-04: Kaggle output の `metrics.json`、小さい `artifacts/*.csv`、`postprocess_selected_params.json`、train log を `experiments/exp013_model_diversity_or_postprocess/` に反映。`row_oof_predictions.csv` は 1.1GB のため実験ディレクトリには常設しない。
- 2026-06-05: `row_oof_predictions.csv` を `/tmp` から `data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/row_oof_predictions.csv` に移動。`exp014` / `exp016` の OOF 監査入力はこの永続ローカルパスを参照する。
- 2026-06-04: `distance_bucket_shrink_fit` が OOF-fit score 13.501824 で best。`config.yaml` の `postprocess.selected_method` と bucket alpha を固定。
- 2026-06-04: `uv run python scripts/validate_experiment.py --experiment exp013_model_diversity_or_postprocess` が通過。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp013_model_diversity_or_postprocess --notebook inference --strict` が通過。
- 2026-06-04: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` を更新。
- 2026-06-04: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp013_model_diversity_or_postprocess --notebook inference --run-on-push --title "exp013 distance bucket shrink inference" --strict` で inference package を再生成。
- 2026-06-04: `kaggle kernels push -p experiments/exp013_model_diversity_or_postprocess/kaggle/inference` で version 1 を push。metadata id と title slug の warning が出たため、実際の slug `kentookumura/exp013-distance-bucket-shrink-inference` を使って監視。
- 2026-06-04: `kaggle kernels status kentookumura/exp013-distance-bucket-shrink-inference` で `KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-04: `kaggle kernels output kentookumura/exp013-distance-bucket-shrink-inference -p /tmp/kaggle-output/exp013_model_diversity_or_postprocess/inference` で `submission.csv` と inference log を取得。
- 2026-06-04: `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp013_model_diversity_or_postprocess/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。14,151 rows、`id,tvt`、欠損/重複なし。
- 2026-06-04: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp013-distance-bucket-shrink-inference -v 1 -f submission.csv -m "exp013_distance_bucket_shrink CV 13.501824"` を実行。
- 2026-06-04: submission ref `53363702` が `SubmissionStatus.COMPLETE`、Public LB 12.271 と確認。従来 anchor `exp012` 12.320 を更新。
- 2026-06-04: `uv run python scripts/record_experiment.py --experiment exp013_model_diversity_or_postprocess --status completed --cv 13.501824 --public-lb 12.271 ...` で metrics と summary を更新。
- 2026-06-04: `uv run python scripts/record_submission.py --experiment exp013_model_diversity_or_postprocess --file /tmp/kaggle-output/exp013_model_diversity_or_postprocess/inference/submission.csv --cv 13.501824 --public-lb 12.271 ...` で `SUBMISSIONS.md` に v008 を記録。

## 変更点

- `lightgbm_no_gr`: exp012 selected LightGBM no-GR residual model を raw anchor として再実行する。
- `control_hgb_no_gr`: HGB no-GR を diversity control として再実行する。
- `raw_lightgbm_no_gr`: raw anchor の OOF score。
- `sg_smooth`: well 内 raw prediction の smoothing。
- `global_residual_shrink`: `last_anchor + alpha * residual` の固定 alpha 比較。
- `near_anchor_damping`: hidden segment 開始近傍を anchor 寄りに damp する。
- `distance_bucket_shrink`: OOF で row 距離 bucket ごとの alpha を fit する。fit と評価が同一 OOF rows なので、clean CV としては扱わない。
- `hgb_lightgbm_blend`: HGB no-GR と LightGBM no-GR の constrained blend を小さく比較する。

## 次のアクション

1. Public LB anchor は `exp013` 12.271、clean CV anchor は raw `lightgbm_no_gr` 13.549257 として分けて扱う。
2. 次は `exp014_postprocess_cv_audit` で、bucket alpha を outer fold / leave-one-fold-out / held-out bucket で fit し直して、postprocess の改善が clean CV でも残るか確認する。
3. exp008-011 の NCC / formation / trajectory / tracker add-only 系は再実行不要。再検討は PF/beam 候補ができた後の routing / confidence / pruning に限定する。
