# exp021_distance_weighted_inference_postprocess セッションノート

## 目的

バックログ先頭の `distance_weighted_inference_postprocess` を実装する。`exp020` で選ばれた `near_down_far_up_lightgbm` を inference notebook に反映し、weighted raw と weighted + exp014 bucket shrink を比較できる OOF 監査と提出候補生成を用意する。

## 現在の状態

- 状態: Kaggle train / inference / submit 完了
- 親実験: `exp020_distance_weighted_training_audit`
- Parent clean CV: `near_down_far_up_lightgbm` 13.470015
- Raw clean CV anchor: `exp013 lightgbm_no_gr` 13.549257
- Held-out postprocess reference: `exp014` leave-one-original-fold-out 13.535596
- Public LB anchor: `exp013 distance_bucket_shrink` 12.271

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp021_distance_weighted_inference_postprocess` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp021_distance_weighted_inference_postprocess --source experiments/exp020_distance_weighted_training_audit` で exp020 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics を exp021 用に更新。
- 2026-06-06: `distance_weighted_inference_postprocess.py` を追加し、selected weighted fit、weighted OOF postprocess audit、weighted inference submission 生成を実装。
- 2026-06-06: train / inference notebook を exp021 用のセル構成に再生成。
- 2026-06-06: `uv run python -m py_compile experiments/exp021_distance_weighted_inference_postprocess/distance_weighted_inference_postprocess.py experiments/exp021_distance_weighted_inference_postprocess/baseline.py experiments/exp021_distance_weighted_inference_postprocess/settings.py` が通過。
- 2026-06-06: `uv run ruff check experiments/exp021_distance_weighted_inference_postprocess/distance_weighted_inference_postprocess.py experiments/exp021_distance_weighted_inference_postprocess/baseline.py experiments/exp021_distance_weighted_inference_postprocess/settings.py` が通過。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp021_distance_weighted_inference_postprocess` が通過。
- 2026-06-06: notebook code cell compile が train / inference とも通過。
- 2026-06-06: `uv run python scripts/update_experiment_summary.py` で exp021 を `experiment_summary.md` に追加。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp021_distance_weighted_inference_postprocess --notebook train --run-on-push --title "exp021 distance weighted inference postprocess train" --strict` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp021_distance_weighted_inference_postprocess --notebook inference --run-on-push --title "exp021 distance weighted inference postprocess inference" --strict` が通過。
- 2026-06-06: 長い default kernel slug での train push は Kaggle API 400 になったため、`--kernel-id kentookumura/exp021-dw-post-train --title "exp021 dw post train"` で再生成。
- 2026-06-06: `kaggle kernels push -p experiments/exp021_distance_weighted_inference_postprocess/kaggle/train` で train version 2 を push。URL: https://www.kaggle.com/code/kentookumura/exp021-dw-post-train
- 2026-06-06: `kaggle kernels output kentookumura/exp021-dw-post-train -p /tmp/kaggle-output/exp021_distance_weighted_inference_postprocess/train` で train output を取得。`kaggle kernels status` は Kaggle API 500 を返したが、output は取得できた。
- 2026-06-06: train result は `weighted_raw` 13.470015、`weighted_distance_bucket_shrink` 13.415799。小さい artifact と log を `artifacts/` に保存し、328MB の `weighted_oof_predictions.csv` は `/tmp/kaggle-output/...` のみ保持。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp021_distance_weighted_inference_postprocess --notebook inference --kernel-id kentookumura/exp021-dw-post-infer --run-on-push --title "exp021 dw post infer" --strict` が通過。
- 2026-06-06: `kaggle kernels push -p experiments/exp021_distance_weighted_inference_postprocess/kaggle/inference` で inference version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp021-dw-post-infer
- 2026-06-06: `kaggle kernels output kentookumura/exp021-dw-post-infer -p /tmp/kaggle-output/exp021_distance_weighted_inference_postprocess/inference` で inference output を取得。
- 2026-06-06: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp021_distance_weighted_inference_postprocess/inference/submission.csv` が通過。submission は `data/external/kaggle-output/exp021_distance_weighted_inference_postprocess/inference/submission.csv` に保存。SHA256 `f0e1289b28453b558978ebc48986fa4fd3a85d1ba05299e4455fca0b4a00611f`。
- 2026-06-06: file upload submit は Notebook-only code competition のため Kaggle API 400。`kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp021-dw-post-infer -v 1 -f submission.csv -m "exp021 weighted_distance_bucket_shrink CV 13.415799"` で code competition 形式の提出に成功。
- 2026-06-06: `uv run python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp021_distance_weighted_inference_postprocess --competition rogii-wellbore-geology-prediction --poll-seconds 60 --timeout-minutes 30` で監視。ref `53406803`、Public LB 12.523、Private LB 未表示。
- 2026-06-06: `uv run python scripts/record_submission.py --experiment exp021_distance_weighted_inference_postprocess --file data/external/kaggle-output/exp021_distance_weighted_inference_postprocess/inference/submission.csv --cv 13.415799 --public-lb 12.523 --notes "ref=53406803; kernel=kentookumura/exp021-dw-post-infer v1; selected=weighted_distance_bucket_shrink; CV improved but Public LB worse than exp013 12.271"` で提出履歴 v009 を記録。

## 変更点

- exp020 の selected weight profile を `audit.training_variants.selected_variant` に固定。
- final inference training で selected sample weights を使う。
- train notebook で selected weighted OOF を生成し、raw / distance bucket shrink を bucket 別に監査する。
- inference notebook で selected weighted model を train wells 全体に fit し、`postprocess.selected_method` を適用して `submission.csv` を作る。

## 次のアクション

1. Public LB anchor は exp013 12.271 のまま維持する。
2. `distance_uncertainty_shrink` で CV 改善が Public LB に出ない理由を距離・tail・uncertainty 別に切り分ける。
3. exp021 は clean CV anchor として残し、LB だけに合わせた追加調整はしない。
