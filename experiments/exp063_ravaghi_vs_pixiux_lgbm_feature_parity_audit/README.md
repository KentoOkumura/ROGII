# exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit

## 状態

- ルート: `ml_model`
- 状態: `completed`
- CV: `pixiux_likpf_public_replay` best `lgb2` 9.628965 / ensemble `lgb_mean` 9.630105
- Public LB: 8.811 (`ref=53632725`)
- Inference: version 2 complete with saved booster + test-only feature generation; submit-check PASS
- 作成日: 2026-06-12
- 親: public notebooks

## 仮説

Pixiux dual-pipeline blend notebook の likelihood-PF replay features は、
Ravaghi LightGBM public notebook の base feature set に対する add-only
LightGBM feature として OOF RMSE を改善する可能性がある。

## 検証方針

旧 exp063 の exp056 artifact ベース audit は公開ノートブック再現として不十分だったため無効化した。
この版では `public_notebook_replay_audit.py` が competition raw train files から
公開ノートブック由来の `build_features` / `build_likpf` / `add_likpf_features` を再生する。

比較対象:

- `ravaghi_public_lgbm_replay`: Ravaghi-style public base features
- `pixiux_likpf_public_replay`: base features + Pixiux public likelihood-PF delta features

学習は公開 LightGBM 3 configs の GroupKFold OOF のみを実行する。
CatBoost、Ridge stack、final blend、projection、pretrained booster、static visible override は含めない。

Inference notebook は train audit で選ばれた `pixiux_likpf_public_replay` `lgb_mean` の
保存済み fold booster を読み、test-side replay features だけを生成して `submission.csv` を作る。
hidden-specific branch、guarded overlap override、static visible override、pretrained booster、
CatBoost、Ridge stack、final public notebook blend、projection postprocess は含めない。
後続実験で再利用できるように、PF/Beam/likelihood-PF tracker feature frame も `id` join 用の csv.gz として保存する。

保存する生成物:

- `ravaghi_vs_pixiux_public_replay_metrics.csv`
- `ravaghi_vs_pixiux_public_replay_feature_importance.csv`
- `ravaghi_vs_pixiux_public_replay_feature_importance_mean.csv`
- `ravaghi_vs_pixiux_public_replay_feature_importance_mean_top.png`
- `ravaghi_vs_pixiux_public_replay_oof_predictions.csv.gz`
- `ravaghi_vs_pixiux_public_replay_feature_schema.csv`
- `ravaghi_vs_pixiux_public_replay_summary.json`
- `ravaghi_vs_pixiux_public_replay_lgb_models/manifest.json`
- `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz`

Inference 実行時に保存する生成物:

- `submission.csv`
- `ravaghi_vs_pixiux_public_replay_inference_metrics.csv`
- `ravaghi_vs_pixiux_public_replay_inference_test_predictions.csv.gz`
- `ravaghi_vs_pixiux_public_replay_inference_feature_schema.csv`
- `ravaghi_vs_pixiux_public_replay_inference_summary.json`
- `ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz`

## 所見

Kaggle version 4 で strict public raw replay が完了した。
rows / wells は 3,783,989 / 773、total runtime は 27,742.572 sec。

`pixiux_likpf_public_replay` は best single `lgb2` OOF RMSE 9.628965、ensemble `lgb_mean` 9.630105 で、
`ravaghi_public_lgbm_replay` `lgb_mean` 10.560537 から -0.930432 改善した。
平均 feature importance plot では、公開実装の add-only likelihood-PF feature である
`likpf_mean_d` が Pixiux 側の最上位だった。

追加で inference port を実装した。初回 Kaggle inference version 1 は train features を再生成する設計だったため手動停止した。
修正版は saved booster + test-only feature generation の直接 submission 化であり、公開 notebook の final blend や visible override は再現しない。
train version 4 は 15 LightGBM boosters と再利用用 PF/Beam/likelihood-PF tracker train features を保存済み。
Kaggle inference version 2 は 127.648 sec で完了し、14,151 rows の `submission.csv` を生成した。
prediction range は 11,593.674805 - 12,240.098633、fallback rows は 0、sha256 は
`36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499`。
Code submission `ref=53632725` は COMPLETE、Public LB は 8.811。
これは ML route anchor の exp039 11.740 から -2.929 改善であり、ML route の Public LB 基準を更新する。
全体 / PF route anchor の exp027 8.781 には +0.030 届かないため、全体基準は exp027 のまま維持する。
