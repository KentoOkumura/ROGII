# exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit 結果

## 状態

- status: `completed`
- route: `ml_model`
- Kaggle kernel: `kentookumura/exp063-ravaghi-pixiux-strict-replay-train` version 4 complete
- Kaggle inference kernel: `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer` version 2 complete
- Public LB: 8.811 (`ref=53632725`)
- Inference: version 1 は train feature 再生成設計だったため手動停止。version 2 は saved booster + test-only 設計で完了、submit-check PASS。

## 評価

- mode: strict public notebook raw replay
- rows / wells: 3,783,989 / 773
- feature generation: 13,034.108 sec
- total runtime: 27,742.572 sec
- feature count:
  - `ravaghi_public_lgbm_replay`: 195
  - `pixiux_likpf_public_replay`: 196

Pooled GroupKFold OOF RMSE:

- `pixiux_likpf_public_replay` `lgb2`: 9.628965
- `pixiux_likpf_public_replay` `lgb_mean`: 9.630105
- `pixiux_likpf_public_replay` `lgb1`: 9.669757
- `pixiux_likpf_public_replay` `lgb0`: 9.846804
- `ravaghi_public_lgbm_replay` `lgb_mean`: 10.560537
- `ravaghi_public_lgbm_replay` best single model `lgb2`: 10.538333

Delta:

- Pixiux `lgb_mean` vs Ravaghi `lgb_mean`: -0.930432
- Pixiux best `lgb2` vs best Ravaghi single LGBM: -0.909367

## 生成物

- `artifacts/ravaghi_vs_pixiux_public_replay_metrics.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_feature_importance.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_feature_importance_mean.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_feature_importance_mean_top.png`
- `artifacts/ravaghi_vs_pixiux_public_replay_oof_predictions.csv.gz`
- `artifacts/ravaghi_vs_pixiux_public_replay_feature_schema.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_summary.json`
- `artifacts/exp063-ravaghi-pixiux-strict-replay-train.log`
- `artifacts/ravaghi_vs_pixiux_public_replay_lgb_models/manifest.json`
- `artifacts/ravaghi_vs_pixiux_public_replay_lgb_models/*.txt` (15 LightGBM boosters)
- `artifacts/ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz`

Inference 生成物:

- `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2/submission.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_inference_metrics.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_inference_test_predictions.csv.gz`
- `artifacts/ravaghi_vs_pixiux_public_replay_inference_feature_schema.csv`
- `artifacts/ravaghi_vs_pixiux_public_replay_inference_summary.json`
- `artifacts/ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz`
- `artifacts/exp063-ravaghi-pixiux-strict-replay-infer.log`

Inference version 2:

- elapsed: 127.648 sec
- feature generation: 98.101 sec
- test wells / rows: 3 / 14,151
- saved model count: 15
- submission rows / predicted rows / fallback rows: 14,151 / 14,151 / 0
- prediction range: 11,593.674805 - 12,240.098633
- prediction mean / std: 11,905.529255 / 279.332552
- submission sha256: `36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499`
- submit-check: PASS
- code submission: `ref=53632725`, status COMPLETE, Public LB 8.811

## 解釈

旧 artifact audit ではなく、公開 notebook 由来の raw feature replay でも Pixiux likelihood-PF feature は明確に有効だった。
公開実装の feature selection では `likpf_mean_d` が実質的な add-only feature になり、平均 feature importance でも Pixiux 側の最上位だった。

追加で inference notebook を実装した。
初回 version 1 は train features を inference 内で再生成する非効率な設計だったため、ユーザーが手動停止した。
修正版では train notebook が `pixiux_likpf_public_replay` の 3 configs x 5 folds の LightGBM booster を保存し、
inference notebook は保存済み booster を読み、test-side replay features だけを生成して `submission.csv` を保存する。
後続実験で再利用できるように、PF/Beam/likelihood-PF tracker feature frame も `id` join 用の csv.gz として保存する。

この inference port には hidden-specific branch、guarded overlap override、static visible override、
pretrained booster、CatBoost、Ridge stack、final public notebook blend、projection postprocess は含めない。
Public LB 8.811 は ML route anchor の exp039 11.740 から -2.929 改善したため、ML route の Public LB 基準を更新する。
一方、全体 / PF route anchor の exp027 8.781 には +0.030 届かないため、全体基準は exp027 のまま維持する。

## 再現性メモ

exp063 の train-side CV は GPU rerun で bitwise 再現されていない。
`train_v3` と `train_v4` は feature schema は同一だったが、`pixiux_likpf_public_replay` `lgb_mean` pooled OOF RMSE は
`9.599138098927096` から `9.630105123038494` に変わり、OOF prediction content SHA も不一致だった。

主な原因候補は LightGBM GPU 実行の `device_type="gpu"` / `gpu_use_dp=false` / `n_jobs=-1` と、
Numba likelihood-PF を `joblib` thread 並列で回している点である。
PyTorch/CuDNN 向けの `torch.manual_seed`、`torch.backends.cudnn.deterministic=True` ではこの経路は制御できない。

提出済みの Public LB 8.811 は `submission.csv` SHA
`36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499` として提出物単位では固定されている。
ただし CV の細かい差分比較では exp063 v4 の値を単発 GPU 実行値として扱い、必要なら CPU deterministic rerun か saved booster/submission SHA 比較を優先する。
