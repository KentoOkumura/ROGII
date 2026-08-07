# exp068_equivalent_pixiux_inference_port 結果

## 状態

- status: discarded
- route: `ml_model`
- Kaggle train: v4 completed for CV only; full model artifact revision discarded before rerun
- Kaggle inference: no further rerun; experiment discarded
- submit-check: historical v2 PASS only; current full-model flow not rerun
- Public LB: no valid LB; ref `53654439` is explicitly not adopted

## 破棄理由

2026-06-16 にユーザー指示でこの実験は破棄した。exp068 は `exp063` を対象に exp039 CV surface で再評価する実験だったが、同じ意図は `exp073` を対象にした backlog `exp073_exp039_cv_reassessment` として作り直す。

既存の CV-only train v4 と invalid submission ref `53654439` は履歴として残すが、exp068 の修正版 Kaggle train / inference / submit は実行しない。

## 評価内容

`exp039 型 branch` の価値を `exp063` 上で再評価する。train 側は、exp039/exp038 系の CV surface に exp063 の tracker/PF/Beam output features を `id` で join し、exp063 の Pixiux LightGBM config family を exp039 CV で再学習評価する。PF/Beam 再生成は行わない。

## 実装済み生成物

Kaggle train 実行後に次を保存する。

- `exp063_model_exp039_cv_metrics.csv`
- `exp063_model_exp039_cv_by_well.csv`
- `exp063_model_exp039_cv_predictions.csv.gz`
- `exp063_model_exp039_cv_summary.json`
- `exp068_exp039_cv_full_lgb_models/manifest.json`
- `exp068_exp039_cv_full_model_feature_schema.csv`
- `exp068_exp039_cv_full_model_feature_importance.csv`

Kaggle inference 実行後に次を保存する。

- `submission.csv`
- `exp068_exp039_cv_full_model_inference_predictions.csv.gz`
- `exp068_exp039_cv_full_model_inference_metrics.csv`
- `exp068_exp039_cv_full_model_inference_summary.json`
- `exp063_branch_submission_diff.csv`

## 解釈

Kaggle train v4 で exp039/exp038 系 CV surface 上の再学習評価が完了した。

| audit | lgb0 | lgb1 | lgb2 | lgb_mean |
| --- | ---: | ---: | ---: | ---: |
| leave_one_original_fold_out | 12.112706 | 11.918170 | 11.930688 | 11.878856 |
| well_hash_holdout | 12.207261 | 12.023019 | 12.017439 | 11.994729 |

join は exp039 rows 1,782,279 に対して 1,781,963 rows 成功し、316 rows が落ちた。features は 65、joined wells は 773。

この評価面では exp063 Pixiux LightGBM family は exp039 系 single-LGBM anchor より大きく強い。一方で exp063 自身の strict public replay CV 9.630105 とは評価面が違うため、ML route / 全体 route の anchor 更新根拠にはしない。

Kaggle inference v2 は exp063 inference prediction artifact の `pixiux_likpf_public_replay` / `lgb_mean` から `submission.csv` を生成した。14,151 rows、fallback 0、SHA256 `26e3238a29ff37d4193cfec073d507fc840082b33fd82be10a0cc619302739c4`、予測範囲 11593.675 - 12240.099。submit-check は PASS。

ローカルで exp063 inference v2 の `submission.csv` と比較した差分は id mismatch 0、diff RMSE 0.000277、max abs diff 0.000484 で、丸め差程度に同一。したがって inference output は実質 exp063 direct inference と同じであり、新規 Public LB 提出価値は低い。

2026-06-14 に提出 ref `53654439` が complete となり、Public LB は `762.715`。これは上記の exp063 inference v2 とのローカル同等性から期待される `8.811` 近傍と大きく矛盾する。レビュー後の判断では、exp068 inference v2 が静的な exp063 public-sample prediction artifact を kernel source として読んでおり、code-submission hidden scoring では hidden id に対する prediction が存在せず fallback した可能性が高い。この LB は exp068 手法の性能として採用しない。

## 2026-06-14 修正方針

目的を「exp068 再学習モデルの提出」に固定した。修正版では、train notebook が exp039 CV surface で CV 評価した後、全 joined rows で `lgb0/lgb1/lgb2` の full LightGBM boosters を `exp068_exp039_cv_full_lgb_models/manifest.json` として保存する。inference notebook は exp068 train output の full boosters を読み、hidden test 上で exp063 replay feature generation code を実行して特徴量を作成し、静的な exp063 inference prediction artifact は使わない。sample id と prediction id が一致しない場合は fallback せず fail する。

この修正後の Kaggle train / inference / submit-check / LB は未実行のまま、2026-06-16 のユーザー指示により exp068 は破棄した。
