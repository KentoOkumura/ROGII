# exp088_sequence_model_residual_diversity 結果

## 状態

- status: `completed`
- route: `ml_model`
- CV: best OOF RMSE `9.509990853500197`
- baseline OOF RMSE: `9.524813476455735`
- Public LB: なし
- Submit: なし

## 結果

exp073 OOF `lgb_mean` を baseline に、GRU / TCN の residual correction を fold-out で作った。

| prediction | RMSE | delta vs baseline |
| --- | ---: | ---: |
| baseline_pred_tvt | 9.524813 | 0.000000 |
| gru32_ctx64_pred_tvt | 10.499027 | +0.974214 |
| tcn32_ctx64_pred_tvt | 10.377744 | +0.852930 |
| alpha_blend_gru32_ctx64_pred_tvt | 9.524813 | 0.000000 |
| alpha_blend_tcn32_ctx64_pred_tvt | 9.524813 | 0.000000 |
| ridge_blend_pred_tvt | 9.509991 | -0.014823 |

## 解釈

sequence NN 単体は baseline より明確に悪い。alpha blend は最適 alpha が 0.0 になり、GRU / TCN を足す価値は出なかった。

ridge blend は `-0.0148` RMSE 改善したが、重みは baseline `1.115`、GRU `-0.150`、TCN `0.035` で、sequence 予測を本命枝として採用するというより、baseline error と高相関な弱い補正を回帰で吸収した結果と見るのが妥当。

distance bucket では、2500+ rows だけ `-0.0307` 改善し、近距離 / 中距離では同等か悪化した。改善幅は小さく、Kaggle inference port や提出候補化はしない。

## 生成物

Kaggle v4 / T4 で完了。output は `experiments/exp088_sequence_model_residual_diversity/kaggle/output/train/artifacts/` に取得済み。

- `exp088_sequence_model_residual_diversity_oof_predictions.csv.gz`
- `exp088_sequence_model_residual_diversity_metrics.csv`
- `exp088_sequence_model_residual_diversity_bucket_metrics.csv`
- `exp088_sequence_model_residual_diversity_diversity_metrics.csv`
- `exp088_sequence_model_residual_diversity_train_history.csv`
- `exp088_sequence_model_residual_diversity_rmse.png`
- `exp088_sequence_model_residual_diversity_summary.json`

主要 SHA:

- OOF prediction content: `27fac6bfff8953d5ad770cef2e69e1d4871d28c9f9b4bbe4afd6219bdee807c3`
- metrics: `41a823a93c573161a46594790c70aaecb26fa0ed652c35c86caf6f3d03223ffa`
- input exp073 OOF content: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- input exp072 feature cache content: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`

## 次アクション

sequence residual route はここで閉じる。long-tail 改善の信号は弱いため、次は `pf_beam_disagreement_sample_weight` / confidence feature 系に優先度を寄せる。
