# exp111_learned_pf_observation_likelihood_probe 結果

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

候補集合には oracle headroom があるが、row-wise selector は `likpf_mean` 単体を超えなかった。候補を直接選ぶのではなく、各候補の observation likelihood を校正すれば、PF weight ablation や ML add-only feature に使える confidence signal になる可能性がある。

## 設定

- 親: `exp099_pf_multi_observation_likelihood_probe`
- 入力: exp099 v2 wide train feature cache
- 候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`
- 検証: GroupKFold by `well`、first fold smoke
- 学習器:
  - LightGBM binary within10 classifier
  - LightGBM L1 expected error regressor

## 判定基準

提出候補化はしない。次へ進む条件は、learned likelihood が exp099 の hand-crafted `multiobs_score` より candidate within10 AUC / calibration / topK coverage を改善すること。改善した場合も、次は PF-weight alpha ablation または ML add-only feature audit に進む。

## Kaggle train v1

- Kernel: `kentookumura/exp111-learned-pf-likelihood-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp111-learned-pf-likelihood-train`
- status: `COMPLETE`
- runtime: 402.23 sec
- rows: 3,783,989
- candidate rows: 3,788,690
- wells: 773
- output: `kaggle/output/train_v1`
- GPU: false
- internet: false

## 結果

| variant | AUC | logloss | brier | observed within10 | pred mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `learned_within10_probability` | 0.913327 | 0.356157 | 0.115817 | 0.470412 | 0.485616 |
| `baseline_multiobs_score` | 0.617355 | - | - | 0.470412 | - |
| `baseline_multiobs_ncc` | 0.500422 | - | - | 0.470412 | - |
| `baseline_negative_multiobs_mae` | 0.652704 | - | - | 0.470412 | - |

learned likelihood は exp099 hand-crafted `multiobs_score` より AUC が `+0.295972` 高い。summary recommendation は `likelihood_supported_for_pf_weight_or_feature_followup`。

diagnostic top1 は以下。

| variant | RMSE | MAE | within10 | pf_ancc selection |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean_single` | 11.604410 | 6.944251 | 0.784312 | 0.000000 |
| `learned_prob_top1` | 11.600926 | 6.968520 | 0.780423 | 0.323032 |
| `learned_error_top1` | 11.579703 | 6.915842 | 0.781725 | 0.469776 |
| `multiobs_score_top1` | 84.205911 | 35.776010 | 0.530026 | 0.210682 |

`learned_error_top1` は RMSE では `likpf_mean_single` より -0.024707 改善するが、within10 は悪化する。top1 replacement は採用しない。

topK coverage は learned likelihood が hand-crafted score を大きく上回った。`learned_prob_top3` within10 coverage は 0.892811、`learned_error_top3` は 0.892955、`multiobs_score_top3` は 0.832037。

## 再現性

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- OOF likelihood decompressed SHA: `3aa5e72e982417012a18f4172df1a233ef0f609cf91d48fb1250fc74fa9e89f8`
- model manifest SHA: `178e8b3124b817a2b230080fc041aaaee1b06941e5a4223a68cc31bf26e68010`
- OOF probability SHA: `f4fb66ffd42de8c8ab07c0bdfe1d935ca89a1b6df216b68e701495ec671cc7f3`

## 解釈

candidate-level likelihood calibration は支持された。exp101 のような row-wise selector として使うのではなく、PF weight への弱い加算、topK verifier、または exp092 系 ML add-only confidence feature として使うのが次の候補。

この実験単体では `submission.csv` を作らない。direct replacement / learned top1 は採用しない。
