# exp112_learned_pf_likelihood_weight_or_feature_followup 結果

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

exp111 learned likelihood は direct top1 replacement では採用しない。一方で、PF weight への弱い加算、`likpf_mean` fallback verifier、または exp092 系 ML add-only feature としてなら有効な confidence signal になる可能性がある。

## 設定

- 親: `exp111_learned_pf_observation_likelihood_probe`
- 入力:
  - exp111 OOF likelihood long cache
  - exp099 v2 wide train feature cache
- 候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`
- 評価: exp111 first-fold OOF rows の train-side posthoc audit
- 新規学習: なし

## Kaggle train v1

- Kernel: `kentookumura/exp112-learned-pf-likelihood-followup-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp112-learned-pf-likelihood-followup-train`
- status: `completed_train_side_posthoc_audit`
- runtime: 310.78 sec
- rows: 757,738
- candidate rows: 3,788,690
- wells: 155
- output: `kaggle/output/train_v1`
- GPU: false
- internet: false

## 結果

| variant | RMSE | MAE | within10 | switch vs likPF |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean_single` | 11.604410 | 6.944251 | 0.784312 | 0.000000 |
| `learned_error_top1` | 11.579703 | 6.915842 | 0.781725 | 0.596719 |
| `learned_prob_top1` | 11.600926 | 6.968520 | 0.780423 | 0.477624 |
| `gate_expected_error_m2p0_d20p0` | 11.573266 | 6.926626 | 0.785064 | 0.004077 |
| `oracle_candidate` | 7.857730 | 3.852781 | 0.908454 | 0.587608 |

best non-oracle は `gate_expected_error_m2p0_d20p0`。`likpf_mean_single` から RMSE `-0.031144`、within10 `+0.000752`、MAE `-0.017625` 改善した。switch rate は 0.4077% と低く、selection は `likpf_mean` 99.592%、`beam_mean` 0.224%、`pf_ancc` 0.169%、`hyb` 0.009%、`sc_ens` 0.006%。

PF weight alpha は全て崩壊した。best でも `pf_weight_expected_error_alpha_0p4` が RMSE 69.484358 / within10 0.584181 で、multiobs score に learned signal を弱く足す設計は採用しない。

## 生成物

- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_summary.json`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_oof_predictions.csv.gz`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_ml_features.csv.gz`
- `kaggle/output/train_v1/artifacts/exp112_learned_pf_likelihood_weight_or_feature_followup_feature_schema.csv`

## 再現性

- exp111 OOF likelihood long decompressed SHA: `3aa5e72e982417012a18f4172df1a233ef0f609cf91d48fb1250fc74fa9e89f8`
- exp099 wide cache decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- metrics SHA: `ac4068b86f4a1f43a5fb3e27c58692419ffd520d1b5570cd38256e8b993757ef`
- OOF predictions decompressed SHA: `e3df222a2bb11432f2474f51d672d6d961eb598c9a17e4905c41017e66bb15d7`
- ML features decompressed SHA: `56c0f62238abfc89f05e5700341344c15815bd3a5f93e5b0a6a079a661b9411e`
- prediction SHA: `642426a934e967310062411cdda05449291dd181a80c33965992d9caa90adfd9`

## 解釈

learned likelihood は hard selector や PF weight alpha としては危険。特に PF weight は multiobs top1 の崩壊をほぼ引き継ぎ、learned signal で救えなかった。

一方で expected-error margin を使った very-low-switch verifier は、`likpf_mean` をほぼ維持したまま小幅に改善した。train-side では支持するが、改善幅は小さく、raw-test score surface parity と worst-well guard なしでは提出候補にしない。

ML feature cache は target-free columns のみで保存済み。次は exp092 系 ML add-only feature、または exp102/exp112 の low-switch gate をまとめた raw-test parity 診断に回す。
