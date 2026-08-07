# exp004_gr_gating 結果

## 仮説

exp003 の no-GR は hard wells の過補正を抑えて CV を改善した一方、GR が効く public/visible 寄り wells では悪化した可能性がある。exp002 all-GR を default にし、inference-safe な well 条件で no-GR へ gating / shrink すれば、CV 改善と Public LB リスク低減を両立できる可能性がある。

## 設定

- 親: `exp003_residual_ablation`
- control baseline: `exp002_drift_minimal`
- 検証: `well_id` GroupKFold、`TVT_input` NaN 行のみ評価
- メトリック: RMSE
- シード: 42
- モデル: `HistGradientBoostingRegressor`
- target: `TVT - last_anchor_tvt`

## 結果

Kaggle full CV:

| Variant | CV | Mean Fold RMSE | exp002 差分 |
| --- | ---: | ---: | ---: |
| `control_exp003_no_gr` | 13.882944 | 13.859376 | -0.241625 |
| `gate_low_gr_coverage_hard` | 13.932968 | 13.908682 | -0.191601 |
| `control_exp002_all` | 14.124569 | 14.101909 | 0.000000 |
| `gate_high_gr_shift_soft` | 14.130448 | 14.107356 | +0.005879 |
| `gate_high_gr_shift_hard` | 14.145053 | 14.121413 | +0.020484 |

Selected inference candidate: `gate_low_gr_coverage_hard`

## 解釈

`control_exp003_no_gr` が全体の CV 最良だが、これは exp003 の再現であり Public LB 12.852 が exp002 の 12.533 より悪かった。今回の目的である gating としては、GR coverage が弱い well だけ no-GR に切り替える `gate_low_gr_coverage_hard` が最良で、exp002 all-GR から CV を 0.191601 改善した。

一方、GR shift / variance で gating する 2 variants は exp002 より悪化した。現状の `prefix_gr_std >= 35` / `gr_delta_abs_mean >= 25` は hard well 判定として弱く、GR を弱める条件としては使わない。

## 次

## Inference / Submit Check

Kaggle inference kernel `kentookumura/exp004-gr-gating-inference` version 1 が完了し、提出 CSV は `data/external/kaggle-output/exp004_gr_gating/inference/submission.csv` に保存済み。

`submit-check` は PASS。14,151 rows、`id,tvt`、sample submission と header / row count が一致し、重複 ID、NaN、Inf はなかった。

Visible duplicate wells の local truth sanity:

| Well | Gate Weight | exp002 RMSE | exp003 RMSE | exp004 RMSE | exp004 - exp002 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000d7d20` | 1.0 | 7.789073 | 7.908222 | 7.908222 | +0.119149 |
| `00bbac68` | 0.0 | 9.393623 | 10.633381 | 9.393623 | 0.000000 |
| `00e12e8b` | 0.0 | 5.356808 | 4.722850 | 5.356808 | 0.000000 |

Aggregate visible RMSE: exp002 7.916353、exp003 8.472623、exp004 7.948310。exp004 は exp003 より戻せているが、visible aggregate では exp002 より 0.031957 悪い。

## Submission

Kaggle submission ref `53247991` は `SubmissionStatus.COMPLETE`。Public LB は 12.730。

exp003 の Public LB 12.852 よりは改善したが、exp002 の 12.533 には届かなかった。CV では exp004 13.932968 が exp002 14.124569 より良い一方、Public LB では exp002 がまだ最良。

## 次

Public LB を 基準 にするなら exp002 を維持する。次に進めるなら、GR coverage gate の閾値を visible/public 寄り wells で過剰に no-GR へ倒さない方向で見直す。
