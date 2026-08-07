# exp005_gr_gate_recalibration 結果

## 仮説

exp004 の low-GR coverage hard gate は CV を改善したが、visible/public 寄り well で no-GR へ倒しすぎた可能性がある。gate 条件を soft blend または prefix/eval 両方の低 coverage 条件に絞れば、Public LB リスクを下げつつ exp002 より良い CV を維持できる可能性がある。

## 設定

- 親: `exp004_gr_gating`
- LB 基準: `exp002_drift_minimal`
- 検証: `well_id` GroupKFold、`TVT_input` NaN 行のみ評価
- メトリック: RMSE
- シード: 42
- モデル: `HistGradientBoostingRegressor`
- target: `TVT - last_anchor_tvt`
- pre-run selected candidate: `gate_low_gr_strict_hard`

## 結果

Kaggle full CV:

| Variant | CV | Mean Fold RMSE | exp002 差分 | Gate Weight | Gated Wells | Eval Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_exp003_no_gr` | 13.882944 | 13.859376 | -0.241625 | - | - | - |
| `control_exp004_low_gr_any_hard` | 13.932968 | 13.908682 | -0.191601 | 1.0 | 297 | 1,506,134 |
| `gate_low_gr_strict_hard` | 13.936732 | 13.913383 | -0.187837 | 1.0 | 214 | 1,115,013 |
| `gate_low_gr_any_soft_050` | 13.998291 | 13.974830 | -0.126278 | 0.5 | 297 | 1,506,134 |
| `gate_low_gr_strict_soft_050` | 14.007188 | 13.984324 | -0.117381 | 0.5 | 214 | 1,115,013 |
| `control_exp002_all` | 14.124569 | 14.101909 | 0.000000 | - | - | - |

## 解釈

`control_exp003_no_gr` が全体の CV 最良だが、これは exp003 の再現であり Public LB 12.852 が exp002 の 12.533 より悪かった。gating 系では exp004 selected gate の再現 `control_exp004_low_gr_any_hard` が 13.932968 で最良だった。

selected の `gate_low_gr_strict_hard` は 13.936732 で、exp004 selected gate より 0.003764 悪い。ただし狙い通り、visible `000d7d20` の gate weight は exp004 条件の 1.0 から 0.0 へ戻った。CV をほぼ維持しつつ public-visible 寄り well を守る候補としては成立している。

soft blend は any / strict とも hard gate より悪く、今回の結果だけなら採用しない。

## 次

## Inference / Submit Check

Kaggle inference kernel `kentookumura/exp005-gr-gate-recalibration-inference` version 1 が完了し、提出 CSV は `data/external/kaggle-output/exp005_gr_gate_recalibration/inference/submission.csv` に保存済み。

`submit-check` は PASS。14,151 rows、`id,tvt`、sample submission と header / row count が一致し、重複 ID、NaN、Inf-like values はなかった。

Visible duplicate wells の local truth sanity:

| Well | exp004 Gate | exp005 Gate | exp002 RMSE | exp004 RMSE | exp005 RMSE | exp005 - exp004 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `000d7d20` | 1.0 | 0.0 | 7.789073 | 7.908222 | 7.789073 | -0.119149 |
| `00bbac68` | 0.0 | 0.0 | 9.393623 | 9.393623 | 9.393623 | 0.000000 |
| `00e12e8b` | 0.0 | 0.0 | 5.356808 | 5.356808 | 5.356808 | 0.000000 |

Aggregate visible RMSE: exp002 7.916353、exp004 7.948310、exp005 7.916353。visible 3 wells では exp005 は exp002 と同じで、exp004 の `000d7d20` 悪化を戻した。

## Submission

Kaggle submission ref `53249562` は `SubmissionStatus.COMPLETE`。Public LB は 12.579。

exp005 は exp004 の Public LB 12.730 から改善し、exp003 12.852 も上回った。一方、exp002 の Public LB 12.533 には届かない。CV は exp005 13.936732 が exp002 14.124569 より良いが、Public LB 基準は引き続き exp002。

## 次

exp005 の strict gate は public-visible guard としては有効だったが、LB 最良ではない。次は exp002 / exp005 の OOF hard well 解析へ進み、GR gate ではなく well-level failure pattern に基づく改善候補を作る。
