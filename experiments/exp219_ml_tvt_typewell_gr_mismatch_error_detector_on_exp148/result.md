# exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148 結果

## 状態

Kaggle CPU train v1 完了。no-training OOF readout としては完了/不採用。

## 結果

- Kernel: `kentookumura/exp219-ml-tvt-gr-mismatch-exp148-train` v1
- Rows / wells / feature columns: 3,783,989 / 773 / 35
- exp148 base `lgb_mean`: RMSE 8.501281182、MAE 5.335654736、within10 0.856332035
- Primary signal: `mlgr_mismatch_signal`
- `abs_error_gt10` AUC: 0.573943003
- high-mismatch q90 bucket: error_gt_rate 0.234520、error_gt_lift 1.632373、abs_error_lift 1.425989、RMSE 12.139380847
- diagnostic correction best: base exp148 のまま。`best_offset` 補正は RMSE を更新しない。

## 判断

high-mismatch bucket は誤差が濃い領域をある程度拾うが、primary AUC は採用目安 0.65 に届かない。`best_offset` を使う補正診断も base exp148 を上回らない。

したがって exp148/exp193 add-only LightGBM、inference、submit には進めない。残す場合は standalone detector ではなく、将来の confidence ensemble 用の weak risk flag / bucket readout に限定する。
