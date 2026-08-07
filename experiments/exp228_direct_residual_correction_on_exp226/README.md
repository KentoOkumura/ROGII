# exp228_direct_residual_correction_on_exp226

## 状態

- 実装済み / Kaggle train 未実行。
- Route: `ensemble`
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- ML feature surface: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 実行方式: CPU-only split train (`train_lgb0`, `train_lgb1`, `train_lgb2`)

## 仮説

exp226 K16 fallback は CV 9.427 / Public LB 9.837 で単体 anchor としては弱いが、誤差には exp218 feature surface で説明できる系統残差が残っている可能性がある。exp218 と同じ特徴量で `TVT - exp226_oof_pred` を学習し、推論で `exp226_pred + residual_pred` とすれば、exp226 の直接予測を補正できるかを検証する。

## 検証方針

- train target は group-safe exp226 OOF prediction から `TVT - exp226_oof_pred` として作る。
- full-train exp226 prediction から train residual を作らない。
- 特徴量は exp218 と同じ surface を使う。
- LightGBM は `lgb0`, `lgb1`, `lgb2` を3つの CPU notebook に分割し、各 5 folds / 5 boosters とする。
- 3 split 完了後に `train_aggregate` で OOF 平均を作る。
- CV、distance bucket、worst-well、hidden-like、residual drift を確認するまで submit しない。

## 所見

未実行。direct residual correction は CV だけ改善して LB を壊すリスクが高いため、提出判断は split OOF aggregate と stress readout 後に行う。
