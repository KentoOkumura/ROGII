# exp234_crossfitted_residual_scale_emission_hmm_on_exp218

## 状態

- Route: アンサンブル
- Status: train-side 完了・不採用（inference / submit なし）
- 親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- CV / Public LB: RMSE 8.427231 / 未提出
- 利用可否: 不可。exp218 OOF より改善したが、exp221 fixed-sigma HMM に届かない。

## 仮説

exp229 が失敗した理由は、quantile q50 を HMM center に替えたことで既存の強い point prediction を失ったことにある。exp218 の保存済み `lgb_mean` OOF を center に固定し、同一 well の残差を学習に含めない cross-fitted residual scale のみを row-wise sigma にすれば、HMM emission の信頼度を変えつつ point center の強さを保てる可能性がある。

## 親実験との差分

- 親の exp218 は GR wavelet / rotation confidence を add-only した 15 booster の ML anchor であり、本実験ではその booster を一切再学習しない。
- exp221 の fixed `sigma=20` HMM と違い、sigma だけを exp218 OOF residual から well-cross-fit する。
- exp229 のように quantile q50 へ center を置換しない。exp218 `lgb_mean` を immutable center に保ち、lambda / floor / cap を single fixed value にして検索しない。

## 検証方針

- exp218 `lgb_mean` OOF と exp072 row context の ID / well coverage を strict に一致させる。
- residual scale は well 単位 5-fold GroupKFold の held-out prediction に限定し、同じ row / well の真値 residual を sigma の fit に使わない。
- HMM より先に scale decile の error lift、Spearman、floor / cap rate、fold separation を保存し、固定 guard を評価する。
- guard を通過した場合だけ、`lambda=0.50`、floor `2.5`、cap `40.0` の single HMM variant を生成する。
- exp218 / exp148 / exp193、distance bucket、hidden-like、by-well、step-delta を比較する。inference / submit は行わない。

## 所見

v1 guard は通過した（Spearman `0.326486`、top/bottom RMSE ratio `3.578534`、floor `0.180690`、cap `0.0`、fold overlap `0`）。v2 readout は v1 HMM cache を再利用し、RMSE `8.427231`（exp218 比 `-0.048573`）となった。hidden-like spatial / typewell-purged もそれぞれ `-0.083018` / `-0.088034` 改善したが、exp221 fixed-sigma HMM の `8.327737` より悪いため不採用とする。

## 生成物

Kaggle train audit が実行された場合、`artifacts/` に次を保存する。

- cross-fitted scale: `*_residual_scale_predictions.csv.gz`、`*_residual_scale_calibration.csv`、`*_residual_scale_folds.csv`
- HMM: `*_crossfitted_residual_scale_emission_hmm_train_features.csv.gz`
- readout: `*_overall_metrics.csv`、distance / hidden-like / by-well / step-delta summary

gzip の比較証拠は raw file SHA ではなく decompressed content SHA を主に記録する。
