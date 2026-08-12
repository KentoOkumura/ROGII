# 設計

## アプローチ

`exp218` の saved `lgb_mean` OOF をそのまま HMM observation center にする。残差 scale は、exp072 full replay cache の target-free row context と exp218 center から作る少数の deterministic covariate を入力に、well 単位 5-fold GroupKFold の held-out prediction として作る。

各 inner fold では、他 well の squared residual（clip 後の `log1p` 変換）だけを target にした固定の `HistGradientBoostingRegressor` を fit する。held-out fold の予測を逆変換して Gaussian sigma にし、固定 floor / cap を適用する。これにより、各 row の真値 residual はその row の sigma に直接も間接にも使われない。

scale artifact の error-lift、calibration、floor / cap rate を first-class artifact として保存し、設定した guard を通過したときだけ、`lambda=0.50` の HMM single variant を実行する。point center の再学習、HMM parameter search、raw-test scale regeneration はこの実験の範囲外とする。

## 実験範囲

- 対象実験: `exp234_crossfitted_residual_scale_emission_hmm_on_exp218`
- Route: `ensemble`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`。HMM implementation / comparator は exp221, exp229 を参照する。
- 変更する変数: exp218 center に対する fold-safe residual scale のみ。
- 固定する変数: exp218 `lgb_mean` point OOF、HMM dynamics、lambda `0.50`、sigma floor / cap、one HMM variant、well-grouped validation。

## 再現性設計

- seed policy: GroupKFold は `shuffle=False` で deterministic。residual scale estimator は `random_state=42`、CPU single worker に固定する。
- stochastic 処理の有無: residual scale estimator の internal randomness を固定する。exact HMM は RNG を使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 cache は comparator の input のみ。新規 PF/Beam / seed bagging は生成しない。
- 並列処理と乱数の関係: HMM は outer worker 1 / numba thread 1 を既定とし、scale cross-fit は逐次 fit する。
- CPU/GPU runtime と deterministic flags: CPU-only。exp218 LightGBM は保存済み OOF を読むだけで再学習しない。
- train cache / test feature regeneration の SHA 記録方針: exp218 source OOF、residual-scale OOF、HMM train feature の gzip / decompressed SHA を記録する。test regeneration は実装・実行しない。
- model manifest / prediction / submission SHA 記録方針: residual-scale model は joblib 保存せず、config / folds / prediction SHA を残す。inference / submission SHA は該当なしとして記録する。
- Kaggle package bootstrap 確認方針: train audit を push する前に bootstrap 内 config、kernel metadata、CPU runtime を確認する。

## リスク

- リークリスク: residual scale が同 row / same-well residual を使うこと。GroupKFold by well と ID coverage / fold membership audit で防ぐ。
- CV/LB 不一致リスク: train OOF HMM の改善が hidden test に転移しない可能性。inference / submit は by-well、hidden-like、floor rate の guard 後に別途判断する。
- ランタイム/メモリリスク: exact HMM は 773 well で長時間になり得る。single variant、outer worker 1、Numba 1 thread を固定し、scale audit と HMM audit を同一 CPU notebook で順に出力する。
- 再現性リスク: HMM / HGB の version 差。kernel version、入力 / output SHA、config、fold assignment を保存して記録する。
