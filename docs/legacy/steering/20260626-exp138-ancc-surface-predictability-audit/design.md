# 設計

## アプローチ

LightGBM を使わず、従来型の surface imputer を fold-safe に比較する。

1. `global_median`: train-fold `ANCC` の中央値を全 row に入れる負の基準。
2. `row_knn_xy`: train-fold rows を deterministic に間引き、`X,Y -> ANCC` を距離重み KNN で推定する。
3. `well_plane_knn`: train-fold well ごとの `median(X), median(Y), median(ANCC)` から、query 周辺 well の距離重み局所平面 `ANCC = aX + bY + c` を解く。

各 fold で valid wells の score rows と prefix anchor row に `ANCC_hat` を出し、絶対誤差と
`ANCC_hat(row) - ANCC_hat(anchor)` の delta 誤差を別々に評価する。

## 実験範囲

- 対象実験: `exp138_ancc_surface_predictability_audit`
- Route: `ml_model`
- 親実験: `KAGGLE_DIRECTION.md` の `ancc_surface_predictability_audit` backlog
- 変更する変数: `ANCC` surface imputer method
- 固定する変数: well-level CV、score row mask、LightGBM なし、PF/Beam なし、target ablation なし

## 再現性設計

- seed policy: fixed global seed 42 + fold offset。`row_knn_xy` の row subsampling だけに使う。
- stochastic 処理の有無: deterministic row subsampling のみ。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: KNN prediction は sampling 後 deterministic。global RNG は使わない。
- CPU/GPU runtime と deterministic flags: CPU only。GPU/AMP なし。
- train cache / test feature regeneration の SHA 記録方針: OOF prediction CSV と summary CSV の SHA256 を `metrics.json` に記録する。
- model manifest / prediction / submission SHA 記録方針: persistent model なし、submission なし。prediction CSV SHA を主証拠にする。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` で metadata と bootstrap を検証する。

## リスク

- リークリスク: validation fold の真 `ANCC` が fitting や anchor 推定に混ざると過大評価になる。実装では fold ごとに train wells だけで surface を fit する。
- CV/LB 不一致リスク: この実験は提出しないため LB は発生しない。将来 target ablation に進む場合は別途 TVT OOF と hidden-like stress が必要。
- ランタイム/メモリリスク: row-level KNN が重い。`max_rows_per_well` と `max_rows_total` で制限する。
- 再現性リスク: row subsampling の seed と output SHA を記録する。
