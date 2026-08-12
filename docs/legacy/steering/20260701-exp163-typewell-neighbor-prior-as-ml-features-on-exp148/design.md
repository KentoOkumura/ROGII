# 設計

## アプローチ

exp148 の full-train learned likelihood add-only runner をベースにし、exp109/120 で支持された native typewell-overlap neighbor prior を LightGBM の補助特徴量として追加する。prior TVT をそのまま採用せず、`prior_delta`、`prior_minus_likpf_mean`、coverage/count/std、longtail interaction、PF/Beam disagreement interaction、clipped correction proxy に変換する。

## 実験範囲

- 対象実験: `exp163_typewell_neighbor_prior_as_ml_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: typewell neighbor prior add-only feature group
- 固定する変数: exp072 cache、exp092 U-projection surface、exp145 learned likelihood confidence features、target、GroupKFold seed、LightGBM config family
- 実行 mode: CPU deterministic threads8
- Train notebook: `train_lgb0` / `train_lgb1` / `train_lgb2` に分割

## 再現性設計

- seed policy: GroupKFold seed 42、typewell prior fold split seed 42、LightGBM deterministic flags
- stochastic 処理の有無: 新規 feature generation に RNG は使わない。GroupKFold well shuffle だけ固定 seed。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 / exp099 / exp145 / exp065 の保存済み cache を読む。
- 並列処理と乱数の関係: LightGBM CPU deterministic flags と固定 thread count で制御する。
- CPU/GPU runtime と deterministic flags: `runtime.kaggle.enable_gpu=false`、`cpu_deterministic_threads8` のみ active。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にし、schema / summary SHA も manifest に残す。
- model manifest / prediction / submission SHA 記録方針: split train manifest、fold prediction SHA、pooled prediction SHA を保存する。submission SHA は inference / submit 時のみ記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train_lgb* --strict` で metadata と source kernels を確認する。

## リスク

- リークリスク: neighbor prior は train-fold wells だけから validation rows へ補間する。validation true TVT、oracle best、true-error rank は feature source に入れない。
- CV/LB 不一致リスク: raw-test/current-test typewell prior parity は未実装のため、train-side positive だけで submit しない。
- ランタイム/メモリリスク: exp148 full-row CPU train に typewell features を追加するため、lgb config ごとに notebook を分割する。
- 再現性リスク: upstream cache の生成自体は過去実験に依存するため、content SHA と manifest を記録する。
