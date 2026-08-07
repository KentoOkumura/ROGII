# 設計

## アプローチ

exp224 の実装をコピーし、feature builder は維持する。差分は feature selection に限定する。

`build_well_scaled_z_dz_features()` は `z` / `dz` / `dzdmd` / `slp_z` から well 内 robust scale 特徴を生成する。生成後、variant の `drop_base_columns` で raw 4 列を `base_feature_columns` から除外し、`well_scaled_z_dz` feature group を追加する。

## 実験範囲

- 対象実験: `exp227_z_scale_replacement_on_exp218`
- Route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: raw `z` / `dz` / `dzdmd` / `slp_z` を model feature list から削除し、well-scaled z 系特徴へ置換する。
- 固定する変数: exp218 の U-projection、learned likelihood、GRWR feature surface、GroupKFold by well、LightGBM config family、early stopping、seed、評価 metric。
- 実行しないもの: parent/control 再学習、direct z correction、candidate replacement、blend、postprocess、hard selector、sample-weight 変更。

## 学習計画

- active variant: 1 (`z_scale_replacement`)
- LightGBM config split: `lgb0`, `lgb1`, `lgb2`
- folds: 5
- boosters: 各 split 5、合計 15
- runtime: CPU deterministic threads8
- notebooks: `exp227_z_scale_replacement_on_exp218_train_lgb0.py` / `train_lgb1.py` / `train_lgb2.py`

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM deterministic flags、fixed `num_threads=8`。
- stochastic 処理の有無: feature generation は deterministic。LightGBM は CPU deterministic mode。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072/exp145 cache と current feature regeneration を使う。新規 PF/Beam 候補生成や seed bagging はしない。
- 並列処理と乱数の関係: GRWR / z-scale feature generation は per-well deterministic。LightGBM は fixed threads。
- CPU/GPU runtime: CPU only。Kaggle metadata は GPU false。
- train cache / test feature regeneration の SHA 記録方針: Kaggle output の summary、feature schema、model manifest、prediction SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: train 完了後に split output と aggregate readout から記録する。submit は CV が exp218 と比較して妥当な場合のみ。
- Kaggle package bootstrap 確認方針: prepare 後に generated package の py_compile と config / bootstrap manifest を検証する。

## リスク

- リークリスク: well 内 scaler は same-row raw feature と same-well feature-frame statistics のみを使う。target TVT は使わない。
- CV/LB 不一致リスク: exp218 は Public LB 7.843 の submitted anchor。CV だけで採用せず、100-1000 bucket と worst-well regression を確認する。
- ランタイム/メモリリスク: exp224 と同等以上の feature build が必要。CPU split で config ごとに notebook を分ける。
- 再現性リスク: split train の出力を aggregate する必要がある。inference は aggregate manifest か split manifest handling ができるまで実行しない。
