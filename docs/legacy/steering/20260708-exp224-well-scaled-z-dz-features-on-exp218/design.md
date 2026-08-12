# 設計

## アプローチ

exp218 の feature assembly と LightGBM 学習フローを維持し、GRWR feature 生成後の `full_frame` に `well_scaled_z_dz` feature group を add-only で追加する。

追加する特徴は以下に限定する。

- `z` / `dz` / `dzdmd` / `slp_z` の well 内 centered、MAD scaled、IQR scaled、p05-p95 scaled、rank pct、relative p05-p95。
- 各 source column の well-level MAD / IQR / p05-p95 range log1p。
- `likpf_tvt = last_known_tvt + likpf_mean_d` の well 内 p05-p95 range を使った `z` / `dz` scale。
- scaled z/dz と raw `gr` の well-scaled interaction。
- scaled z/dz と `grwr_fft_rotation_ratio_x_log1p_md_since`、`grwr_candidate_tvt_std`、`grwr_candidate_tvt_range`、`grwr_known_prefix_fraction` の interaction。

## 実験範囲

- 対象実験: `exp224_well_scaled_z_dz_features_on_exp218`
- Route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: `well_scaled_z_dz` feature group の追加
- 固定する変数: exp072 base cache、exp092 U projection、exp145 learned likelihood、exp218 GRWR features、LightGBM config family、GroupKFold split、seed、CPU deterministic flags
- control 再学習: なし
- train 実行: CPU-only。`train_lgb0` / `train_lgb1` / `train_lgb2` に分割し、各 notebook は 1 LightGBM config x 5 folds だけを学習する。

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM deterministic flags、fixed threads。
- stochastic 処理の有無: 新規 feature generation には乱数なし。CPU LightGBM は deterministic flags と fixed threads で実行する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成はなし。exp072 / exp145 の保存済み cache を参照する。
- 並列処理と乱数の関係: feature generation は per-well deterministic。LightGBM は `deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- CPU/GPU runtime と deterministic flags: GPU は使わず、`cpu_deterministic_threads8` を active mode にする。
- train cache / test feature regeneration の SHA 記録方針: Kaggle output の feature schema、prediction、model manifest SHA を記録する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: train 後に manifest / prediction SHA、inference 後に prediction / submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` と package py_compile で確認する。

## リスク

- リークリスク: well 内 scaling が target を使わないこと、OOF error や true TVT を scaler にしないことを確認する。
- CV/LB 不一致リスク: exp218 で残った 100-1000 bucket / worst-well regression が悪化する可能性がある。
- ランタイム/メモリリスク: 追加 feature は約50列。exp218 の 380 features から増えるため、CPU split でも train runtime と memory を確認する。
- 再現性リスク: CPU LightGBM と Kaggle runtime の差分。deterministic anchor 化する場合は SHA と kernel version を揃える。
