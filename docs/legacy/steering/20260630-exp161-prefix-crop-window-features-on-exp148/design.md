# 設計

## アプローチ

exp148 の full-train LightGBM flow をコピーし、`prefix_crop_window` feature group を追加する。window は `tail1000`、`tail2000`、`last50`。既存の exp072 / exp092 / exp145 feature surface はそのまま残し、crop-window 版を add-only で渡す。

## 実験範囲

- 対象実験: `exp161_prefix_crop_window_features_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 変更する変数: known prefix crop-window features の追加
- 固定する変数: exp148 base features、U-projection settings、learned likelihood features、LightGBM config family、GroupKFold by well、residual target
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960

## Feature 設計

- Prefix stats: crop known rows、MD span、`slp_md`、`slp_z`、`ktvt_range`、`ktvt_std`、`pfx_rmse`、`cal_a`、`cal_b`
- SC/NCC: crop prefix GR library から `sc8` / `sc15` / `sc25` / `sc_cons` / `sc_ens` を再計算
- Multiobs: candidate `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` について crop-window score / MAE / NCC / full-minus-crop / outside-range flag を追加

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM deterministic flags、fixed `n_jobs` / `num_threads`
- stochastic 処理の有無: 新規 crop feature generation には RNG なし
- PF/Beam / likelihood-PF / seed bagging の有無: upstream exp072 / exp145 生成物を使う。inference の raw-test replay は既存 exp148 flow に従う
- 並列処理と乱数の関係: crop feature derivation は well ごとの deterministic arithmetic
- CPU/GPU runtime と deterministic flags: active mode は `cpu_deterministic_threads8`、Kaggle `enable_gpu=false`
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を主証拠にする
- model manifest / prediction / submission SHA 記録方針: train manifest に feature group、model SHA、OOF prediction SHA を残す
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata の `enable_gpu=false` を確認する

## リスク

- リークリスク: crop window は raw known prefix と target-free candidate values だけを使う。validation/test true TVT は使わない。
- CV/LB 不一致リスク: prefix features は Public LB に転移しない可能性があるため、global OOF だけで submit しない。
- ランタイム/メモリリスク: exp148 294 features に crop features を追加するため CPU train は遅くなる。
- 再現性リスク: upstream PF/Beam cache と raw-test replay は deterministic anchor としては別途 SHA 記録が必要。
