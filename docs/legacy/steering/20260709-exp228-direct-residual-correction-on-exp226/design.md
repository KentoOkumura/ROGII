# 設計

## アプローチ

exp218 の feature generation / LightGBM training flow を親実装として再利用し、target/base prediction だけを exp226 residual correction 用に差し替える。train では exp226 train v1 OOF artifact を `well_id + row_idx` で exp218 feature surface に join し、`target_tvt - exp226_oof_pred` を学習する。inference では exp226 inference v1 の `submission.csv` を base prediction として読み、3 split の saved LightGBM boosters から平均 residual を加算する。

## 実験範囲

- 対象実験: `exp228_direct_residual_correction_on_exp226`
- Route: `ensemble`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 変更する変数:
  - target: `TVT - exp226_oof_pred`
  - base prediction: exp226 OOF / exp226 inference submission
  - runtime: CPU split train
- 固定する変数:
  - exp218 feature surface
  - LightGBM config family `lgb0/lgb1/lgb2`
  - 5-fold GroupKFold by well
  - control / parent retraining なし

## 再現性設計

- seed policy: exp218/exp063 LightGBM config seeds と fixed GroupKFold seed 42 を継承する。
- stochastic 処理の有無: LightGBM training は seed 固定。GRWR feature generation は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam は実行しない。exp072 / exp145 / exp226 の保存済み Kaggle outputs を入力として使う。
- 並列処理と乱数の関係: CPU LightGBM は `deterministic=true`, `force_col_wise=true`, `n_jobs=8`, `num_threads=8`。
- CPU/GPU runtime と deterministic flags: Kaggle GPU は無効。split train は `train_lgb0/lgb1/lgb2` の3 kernel。
- train cache / test feature regeneration の SHA 記録方針: exp226 OOF は decompressed SHA、feature schema / prediction / model manifest は生成物 summary に記録する。
- model manifest / prediction / submission SHA 記録方針: 各 split train は model manifest と prediction SHA、inference は submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` と py_compile / ruff / validate-exp で確認する。

## リスク

- リークリスク: full-train exp226 prediction を train residual に使うと leakage。group-safe OOF だけを使う。
- CV/LB 不一致リスク: direct residual correction は CV 改善だけで LB を壊しやすい。submit 前に split aggregate、bucket、worst-well、hidden-like を見る。
- ランタイム/メモリリスク: exp218 feature surface 生成と LightGBM 15 boosters は重い。CPU split で各 kernel 5 boosters に分割する。
- 再現性リスク: 3 split kernel の outputs を inference/aggregate で正しく揃える必要がある。manifest と feature schema を確認する。
