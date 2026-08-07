# 要件

## 依頼

`exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe` の可視化に、exp226 `connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` の train OOF 予測も重ねて表示する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新規学習、PF/Beam 再生成、推論、提出は行わない。
- exp226 train output の既存 OOF gzip を入力として読むだけにする。
- exp226 OOF の `well_id,row_idx,tvt_pred` を exp083 plot frame の `id,well` に合わせて結合する。

## 受け入れ基準

- TVT パネルに exp226 OOF 予測線が追加される。
- per-well title、manifest、summary に exp226 OOF RMSE / coverage が記録される。
- summary に exp226 OOF source path と decompressed SHA が記録される。
- notebook `.py` から `.ipynb` を再生成し、構文チェックと Jupytext round-trip 検証が通る。
- deterministic anchor として新規扱いしない。既存 exp226 の OOF SHA / Kaggle kernel version は exp226 側の記録を参照する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
