# 設計

## アプローチ

公開 source `/tmp/kaggle-notebooks/connortynan-k16-versioned/rogii-k16-spline-kernel-knn-adaptive-kappa.py` の deterministic fallback を、実験 helper と notebook orchestration に分けて移植する。v7 neural committee / v8 GBM meta-layer は external weights がないため Stage 1 では実装対象外とし、K16 geometry + donor field + adaptive kappa + GR correction + U-projection の単独性能を確認する。

## 実験範囲

- 対象実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- Route: `pf_beam`
- 親実験: `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline` の失敗切り分け
- 変更する変数:
  - K=16 segment spline と raw/smoothed donor coefficient field。
  - XY local-linear kNN と donor-distance bins。
  - Adaptive kappa fit。
  - Near-strike gate と ANCC local theta substitute。
  - Typewell GR correction と U-projection。
- 固定する変数:
  - No GPU。
  - No LightGBM / booster。
  - No v7/v8 external weights。
  - No LB weight search。

## 再現性設計

- seed policy: deterministic sorting と SHA256 well-id fold assignment。
- stochastic 処理の有無: なし。Stage 1 では `np.random` を使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。route は `pf_beam` だが、実装は deterministic geometry / field estimator。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。
- CPU/GPU runtime と deterministic flags: CPU-only、`enable_gpu=false`、`enable_internet=false`。
- train cache / test feature regeneration の SHA 記録方針:
  - train OOF gzip は decompressed CSV content SHA を summary に記録する。
  - inference submission は CSV content SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針:
  - model artifact はないため model SHA は不要。
  - kappa term CSV、OOF prediction、submission SHA を Kaggle 実行後に記録する。
- Kaggle package bootstrap 確認方針:
  - Jupytext source から ipynb を生成。
  - `prepare-kaggle-notebooks --strict` で kernel metadata / config packaging を確認。

## リスク

- リークリスク:
  - 公開 script の full-train fit をそのまま train OOF に使うと target well の TVT / ANCC が漏れる。exp226 train CV では validation target wells を donor field と kappa fit から除外する。
- CV/LB 不一致リスク:
  - group-safe CV は public full-train inference より厳しい。LB が CV より良く見える可能性がある。
  - Typewell GR correction が public/private の well composition に依存する可能性がある。
- ランタイム/メモリリスク:
  - 5 folds で kappa refit を 5 回行うため exp206 より CPU が重い。
  - LOO full refit は重すぎるため、初回は 5-fold group-safe を採用する。
- 再現性リスク:
  - gzip artifact は mtime=0 で保存し、decompressed content SHA を主証拠にする。
  - External weights を見つけても Stage 1 では使わない。
