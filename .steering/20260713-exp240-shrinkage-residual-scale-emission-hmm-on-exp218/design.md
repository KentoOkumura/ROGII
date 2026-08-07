# 設計

## アプローチ

exp221 の RMSE `8.327737` は fixed `sigma=20` を使うが center は exp148 であり、
exp234 の exp218-center row-wise sigma RMSE `8.427231` と直接比較できない。そこで最初に
exp218 center + scalar `sigma=20` を厳密な対照として作る。

対照結果を記録した後だけ、exp234 と同じ well GroupKFold residual-scale model を再生成し、
`sigma_eff = sqrt((1-alpha)*20^2 + alpha*sigma_cf^2)` で分散を固定 sigma へ縮小する。
候補は alpha `0.25 / 0.50` に限定し、Kaggle timeout を避けるため 1 version につき
1 HMM variant だけを実行する。

## 実験範囲

- 対象実験: `exp240_shrinkage_residual_scale_emission_hmm_on_exp218`
- Route: `ensemble`
- 親実験: `exp234_crossfitted_residual_scale_emission_hmm_on_exp218`
- 変更する変数: exp218 Gaussian emission の sigma のみ。
- 固定する変数: exp218 OOF center、HMM dynamics、lambda `0.50`、scalar sigma `20`、floor `2.5`、cap `40`。
- 初期 active stage: scalar control 1 本。
- deferred stages: alpha `0.25`、`0.50`。同時実行しない。

## 再現性設計

- seed policy: scalar HMM は RNG なし。deferred scale は GroupKFold shuffle なし、HGB `random_state=42`。
- stochastic 処理: deferred HGB internal binning のみ seed 固定。
- PF/Beam / likelihood-PF / seed bagging: 新規生成なし。
- 並列処理: HMM outer worker 1、Numba thread 1、scale fit 逐次。
- CPU/GPU: CPU only。保存済み exp218 OOF を読み、booster 0。
- SHA: exp218 input、row context、scale predictions、shrinkage sidecar、HMM cache は gzip/decompressed を分離記録する。
- model/submission SHA: 新規 model / submission がないため対象外。
- bootstrap: prepare 後に config の selected stage、CPU、kernel sources、1 variant 契約を確認する。

## リスク

- leakage: residual-scale fit に held-out well residual が混入する。GroupKFold well overlap 0 を必須にする。
- 比較不成立: scalar control より先に shrinkage を走らせる。status と stage guard で拒否する。
- timeout: 複数 alpha の同時 HMM。1 version 1 variant を強制する。
- CV/LB: train-side 改善が raw test に移らない。inference / submit はこの実験実装から除外する。
- 再現性: gzip metadata 差。decompressed content SHA を主証拠にする。
