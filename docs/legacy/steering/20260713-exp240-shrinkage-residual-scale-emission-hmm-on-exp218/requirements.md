# 要件

## 依頼

バックログ `shrinkage_residual_scale_emission_hmm_on_exp218` を
`exp240_shrinkage_residual_scale_emission_hmm_on_exp218` として実装する。

exp218 保存済み `lgb_mean` OOF を exact HMM Gaussian emission center に固定し、
最初に同一 center の scalar `sigma=20` 対照を 1 本実行できるようにする。その結果を
記録した後だけ、exp234 と同じ well-cross-fitted residual scale を scalar sigma へ
variance shrinkage した事前固定 alpha を 1 version 1 候補で評価できるようにする。

## 制約

- Route: `ensemble`。
- scalar control が未完了の状態では shrinkage stage を実行しない。
- shrinkage は `sqrt((1-alpha)*20^2 + alpha*sigma_cf^2)` とする。
- alpha は `0.25 / 0.50` のみ事前登録し、1 Kaggle version につき 1 候補だけ実行する。
- exp218 / exp234 / control の LightGBM は再学習しない。
- parent/control 再学習、GPU、raw-test inference、submission、alpha grid 拡張は禁止する。
- residual-scale stage は同じ row / well の true residual を fit に使わない。
- `docs/06_reproducibility.md` に従って入力・生成物 SHA と runtime 契約を記録する。

## 受け入れ基準

- `config.yaml` が `ensemble` route、scalar-first stage、有限 alpha、1 HMM/version を表現する。
- scalar stage は residual-scale fit 0、HMM 1、LightGBM config/booster 0 である。
- shrinkage stage は GroupKFold by well 5 fit、HMM 1、LightGBM booster 0 である。
- scalar stage と shrinkage stage の同時 enable を fail-fast する。
- alpha が事前登録集合外なら fail-fast する。
- variance-shrunk sigma sidecar の gzip/decompressed SHA を保存する。
- overall、distance、hidden-like、by-well、step-delta の比較を生成する。
- inference notebook は no-output contract を強制する。
