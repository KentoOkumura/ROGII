# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `public_raw_gr_residual_scale_control` を `exp214_public_raw_gr_residual_scale_control` として実装する。目的は GRCAL-PFBEAM 系の改善実験に対する public-like raw GR residual-scale control を固定すること。

## 制約

- Route: `pf_beam`
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- LightGBM 学習、fold CV、inference、submit は行わない。
- `exp211/213` と同じ exp072-compatible pseudo-tail 評価面を使う。
- public notebook title LB や visible-test micro-tune は deterministic evidence として扱わない。
- stochastic PF は `docs/06_reproducibility.md` に従い、per-well stable seed を使う。
- gzip 生成物を比較する場合は decompressed content SHA を主証拠にする。

## 受け入れ基準

- `config.yaml` に `experiment.route: pf_beam` と public-like PF 設定が明記されている。
- train notebook で入力、target wells、PF runtime、scale、生成物が確認できる。
- PF は raw horizontal GR、raw typewell GR、known-prefix residual scale `gs = clip(std(GR - typewell_GR(TVT_input)), 10, 60)` を使う。
- PF は `TVT + Z` surface-state convention で生成する。
- `pf_raw_scale_3`、`pf_raw_scale_5`、`pf_raw_scale_8`、`pf_raw_scale_12` を row candidates に保存する。
- 実行予定 variant/config/fold/booster 数が `SESSION_NOTES.md` に記録されている。
- `jupytext --test`、`py_compile`、`ruff --select F821`、`make validate-exp` が通る。
