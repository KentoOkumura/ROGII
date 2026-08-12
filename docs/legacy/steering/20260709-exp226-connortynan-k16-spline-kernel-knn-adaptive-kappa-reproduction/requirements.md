# 要件

## 依頼

バックログ `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` を実装する。Connor Tynan の公開 notebook `ROGII K16 spline + kernel kNN + adaptive kappa` を source-port し、exp206 の線形 `dTVT ~= a*dZ+b` 再現失敗と切り分ける。

## 制約

- Route: `pf_beam`
- Stage 1 は外部 `e2e2/stacker/gbm` weights を使わない。
- K=16 segment spline、raw/smoothed fused ridge coefficients、XY local-linear kNN、adaptive kappa、near-strike gate、ANCC surface local theta、U-projection を実装する。
- 公開 script の v8 score claim は optional external weights 前提の可能性があるため、Stage 1 の成功証拠にしない。
- exp206 の `dTVT ~= a*dZ+b` 線形回帰は流用しない。
- Train OOF / CV では target well 自身の TVT、ANCC、donor segment を donor field と kappa fit から除外する。
- Blind LB weight search はしない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/` に config、README、SESSION_NOTES、result、metrics、train/inference notebook source がある。
- `config.yaml` の `experiment.route` は `pf_beam`。
- train notebook は group-safe CV を実行でき、target well 除外ポリシーを表示する。
- inference notebook は full train field/kappa から `submission.csv` を生成し、sample order / finite value guard を持つ。
- v7/v8 external weights は無効化され、Stage 2 audit として deferred になっている。
- Kaggle push 前コストとして active variants、LightGBM configs、folds、boosters、control retraining の有無が `SESSION_NOTES.md` に記録されている。
- Jupytext 変換、構文チェック、F821、experiment validation が通る。
- deterministic anchor として扱う場合は、feature content SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
