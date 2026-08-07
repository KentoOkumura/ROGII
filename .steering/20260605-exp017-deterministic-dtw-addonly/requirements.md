# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭 `deterministic_dtw_addonly` を実装する。

目的は、公開 notebook 系の DTW / DWT alignment route を重い探索として移植する前に、fold-safe な小さい deterministic feature set として再現し、`exp012/exp013` の `lightgbm_no_gr` raw anchor と比較できる状態にすること。

## 制約

- 親実験は `exp013_model_diversity_or_postprocess` 相当の raw LightGBM no-GR anchor を維持する。
- valid fold の `TVT`、`TVT_input` hidden tail、train-only formation columns は alignment feature 生成に使わない。
- DTW/DWT feature は 見えない test well 推論 で利用可能な `MD`、`GR`、known `TVT_input` prefix、paired typewell GR だけから作る。
- `pywt` 依存は入れず、rolling smooth / detail energy で CWT-DWT 風の multi-scale texture を近似する。
- score 記録では raw CV、DTW/DWT feature 追加 CV、postprocess OOF-fit/held-out 値を混同しない。

## 受け入れ基準

- `.steering`、`experiments/exp017_deterministic_dtw_addonly/`、`config.yaml`、train/inference notebook、`SESSION_NOTES.md` が整っている。
- `uv run python scripts/validate_experiment.py --experiment exp017_deterministic_dtw_addonly` が通る。
- `baseline.py` の compile / lint が通る。
- Kaggle train/inference notebook を `--strict` で準備できる。
- Full CV 実行後は `metrics.json`、主要 artifacts、`SESSION_NOTES.md`、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。
