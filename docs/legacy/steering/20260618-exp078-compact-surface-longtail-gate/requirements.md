# 要件

## 依頼

`exp078_compact_surface_longtail_gate` として、exp073 deterministic full replay ML anchor に exp075 compact surface prediction を long-tail 条件付きで薄く足す候補を実装する。

## 制約

- Route: `ml_model`
- 実験番号は最新 `exp077` の次として `exp078` を使い、実験名から古い backlog の `exp073_...` 表記は削除する。
- exp073 は親 anchor として参照するだけで、実験 ID として再利用しない。
- hard switch と global replacement はしない。
- exp075 compact surface は `w=0.05/0.10/0.20` 程度の小さい重みでのみ評価する。
- long-tail と評価指標に関する discussion を考慮する。
  - `docs/discussions/rogii-wellbore-geology-prediction-698860-rmse-vs-ssr-error.md`: RMSE は SSE/SSR の平方根スケールなので、少数の長い tail の SSE 改善が全体に効きやすい。
  - `docs/discussions/rogii-wellbore-geology-prediction-700340-oof-vs-lb-should-we-track-worst-well-improvements-instead-of-only-global-rmse.md`: global OOF RMSE だけでなく worst-well regression と fold/well transfer を見る。
- 再現性: `docs/06_reproducibility.md` に従い、prediction SHA、submission SHA、Kaggle kernel source/version を記録する。

## 受け入れ基準

- exp073 OOF prediction と exp075 OOF prediction を id align して、候補 policy ごとの RMSE、SSE delta、tail bucket delta、well-level regression を保存する。
- best policy の row-level prediction と gate weight を保存する。
- inference notebook は exp073/exp075 saved test predictions を読み、固定 policy で `submission.csv` を生成できる。
- `config.yaml`、train notebook、inference notebook、`SESSION_NOTES.md`、`result.md` が exp078 の目的を反映している。
- `validate_experiment.py`、notebook JSON validation、`py_compile`、`ruff check` が通る。
