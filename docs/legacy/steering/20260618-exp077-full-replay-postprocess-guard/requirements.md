# 要件

## 依頼

`exp073_full_replay_postprocess_guard` を実装する。ただし既存の `exp073` は使わず、実験番号を正しくインクリメントして `exp077_full_replay_postprocess_guard` とする。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- exp073 の deterministic full replay OOF prediction を基準に、再学習なしの保守的後処理を監査する。
- 後処理候補は residual clip、tail 開始部 fade、flat-prefix hold blend、PF confidence residual clip、PF-vs-ML disagreement tiny gate、long-tail tiny gate に限定する。
- 学習コードは fold/model ごとの LightGBM feature importance を保存し、fold 平均の重要度を matplotlib で表示できるようにする。
- 再現性: `docs/06_reproducibility.md` に従い、exp073 / exp072 の SHA と Kaggle source を追える設計にする。

## 受け入れ基準

- `experiments/exp077_full_replay_postprocess_guard/` が存在する。
- train notebook が postprocess audit を実行し、metrics / bucket metrics / predictions / summary を保存する。
- train notebook が exp073 saved booster から fold 平均 feature importance plot を表示する。
- `config.yaml`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が exp077 として整合している。
- `validate_experiment.py` と notebook JSON validation が通る。
