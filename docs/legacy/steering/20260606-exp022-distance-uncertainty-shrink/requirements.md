# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭である `distance_uncertainty_shrink` を個別実験として実装する。`exp021_distance_weighted_inference_postprocess` の weighted LightGBM + bucket shrink を親にし、inference-safe な不確実性 proxy による residual shrink を比較できる train / inference notebook を用意する。

## 制約

- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- `TVT_input` の既知 prefix より後ろの true target を特徴・後処理係数に使わない。
- GroupKFold by well を維持し、valid well の行を学習に混ぜない。
- train-only formation columns は使わない。
- 初回候補は config 固定パラメータとし、同じ OOF target residual で shrink 係数を fit した値を clean CV として扱わない。
- 生データ、大きな OOF artifact、モデル重みを Git に保存しない。

## 受け入れ基準

- `docs/legacy/steering/20260606-exp022-distance-uncertainty-shrink/` に仮説、設計、タスクが記録されている。
- `experiments/exp022_distance_uncertainty_shrink/` が `config.yaml`、`settings.py`、train/inference notebook、実験モジュール、記録ファイルを持つ。
- train notebook で `weighted_raw`、`weighted_distance_bucket_shrink`、固定 uncertainty shrink 候補を OOF 比較できる。
- inference notebook で `postprocess.selected_method` の候補を使って `submission.csv` を生成できる。
- `validate_experiment.py`、Python compile、notebook code compile が通る。
