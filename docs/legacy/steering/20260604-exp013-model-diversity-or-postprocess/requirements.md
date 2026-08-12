# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ最上位
`exp013_model_diversity_or_postprocess` を実験化する。

`exp012_single_catboost_lightgbm_residual` の `lightgbm_no_gr` を anchor とし、
CV を崩さない保守的な postprocess と、小さな model diversity 候補を同一
GroupKFold OOF 上で比較できるようにする。

## 制約

- hidden test の未来 `TVT`、評価区間の集計値、train-only formation columns は使わない。
- split、target、row sampling、主要 residual feature は `exp012` と同じにする。
- 初回 full notebook 実行は Kaggle 上を正とし、ローカル notebook 実行はしない。
- 提出候補は CV が `exp012 lightgbm_no_gr` 13.549257 を下回る、または悪化が十分小さく Public LB anchor 目的が明確な場合だけにする。

## 受け入れ基準

- `experiments/exp013_model_diversity_or_postprocess/` が作成され、config と notebook 名が exp013 に更新されている。
- train notebook が `lightgbm_no_gr` OOF の row-level artifact を保存し、raw / Savitzky-Golay smoothing / drift shrink / anchor 近傍 damping / 距離 bucket residual shrink / 小さな diversity OOF を比較する。
- inference notebook が `ablation.selected_variant` と `postprocess.selected_method` に従って最終予測へ同じ後処理を適用する。
- `validate_experiment.py`、ruff、py_compile、Kaggle notebook package 生成が通る。
