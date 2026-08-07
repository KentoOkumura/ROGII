# 要件

## 依頼

`gpu_reproducibility_guard_for_exp063_full_replay` を実装する。目的は LightGBM training だけではなく、PF/Beam/likelihood-PF test feature regeneration まで含む end-to-end reproducibility guard とする。`exp072_exp063_full_replay_feature_cache` が未完了の時点では Kaggle push 可能な notebook 準備まで進め、完了後に train / inference を順に実行する。

## 制約

- Route: `ml_model`
- 親実験は `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`。
- train feature input は `exp072_exp063_full_replay_feature_cache` が生成する 196-feature full replay train cache。
- exp070 の 65-feature compact tracker surface は使わない。
- train notebook は LightGBM reproducibility guard のみを実行し、raw test files を読まない。
- inference notebook は exp072 の test cache を使わず、current raw test files から exp063 public replay PF/Beam/likelihood-PF features を stable per-well seed で再生成する。
- exp072 完了前に Kaggle push / 実行はしない。

## 受け入れ基準

- `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/` に config、settings、train/inference notebooks、実装 script、記録ファイルがある。
- train notebook は exp072 cache の存在、行数、列数を確認し、feature count が 196 でない場合は fail する。
- LightGBM mode は GPU double precision + deterministic flags + fixed threads を既定にする。
- metrics、OOF prediction SHA、model manifest / model SHA、runtime を保存する。
- inference notebook は exp073 train saved boosters を読み、deterministic raw test regeneration で `submission.csv` を作れる。
- inference output では feature SHA と submission SHA を記録し、同一条件 rerun で固定されるか確認できる。
- `py_compile`、notebook JSON validation、`validate_experiment.py`、Kaggle notebook preparation が通る。
