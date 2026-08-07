# 要件

## 依頼

exp070 は exp063 の再現性確保が目的だったが、exp063 の full public replay 196 features ではなく compact tracker 65 features を入力にしてしまったため無効。
exp071 では exp063 と同じ raw public replay feature generation を使い、`pixiux_likpf_public_replay` の full 196 feature surface を維持したまま、LightGBM の再現性向け実行設定だけを変更して train/inference をやり直す。

## 制約

- Route: `ml_model`
- 親実験は `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`。
- 特徴量生成は exp063 の `public_notebook_replay_audit.py` と同じ public replay 実装を使う。
- train の feature count は exp063 `pixiux_likpf_public_replay` と同じ 196 features にする。
- compact tracker 65 features は学習入力に使わない。
- 変更してよいのは LightGBM の device / deterministic / thread / gpu precision 系の実行設定だけ。
- inference は current raw test files から exp063 public replay PF/Beam/likelihood-PF features を再生成し、保存済み booster を適用する。
- Kaggle 実行を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- train notebook/package が `pixiux_likpf_public_replay` full 196 features で学習する。
- feature schema と manifest に 196 feature names が保存される。
- GPU train は `enable_gpu=true`、`device_type=gpu`、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`num_threads=8` を明示する。
- inference notebook/package は train booster manifest を読み、raw test feature replay を実行して `submission.csv` を生成する。
- `validate_experiment.py`、notebook JSON validation、`py_compile`、`ruff check` が通る。
