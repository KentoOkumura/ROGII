# 要件

## 依頼

今後の実験で exp063 の train-side PF/Beam/likelihood-PF features を再利用しやすくするため、モデル学習を行わず、train 特徴量を再生成して保存するだけの Kaggle Notebook を作成する。

## 制約

- Route: `pf_beam`
- GPU quota を消費しない。Kaggle metadata は `enable_gpu=false`。
- LightGBM / CatBoost / Ridge / submit 用 prediction は実行しない。
- exp063 と同じ `public_notebook_replay_audit.py` の raw public replay 実装を使う。
- train の PF/Beam/likelihood-PF features だけを再生成する。
- test features は各実験の inference notebook 内で current raw test files から再生成するため、この cache notebook では作らない。
- 後続実験が kernel source として読みやすいように、full feature frame と schema/summary を保存する。

## 受け入れ基準

- Notebook は setup、raw input check、train/test feature generation、artifact summary のセル構成を持つ。
- 出力に full `pixiux_likpf_public_replay` train feature CSV を含む。
- 出力に feature schema と summary JSON を含む。
- train feature count は exp063 `pixiux_likpf_public_replay` と同じ 196 features であることを summary に記録する。
- `validate_experiment.py`、notebook JSON validation、`py_compile`、`ruff check` が通る。
- Kaggle package は CPU-only (`enable_gpu=false`) で生成される。
