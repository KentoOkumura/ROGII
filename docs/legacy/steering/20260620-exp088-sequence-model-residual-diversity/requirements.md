# 要件

## 依頼

`sequence_model_residual_diversity` を実装する。既存 discussion / public notebook で参考になりそうな情報を参照し、古い exp063 前提は現行 anchor に合わせて更新する。

## 制約

- Route: `ml_model`
- 親実験は古い exp063 ではなく、現行 raw deterministic ML anchor の `exp073_gpu_reproducibility_guard_for_exp063_full_replay` とする。
- 入力特徴は `exp072_exp063_full_replay_feature_cache` の train-only deterministic cache を使う。
- sequence model は本命化せず、LightGBM anchor との residual diversity 診断に限定する。
- `submission.csv` は生成しない。
- PyTorch は float32 固定。AMP / bfloat16 は使わない。
- stochastic 処理は fold / variant から stable seed を作り、`docs/06_reproducibility.md` に従って記録する。

## 参照した既存情報

- discussion 699289: pure tabular model は sequence / spatial context を落とすため、PF / Beam / spatial consensus が重要。
- discussion 699853: CNN/MTP は multi-mode trajectory の候補として有用だが、learned GR matcher は一般化が難しい。
- discussion 707613: PF を NN 化する前に、candidate が truth trajectory 近傍を含むかを測るべき。
- discussion 703344: transformer / bfloat16 で copy task failure。float32 sanity check が必要。
- public notebook inventory 2026-06-11: CNN/seq 系は低優先度、上位は PF/Beam/TabICL/physical stack が中心。

## 受け入れ基準

- exp088 の steering / experiment directory が作成されている。
- `config.yaml` に route、親実験、入力 cache、runtime、seed policy が明記されている。
- train notebook が exp088 の目的、入力確認、実行、生成物確認を含む。
- inference notebook は diagnostic-only no-op で、submission を作らないことを明示する。
- GRU / TCN の fold-out OOF prediction、overall metrics、bucket metrics、diversity metrics、blend metrics を保存する実装がある。
- 入力 cache / prediction と出力 prediction の decompressed content SHA を summary に記録する。
- `ruff`、`py_compile`、notebook JSON validation、`validate_experiment`、Kaggle notebook prepare が通る。
