# 要件

## 依頼

`row_neighbor_input_context_features_on_exp148` backlog を実装する。CPU 実行で timeout を避けるため、LightGBM 学習コードは `lgb0`、`lgb1`、`lgb2` に分割する。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- control / parent 再学習はしない。
- 学習は `row_neighbor_input_context_addonly` のみ。
- `TVT_input`、OOF prediction、前行 model prediction、valid/test true TVT、oracle best、true-error rank、evaluation label を feature source に使わない。
- lead / centered rolling を使う場合は、hidden inference でも同一 well の評価区間全体が見える前提を記録する。
- Kaggle GPU は使わず、CPU LightGBM deterministic flags で実行する。
- 再現性は `docs/06_reproducibility.md` に従い、入力 cache、feature schema、model manifest、prediction SHA を記録する。

## 受け入れ基準

- exp220 の `config.yaml` に `experiment.route: ml_model` と row-neighbor feature design が明記されている。
- `train_lgb0` / `train_lgb1` / `train_lgb2` notebook があり、それぞれ 1 LightGBM config x 5 folds だけを学習する。
- 標準 `train` notebook は分割実行契約を表示し、誤って 15 boosters を一括学習しない。
- `validate-exp`、Jupytext 変換、`py_compile`、`ruff --select F821,F401` が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
