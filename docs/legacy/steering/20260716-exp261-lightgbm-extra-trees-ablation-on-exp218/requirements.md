# 要件

## 依頼

`lightgbm_extra_trees_ablation_on_exp218` を実験化し、exp218 の回帰 LightGBM に
`extra_trees=True` を加える単一変数 ablation を実装する。

## 制約

- Route: `ml_model`。
- 親は `exp218_gr_wavelet_rotation_confidence_features_on_exp148` とし、380-feature surface、GroupKFold、seed、GPU deterministic flags、early stopping、その他の LightGBM parameter を固定する。
- exp218 control は再学習せず、保存済み metrics / OOF prediction を historical baseline として使う。
- 実行プランは `lgb1_probe`（1 config × 5 folds = 5 boosters）と `full_family`（3 configs × 5 folds = 15 boosters）を用意する。
- ユーザーが実行プランと booster 数を明示承認するまで Kaggle train は fail-closed とする。
- inference / submission は train-side guard 通過後に同じ exp261 内で実装判断し、初回実装範囲には含めない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、Kaggle bootstrap、feature/model/prediction SHA を記録する。

## 受け入れ基準

- `config.yaml` に `extra_trees: true`、実行プラン、config index、fold数、booster数、control非再学習を明記している。
- Jupytext percent形式のcompact self-contained train sourceと生成した正規train notebookで、入力確認、exp218 feature surface再構築、選択configのCV、metrics、OOF、by-well / bucket / hidden-like stress、feature importance、model manifest保存を追える。
- `extra_trees=True` 以外の対応する親LightGBM parameterが変わっていないことをruntime assertionで確認する。
- 親exp218 OOFを使い、OOF相関と事前固定blend weight `0.25/0.50/0.75` を監査できる。
- `lgb1_probe` と `full_family` のどちらもコード上選択できるが、未承認状態では学習開始前に停止する。
- `jupytext --test`、`py_compile`、`ruff --select F821`、strict experiment validationが通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel versionが記録されている。
- gzip生成物を比較する場合は、raw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録している。
