# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- exp193 の steering docs を作成した。
- exp190 の exp148 add-only runner をコピー元に実験フォルダを作成した。
- `typewell_late_interval_context` feature group を config に追加した。
- train runner に raw typewell late interval context feature builder を追加した。
- planned Kaggle train cost を 1 variant、3 LightGBM configs、5 folds、15 boosters、control retraining なしとして記録した。
- Jupytext train / inference を `.ipynb` に変換し、`--test` を通した。
- `py_compile` と `ruff --select F821,F401` を通した。
- raw-only feature builder smoke を通した。
- `validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148` を通した。
- Kaggle train package を `kentookumura/exp193-typewell-late-interval-context-features-addonly-exp148-train` / `exp193 typewell late interval context features addonly exp148 train` で prepare し、package py_compile を通した。
- Kaggle train v1 を `kentookumura/exp193-typewell-late-context-exp148-train` で完了した。
- CV、fold metrics、feature importance、prediction SHA、model manifest SHA、model SHA を記録した。
- train-side は `lgb_mean` 8.456665438542778、exp148 から -0.04461574335304164 改善として supported と判断した。
- 同じ exp193 内で current-test typewell context feature generation と saved-booster inference を実装した。
- Kaggle inference package を `kentookumura/exp193-typewell-late-context-exp148-inference` / `exp193 typewell late context exp148 inference` で prepare し、package py_compile と `ruff --select F821,F401` を通した。
- Kaggle inference v1 は `generator.candidates` 欠落で失敗し、exp145/exp148 と同じ generator block を追加して修正した。
- Kaggle inference v2 を `kentookumura/exp193-typewell-late-context-exp148-inference` で完了した。
- inference output で feature schema exact match、generated typewell context feature count 19、fallback 0、sample submission 互換、prediction/submission SHA を確認した。
- submit-check を PASS した。
- competition submit ref `54347471` の scoring 完了を確認し、Public LB 7.946 を記録した。
- exp148 GPU inference v7 Public LB 7.960 からは -0.014 改善したが、exp148 CPU runtime submission Public LB 7.921 には +0.025 届かないため、ML route submitted anchor には採用しないと補正した。
