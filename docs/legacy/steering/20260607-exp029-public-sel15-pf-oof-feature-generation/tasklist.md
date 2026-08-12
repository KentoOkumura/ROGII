# タスクリスト

## TODO

- all-well summary で PF が hold より悪い条件を分析する。
- exp026 OOF と結合し、次の selector / residual correction 実験を切る。

## 進行中

- なし

## ブロック中

- なし

## 完了

- Steering requirements/design/tasklist を作成した。
- `config.yaml` に PF/Beam OOF-like feature generation の設定を追加した。
- `public_sel15_pf_oof.py` を追加し、train well の途中以降を隠す cutoff、PF ensemble、beam ensemble、selector、feature CSV 出力を実装した。
- Train/inference notebook と実験記録を exp029 用に更新した。
- `py_compile`、`ruff`、`validate-exp`、train package 生成を通した。
- 1 well / 2 seeds / 20 particles の local smoke で 1056 feature rows を生成した。
- Kaggle smoke version 1 の `distance_bucket` failure を修正した。
- Kaggle smoke version 2 を `kentookumura/exp029-sel15-pf-oof-train` で完了し、20 wells / 43,542 rows の feature artifact を取得した。
- smoke output を取得して `SESSION_NOTES.md`、`result.md`、`metrics.json` に rows / RMSE / output path を記録した。
- all wells / cutoff 0.65 / 16 seeds / 250 particles / gzip output で version 3 を実行した。
- version 3 output を取得し、773 wells / 1,782,279 rows の feature artifact を同期した。
- all-well 結果を `SESSION_NOTES.md`、`result.md`、`metrics.json`、`KAGGLE_DIRECTION.md` に記録した。
