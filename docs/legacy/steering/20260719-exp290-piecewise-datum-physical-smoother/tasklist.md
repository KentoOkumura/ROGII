# タスクリスト

## TODO

- なし。Stage 0 scientific guard FAILによりbranchを閉じた。

## 進行中

- なし

## ブロック中

- Stage 1、raw-test inference、submissionはStage 0 scientific guard FAILによりclosed。

## 完了

- 物理モデル単独LB 6.5目標とoracle禁止要件を固定した。
- `TVT=g+delta`、absolute `delta in [-15,+15] ft`、0.5 ft gridのsingle-model contractを固定した。
- well固有pseudo-cut calibrationと、Type Well/neighborをscale/hazard/noiseだけに使う階層事前を固定した。
- minimum duration、multi-window reset gate、non-cumulative transition、posterior mean出力を固定した。
- Stage 0 / Stage 1の順序、成功条件、停止条件を固定した。
- outer-valid formation/true suffix除外、truth-after-freeze、fold-safe/raw-test-compatible入力契約を固定した。
- 再現性、SHA、runtime、leakage、CV/LB、oracle禁止guardを設計へ記録した。
- `exp290`のdesign-only experiment scaffold、`config.yaml`、README、SESSION_NOTES、result、metricsを作成した。
- `KAGGLE_DIRECTION.md`の未着手backlog最上位と`experiment_summary.md`へ記録した。
- `make validate-exp EXP=exp290_piecewise_datum_physical_smoother`をstrict PASSした。
- 2026-07-19 の追加依頼を Stage 0 実装承認として記録した。
- Jupytext percent形式のcompact self-contained train source/notebookを実装した。
- exp226 fold-safe geometry replay、3 fixed pseudo-cut、Type Well Huber calibration、outer-train scale hyperprior、variance-only spatial k=16を実装した。
- absolute 61-state x 5 duration phaseのlog-space exact forward-backward、posterior mean、entropy、reset probabilityを実装した。
- window単位prediction SHA freezeとtruth-after-freeze評価、Stage 0 guard、生成物/SHA manifestを実装した。
- fail-closed inference source/notebookを実装した。
- target/formation exclusion、pseudo-cut mask、stable neighbor、transition、state bound、truth-after-freeze、hierarchy符号禁止、disabled inferenceの専用tests 11件を追加した。
- Jupytext変換/`--test`、`py_compile`、`ruff`、専用pytest、strict experiment validationをPASSした。
- 追加依頼「実行してください」をcanonical train notebook採用とKaggle CPU Stage 0 push承認として記録した。
- canonical id/title、CPU、GPU/internet off、exp226 kernel source、bootstrap config/source SHAを監査してpackage化した。
- `kentookumura/exp290-piecewise-datum-physical-smoother-train` version 1（id_no `127881061`）を完走した。
- 296,832 rows / 773 wells / 2,319 windowsでtechnical guardを全PASSした。
- pseudo-tail RMSE改善0.033519 ft、sign accuracy 0.483111、fold改善5/5、well p95微悪化によりscientific guard FAILを確定した。
- 必要なoutputだけを取得し、prediction raw/decompressed SHA、state manifest SHA、hyperprior/pseudocut/neighbor content SHAを記録した。
- failure policyどおりparameter/group rescue、Stage 1、raw-test inference、submissionを実施せずbranchを閉じた。
