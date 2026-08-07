# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `AGENTS.md`、`KAGGLE_DIRECTION.md`、`docs/06_reproducibility.md`を確認した。
- 親exp268、exp209 emission、exp288可視化、exp170/211/132 negative evidenceを確認した。
- 親、5候補、calibration、score、negative control、success/stop guard、禁止事項をsteeringへ固定した。
- target-free loader/scorer、aggregate hard guard、post-freeze truth readout、fail-closed inferenceを実装した。
- canonical Jupytext train notebookを採用し、専用contract test 11件、Jupytext、py_compile、Ruff、
  strict experiment validation、template validationを通した。
- 1 audit variant / 0 configs / 0 trained folds / 0 boosters / 0 HMM/PF runs、control再生成なし、
  private CPU、GPU/TPU/internet offをpush前に確認した。
- exp268 aggregate version 1を先に完了し、773 wells / 3,783,989 rows、candidate diversity、固定SHAを確認した。
- exp292 private CPU version 1、kernel id `127888550`を完了した。
- technical guard、入力・artifact・target-free score/selection・fold manifest SHAを確認した。
- H256 coverage 29/773 wells、AUC lift -0.046991、RMSE gain 0、0/5 fold改善を記録した。
- 事前登録どおり`FAIL_CLOSE_NO_RESCUE_GRID`とし、inference/submissionを生成せずbranchを閉じた。
- `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md`を更新した。
