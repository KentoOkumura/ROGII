# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- backlog、exp209/268、`docs/06_reproducibility.md`を確認した。
- ユーザー承認によりper-well known-prefix Huber plane、5 ellipse prototypes、saved scalar control、
  candidate-bank auditのみという設計を固定した。
- experiment scaffold、Huber plane、geometry guard、5 prototype、gradient-residual-rate exact HMMを実装した。
- 2 shard generator、aggregate audit、disabled inferenceをself-contained notebookとして実装した。
- exp273固有contract test、Jupytext、py_compile、Ruff F821、strict validationを完了した。
- `experiment_summary.md`と`KAGGLE_DIRECTION.md`を実装済み・Kaggle CPU未実行へ更新した。
- 2026-07-18の実行承認後、5 variants / 2 shards / 最大3,865 HMM well-runs / 0 boosters、
  parent/control再生成なしを再確認した。
- shard 0/1 packageのmetadata、run-on-push、CPU/private/internet設定、bootstrap config SHAを照合した。
- Kaggle CPU shard 0/1 version 1を完了し、773 wells / 3,783,989 rows、well overlap 0、
  stable shard assignment、raw/decompressed/schema/by-well/input-manifest SHAを確認した。
- aggregate packageへshard別rows / wells / raw/decompressed SHA hard guardを追加し、
  metadata、4 kernel sources、bootstrap config SHA、0-booster契約を確認した。
- Kaggle aggregate version 1（id_no `127731254`）を完了し、773 wells / 3,783,989 rows、
  saved-control parity、10 CSV SHA、aggregate prediction content SHAを確認した。
- scalar RMSE 11.938287に対してbest gradientは12.169871（`+0.231584 ft`）で、5候補を不採用とした。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`KAGGLE_DIRECTION.md`へnegative resultと
  0-booster prefix-stability readout候補を反映した。

## 成果物

- `experiments/exp273_two_dimensional_formation_gradient_transition/` 配下の4 notebook source / ipynb。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`。
- `tests/test_exp273_two_dimensional_formation_gradient_transition_contract.py`。
