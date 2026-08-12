# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 1 exact-HMM、inference、submissionはStage 0固定gate FAILにより禁止する。

## 完了

- 2026-07-22: `exp337_prefix_backtested_structure_sigma_gr`として採番し、steeringと実験scaffoldを作成した。
- 2026-07-22: 科学的親をexp209、exp307を失敗根拠として固定した。
- 2026-07-22: `sigma_eff^2=sigma_finite^2+tau_structure^2`、60/40 calibration、60/80 rolling origins、fallback/clipを固定した。
- 2026-07-22: compact self-contained Stage 0 train、fail-closed inference、専用testを実装した。
- 2026-07-22: Jupytext round-trip、py_compile、Ruff、専用・shared Notebook test、strict experiment/template validationをPASSした。
- 2026-07-22: ユーザー承認に基づきcompact trainを正規Notebookへ採用し、Kaggle CPU version 1を実行した。
- 2026-07-22: Stage 0は773 wells、両origin coverage 100%、fallback 0、zero-fill比5/5 folds改善だったが、finite-only比0/5 folds、full-prefix median tau `0.0`で固定gateをFAILした。
- 2026-07-22: 同一結果上の救済をせず枝を閉じ、Stage 1、inference、submissionを未実行のまま終了した。
