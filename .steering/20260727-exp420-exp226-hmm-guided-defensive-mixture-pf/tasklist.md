# タスクリスト

## 未着手（別承認後）

- なし。現行契約の正規Notebook採用、package、push、Stage 0 runは行わない。

## 進行中

- なし

## ブロック中

- exp411の同一scheduleがdirection / control mechanism gateをFAILしたため、
  現行exp420 Stage 0 prerequisiteは不成立。

## 完了

- HMM / PF / exp226の誤差原因とexp408 / exp410 / exp419 / exp411契約を確認した。
- routeを`pf_beam`へ固定した。
- steeringと実験scaffoldを作成した。
- HMMはrate innovation scheduleだけ、exp226はgeometry rateだけ、最終出力は
  importance-corrected PF 1 variantだけとする統合境界を固定した。
- Stage 0 fixed44、full 773-well、実行量、gate、再現性、禁止救済を事前登録した。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へ設計を反映した。
- Jupytext percent形式のcompact self-contained train候補を別名で作成した。
- untreated HMM forward schedule、inactive / active proposal、importance correction、
  temperature-5 aggregationを実装した。
- fixed32 / fixed12 union 44-well manifestを入力SHAつきでfreezeする実装を作成した。
- proposal allowlist、truth-late join、schedule / prediction / diagnostic SHAを実装した。
- all-guidance-zero exp404 parity、HMM-weight-zero exp419 proposal parity、
  `p0/q <=2`、schedule state-machineのcontract testを作成した。
- Stage 0 / full OOFの4-shard orchestrationとfail-close gateを実装した。
- Jupytext test、`py_compile`、Ruff F821、`make validate-exp`をPASSした。
- 正規Notebook編集、package、Kaggle実行、inference、submissionは行っていない。
- exp411 Version 5の同一schedule / fixed32結果を確認し、現行exp420契約を
  prerequisite FAIL・未実行として停止した。
