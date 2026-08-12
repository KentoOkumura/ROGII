# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage Aが`real_rmse_vs_exp209`、well p95、worst-wellをFAILしたため、Stage B、推論、提出は事前契約により閉鎖。
- exp331内のarchitecture/loss/band/temperature/view/epoch rescue gridは禁止。

## 完了

- [x] exp295 runtime failureをtraining objectiveの計算量問題として分離した。
- [x] exp331をlocal CE-only + evaluation-time exact SSMとして採番した。
- [x] architecture、input、decoder、fold、controls、runtime gate、Stage A/B/Cを固定した。
- [x] reproducibility/SHA方針を固定した。
- [x] design-only experiment scaffoldを作成した。
- [x] compact self-contained Jupytext train候補とNotebook候補を実装した。
- [x] inference候補をStage B promotionまでfail-closedにした。
- [x] local CE-only、SSM非呼び出し、truth-late、controls、SHA、Stage 0選定/外挿の専用testを実装した。
- [x] 固定16-view T4 microbenchmarkをKaggle version 1で実行し、保守的fold外挿`4.516839 h`、peak GPU memory`1.924052 GB`で8.5時間/14 GB gateをPASSした。
- [x] report SHA`401d98f2cdc9ced437d66fc02bbe49b9287d4772e4d9036719c573a90b785c59`を固定し、Stage Aは別承認待ちへ戻した。
- [x] Stage Aのactive variant/model/fold/booster数とcontrol再学習0を再確認し、fold 0のGPU push承認を得た。
- [x] Stage A fold 0の1 neural modelをKaggle T4 version 1で完走し、runtime`4.115497 h`、peak`1.889884 GB`を確認した。
- [x] real GR RMSE`24.760360`対exp209`12.671087`、well p95`44.560719`対`26.301518`、worst regression`+63.109520 ft`として科学gate FAILを確定した。
- [x] truth-freezeと全artifact SHAを照合し、decision`close_stage_b_without_exp331_rescue_grid`を設定・記録・戦略へ反映した。
