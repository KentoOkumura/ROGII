# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 実験仮説、pair set、512-row block、K=3、leakage/reproducibility guardを固定した。
- Stage 0 0 booster、Stage 1条件付き30 CPU booster、control再学習0を記録した。
- exp264 Stage Bのcanonical score、model manifest、metricsの3 artifactをfail-closed入力に固定した。
- Stage 0 pairwise divergence fingerprint、outer-fold regime、streaming score auditを実装した。
- 15 pair、target-free schema、offset耐性、cluster再現性、streaming集計のunit test 5件を追加した。
- Jupytext、構文、ruff、targeted 5 / repository 74 tests、strict validate-exp、doc reviewをPASSした。
- private CPU / internet off / 2 kernel sources / run_on_push falseのcanonical packageをprepareした。
- Stage 0の0 booster / control再学習0を再提示し、ユーザーからKaggle実行承認を受領した。
- canonical kernel version 1をpushし、id_no `127531288`、RUNNINGを確認した。
- version 1は`confidence__*__sigma_tvt__median`の誤検出でERROR。segment完全一致guardへ修正した。
- targeted 6 / repository 79 tests、Jupytext、構文、F821、strict validate-expをPASSした。
- 修正版を同じcanonical slugへversion 2としてpushし、初期status `RUNNING`を確認した。
- version 2 `COMPLETE`、3,783,989 rows / 773 wells / 7,787 blocks / 295 featuresを確認した。
- occupancy 2.20% / 5.28% / 92.53%でStage 0 FAIL、stability 1.000と数値separabilityはPASS。
- partial terminal blockとrare selfgr-vs-exact gap外れ値がclusterを支配した原因を診断した。
- Stage 1 30 CPU boosters、inference、submissionを未実行のまま不採用として閉じた。
