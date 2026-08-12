# タスクリスト

## TODO（別実験・別承認）

- BUDAをtarget-free source supportだけで事前除外する固定5面contractを検討する場合は、
  exp436を再実行せず、新しいsteeringと実験番号へ切り出す。

## ブロック中

- Stage 1、Stage 2、full-train inference、submission、exp226/exp263とのblend。
- formation選択/重み学習、fault、GR、HMM/PF/Beam、ML、selector。

## 完了

- 2026-07-29: ユーザー指示`exp436を実装してください`により実装承認を取得した。
- 2026-07-29: compact self-contained train候補へStage 0/1/2のfail-closed
  orchestrationを実装した。
- 2026-07-29: contact / fold exclusion / graph / IRLS+LSQR / fixed support /
  rolling-origin / truth-late gateの10 contract testsを追加した。
- 2026-07-29: 構文、Ruff F821、専用pytest、Jupytext変換/testを実行した。

- 2026-07-29: 誤った単一`P(X,Y)`を廃止し、6地層面別`U_k(X,Y)`へ訂正した。
- 2026-07-29: outer-train first contact、1 global sparse surface/formation/foldへ固定した。
- 2026-07-29: target raw formation不使用、固定`K_w`、等重みanchor差を固定した。
- 2026-07-29: exp381 absolute contact-TVT FAIL、exp383 resource FAIL、
  exp273 prefix-plane negativeとの境界を固定した。
- 2026-07-29: Stage 0 resource/integrity、Stage 1 prefix rolling-origin、
  Stage 2 truth-late OOFのAND gateを固定した。
- 2026-07-29: route、実行量、禁止事項、SHA、determinism契約を固定した。
- 2026-07-29: ユーザー指示`実行してください`により正規train notebook採用、
  Kaggle CPU Stage 0 package/push/runの承認を取得した。
- 2026-07-29: version 1はBUDA `5 < 32`をfail-close保存前に例外化したため、
  gateを変えずにsolver failure manifestと正常なfail-closeを実装した。
- 2026-07-29: 専用testを11件へ増やし、Jupytext、構文、Ruff F821、
  strict experiment validationを再度PASSした。
- 2026-07-29: Kaggle private CPU version 2（id_no `129058940`）がCOMPLETE。
  BUDAは5 foldsで4–6 source wells、固定最小32をFAILし、Stage 0でbranchを閉じた。
- 2026-07-29: Stage 1、Stage 2、inference、submissionは未実行。
