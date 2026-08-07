# exp386_cycle_consistent_rgt_scenario_bank 結果

## 仮説

順序付き地層区間を RGT に変換し、outer-train の井戸間対応をサイクル整合グラフとして扱えば、単一の局所地層面補間よりも対象井戸の真値経路を含む複数の物理解を生成できる。

## 設定

- 系統: topology-first RGT の独立系統
- route: `pf_beam`
- 比較: exp226、exp293、exp301、exp377、exp383
- 検証: exp226 と同じ outer 5-fold、`well_id` group
- 主判定: scenario-bank oracle RMSE
- scenario: 井戸あたり 8〜32
- target GR: 使用禁止

## 結果

Kaggle private CPU version 1（id_no `128478384`）は16-well / 5-fold
Stage 0 preflightを`2411.033 sec`で完了した。Stage 0はFAIL_CLOSEだった。

| メトリック | 値 |
| --- | --- |
| 実装検証 | 専用test 11件、py_compile、Ruff、Jupytext round-trip PASS |
| RGT source coverage | 0.989847（PASS） |
| graph query coverage | 0.0（FAIL） |
| scenario-bank well coverage | 0.0（FAIL） |
| scenario count p05 | 0.0（FAIL） |
| finite-path coverage | 0.0（FAIL） |
| cycle residual p95 | 2.363303 interval（FAIL、上限0.10） |
| projected full runtime | 2867.246 sec（PASS） |
| peak RSS | 1.145931 GB（PASS） |
| target GR / valid Formation / suffix truth read | 0 / 0 / 0（PASS） |
| CV | Stage 1/2未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 合格条件

- Stage 0: leakage 0、bank coverage 98%以上、cycle residual p95 0.10以下、resource gate 内
- Stage 1: prefix oracle gain 0.50 ft以上、5 fold 中4 fold以上で改善
- Stage 2: scenario oracle RMSE 5.50 ft以下、全5 foldで改善

## 再現性

- deterministic anchor: 未確立
- seed policy: RNG なし、不変キーによる安定順序
- artifact SHA / prediction SHA: 実行時にlogical/decompressed content SHAを保存する実装済み
- rerun: 初回成功後に content SHA 一致を要求

## 解釈

ordered RGT自体のcoverageとtarget-free境界、計算資源は成立した。しかし、global edge
potentialは固定cycle gateを大幅に超え、16対象井戸では有効routeが1本も残らなかった。
空のtarget path SHAは
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
である。現行graph/path contractはscenario bankを生成できず、科学評価へ進めない。

logsはrouteの候補数、integration、monotonicity、cost-finiteのどこで棄却されたかを
分解していないため、scenario 0の詳細原因は推測しない。少なくとも
cycle residualの非整合と全route棄却は独立したStage 0停止理由である。

## 次

full run、Stage 1/2、exp387、inference、submissionを行わず閉じる。再訪時は
edge/cycle/pathの値を調整せず、固定設定のrejection funnelとedge residual成分だけを測る
診断readoutから開始する。
