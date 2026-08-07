# exp459_persistent_acceleration_state_likelihood_pf 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了し、technical gateは全PASSしたが
mechanism all-AND gateをFAILした。`stage0_fail_closed`としてbranchを閉じ、
Stage 1、inference、submissionへ進まない。fixed32はmechanism preflightであり、
CVではない。

## 仮説

exp404/417 likelihood-PFへ3値persistent acceleration状態を1つ追加すると、
exact HMMのposition×rate×acceleration全列挙を避けながら、曖昧GR区間の
rate trendを維持できる。

## 固定設定

- 親: exp417
- 実装参照・保存control: exp404 x1.0 / temperature-5
- Route: `pf_beam`
- state: `(TVT, U-rate, U-acceleration)`
- acceleration: `[-0.0005, 0, +0.0005]`
- transition: boundary-folded `0.08 / 0.84 / 0.08`
- particles / seeds: `500 / 128`
- scientific variants: 1
- Stage 0 candidate PF wells: 32
- control rerun / model / booster / HMM / Beam / GPU: すべて0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle kernel | `kentookumura/exp459-persistent-acceleration-state-likpf-train` |
| version / id_no | `1 / 129167965` |
| Kaggle status | `COMPLETE` |
| Stage 0 | `stage0_fail_closed` |
| contract test | 10 passed |
| Jupytext / py_compile / ruff | PASS |
| technical gate | 全PASS |
| mechanism gate | FAIL |

## Stage 0測定

| 測定 | 値 | gate |
| --- | ---: | --- |
| nonzero acceleration mass | `0.666245` | PASS |
| future curvature方向一致 | `0.501086` | FAIL（下限`0.60`） |
| direction positive folds | `0 / 5` | FAIL（下限`4 / 5`） |
| persistent episode SSE reduction | `-11.6190%` | FAIL（悪化） |
| persistent改善well | `7 / 16` | FAIL（下限`10`） |
| persistent改善fold | `3 / 5` | FAIL（下限`4 / 5`） |
| matched-control pooled delta RMSE | `+0.435213 ft` | FAIL（上限`+0.02 ft`） |
| matched-control by-well delta p95 | `+1.785604 ft` | FAIL（上限`+0.25 ft`） |
| candidate PF runtime | `928.287 sec` | PASS |
| full 773-well projection | `22,423.933 sec` | PASS（上限`30,600 sec`） |
| peak RSS | `0.795540 GiB` | PASS |

4 zero-acceleration sentinel wellsではprediction、log-likelihood、
resampling count、minimum ESS、position clip countがすべてexp404とbitwise一致し、
最大絶対誤差は全項目`0.0`だった。prediction、acceleration ledger、runtime ledger、
全content SHAのfreeze前にtruth / control / role-fold / episodeを読んだ行数もすべて0。

## 再現性

- deterministic anchor: false（独立rerunなし）
- base PF streamとacceleration Park-Miller streamを実装上分離
- real fixed32 sentinel 4 wellsでzero-acceleration exp404 5生成物のbitwise parityを確認
- scientific contract SHA:
  `4949c627aca356e83bbedf568aa59aa90eff142674e254159addbbb6a2f51ffe`
- prediction decompressed SHA:
  `c1464190c5949de817aa7a6e287ccc4dbca314db3f07948f6834b51cff9922ef`
- acceleration ledger decompressed SHA:
  `5c465f7ada33ec6583f7de0e08f84abbd827cb5fdcab694e1210021b60ba5206`
- gate report SHA:
  `14887feb2fcf563af173ed8c7b8abf27389e34c3b9e97ca805a2a2dacd1d2727`
- terminal log SHA:
  `01049dd17616920cb0d39d73ef40b05eeed0f317d2488ae7dc2ae4f26a420525`
- logsに全gate、runtime、SHAが揃ったためKaggle output archiveは取得していない。

## 解釈

実装は正しく動き、acceleration stateも十分なnonzero massを持ったが、filtered
acceleration符号の将来curvature一致はほぼcoin flipだった。さらにpersistent
episodeとmatched controlの両方を悪化させたため、失敗原因はstate collapseや
runtimeではなく、固定したpersistent acceleration mechanismのGR識別力と安全性に
ある。exp444またはexp367の判断は変更せず、acceleration値、transition、noise、
particle、seed、temperature、gateを同じfixed32で救済しない。

## 次

`close_branch_without_parameter_or_gate_rescue`。Stage 1、inference、submissionを
実行しない。新しいacceleration rescue候補は追加せず、既存の独立した非acceleration
仮説を優先する。
