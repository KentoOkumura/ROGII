# exp370_triggered_reset_rejuvenation_pf

## 状態

- Route: `pf_beam`
- 状態: Stage 0 technical PASS / scientific FAIL、branch閉鎖
- Kaggle: private CPU version 2 `COMPLETE`、id_no `128591535`
- CV / LB / Submit: 予測CV・提出なし
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

GR changeとESS collapseが同時に起き、fold-safe atlasがalternative TVT modeを
coverageできる場合だけ、10%粒子を再注入すればmode slipを回復できる。

## Stage 0

- 500 particles × 1 seed × 773 wells = 773 diagnostic seed-well runs。
- triggerはknown-prefix q99.5 GR change AND ESS/N `<=0.20`、refractory 512行。
- atlasはouter-train wellsだけで構築し、256-row GR ZNCCのtop 3を10 ft以上離す。
- target truthとhidden-like roleはtrigger / atlas / proposalのSHA freeze後にjoinする。
- saved exp072 `likpf_mean`をread-only baseとし、full parent PF replayは0。

## 検証方針

trigger AUC、circular negative-control差、trigger率、atlas top-3 coverage、
saved base比coverage gain、5-fold再現性、hidden-like 2面をAND評価する。
全gate PASSと別承認が揃う場合だけStage 1へ進むfail-closed設計とした。

## 実行結果

- technical gate: PASS
- scientific gate: FAIL
- accepted triggers: 13 / 3,685,818 eligible rows
- trigger rate: 0.000003527
- trigger AUC / circular差: 0.499998 / -3.76e-12
- atlas top-3 within10 coverage: 0.076923
- saved likPF coverage / atlas gain: 0.846154 / -0.769231
- passing folds: 0 / 5
- Stage 1 eligible: false
- runtime: 671.342秒

version 1はKaggle competition mount resolverの欠陥で科学計算前に停止した。
version 2では`competitions/<slug>/train`優先とpaired 773-well guardを追加し、
全773 runsを完了した。

## 実装入口

- 正規train:
  `exp370_triggered_reset_rejuvenation_pf_train.ipynb`
- Jupytext source:
  `exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py`
- fail-closed inference候補:
  `exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py`

正規trainはcompact self-contained候補を採用済み。正規inferenceはplaceholderのまま。
完了後は`execution.run_stage_0=false`、`run_stage_1=false`、
`run_inference=false`、`create_submission=false`へ戻した。

## 所見

triggerはほぼ発火せず、発火した行のatlas proposalもsaved likPFを大幅に下回った。
閾値・top-k・注入率を救済せず、このrejuvenation branchは閉じる。Stage 1、
inference、submissionへ進まない。

## 結論

target-free triggerとatlas proposalの双方が固定gateを満たさず、仮説は支持されなかった。
