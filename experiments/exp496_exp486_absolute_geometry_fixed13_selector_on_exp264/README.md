# exp496_exp486_absolute_geometry_fixed13_selector_on_exp264

## 状態

- Route: `ensemble`
- 状態: Kaggle CPU version 1 `COMPLETE` / scientific FAIL / terminal close
- CV: fixed13 hard `8.461357622`、parent fixed12比`-0.191174334 ft`
- LB / Submit: なし
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp486_exp226_geometry_residual_likelihood_pf`

## 仮説

exp486 Absolute版はpooled RMSEを`1.187584 ft`改善したが、一部wellを大きく
悪化させた。exp264 selectorへ13本目として追加し、target-free confidenceと
既存候補とのdisagreementから有効な局所だけを選べれば、平均改善とtail安全性を
両立できる可能性がある。

## 変更点

- exp264 fixed12へ`exp486_absolute_geometry_likpf`を1本だけ追加する。
- exp486のgeometry residual / log factor / ESS / resampling率をnative confidenceにする。
- fixed fallback 7候補、既存12候補、fold、objectives、LightGBM設定は変更しない。
- Residual版、HMM 50:50、新規pair/blend、候補置換は行わない。

## 検証方針

- outer 5 × inner 4、2 objectives、40 CPU selector boosters
- 親/control再学習、PF/HMM/Beam再実行、GPU、downstream TVT: 0
- parent fixed12とのpooled/fold/scope/by-well paired AND gate
- exp486 candidate利用率とincumbent rerankingはfreeze後に診断

## 所見

Absolute版には大きな平均改善がある一方、単独のwell-tail悪化とfixed13
selector系のreranking不安定性が強い。したがって本実験は高リスクなP3候補とし、
pooled改善だけでは昇格させず、固定tail AND gateを必須にする。

## 実行結果

- 40/40 CPU selector boosters、25 compact partitions、technical/leakage/score guard: PASS
- exp486 top1: `11.104974%`、positive usage folds: `5/5`
- parent fixed12比: pooled `-0.191174 ft`、改善fold `4/5`
- raw observed/missing、高missing、0--250、1000+、hidden-like 2面: 全7 scope PASS
- by-well: 416改善 / 357悪化、p95 `+1.109360 ft`、worst `14fee784 +9.361781 ft`
- decision: `FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`

平均・fold・固定scopeは良いが、事前固定したwell-tail上限`+0.25 ft`をp95とworstの
両方で超えた。same-OOF rescueは行わず、current-test生成、downstream TVT、
inference、submissionへ進めない。

## 実装済み

- steering requirements / design / tasklist
- 実行量とSHAを固定した`config.yaml`
- `candidate_contract.yaml` / `feature_contract.yaml` / `output_contract.md`
- 9章構成のcompact self-contained train候補（`.py` / `.ipynb`）
- fail-closed compact inference guard（`.py` / `.ipynb`）
- exp486 prediction / absolute ledger / freeze manifestのSHA・allowlist loader
- global key join、5 confidence、Stage A schema freeze、Stage C 40-model orchestration
- 7 scopeのscientific AND gate、H512 / whole-well oracle、reranking診断
- 専用契約test 10件

compact trainをcanonical train Notebookへ採用し、CPU strict packageを作成した。
canonical inference Notebookはplaceholderのまま維持し、推論・提出は行わない。

## 次

branchをterminal closeする。追加原因確認が必要な場合だけ、既存backlogの
`fixed13_selector_incumbent_reranking_instability_readout`へexp496保存scoreを加え、
0-booster・別承認の診断として扱う。
