# exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264

## 状態

- Route: `ensemble`
- 状態: Kaggle CPU version 2 `COMPLETE` / scientific FAIL / terminal close
- CV: fixed13 hard `8.264890209`、parent fixed12比`-0.387641747 ft`
- LB / Submit: なし
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 追加候補親: `exp490_geometry_centered_mean_reverting_offset_hmm`

## 仮説と変更点

exp490 mean-reverting HMMの強い平均改善を、exp264 fixed12 dual selectorへ13番目の
score candidateとして追加し、安全な行だけ利用できるか検証した。

- 追加候補は`exp490_geometry_mean_reverting_hmm`だけ。
- exp490固有fieldはtarget-freeなresidual-state meanとposterior stdだけ。
- 7候補fixed fallback、2 objectives、outer 5 × inner 4、sampling、LightGBM、scope、gateは固定。
- exp498/499 featureは使わず、保存OOFを再利用してHMM/PF/Beamを再生成しない。

## 検証方針

- exp263/264 outer 5、各outer-train内inner 4、group=`well`
- 1 variant / 2 objectives / 40 CPU selector boosters
- exp490はglobal key join後にexp263 foldへ再partitionし、source foldやtruth/errorをfeatureにしない
- technical、selector score、candidate利用、pooled/fold、固定7 scope、by-well p95/worstを全AND判定
- tail FAIL時はsame-OOF rescueなしで閉じる

## 実行結果

- 40/40 CPU selector models、25 compact partitions、technical/leakage/score guard: PASS
- exp490 top1: `55.335335%`、positive usage folds: `5/5`
- parent fixed12比: pooled `-0.387642 ft`、改善fold `5/5`
- raw observed/missing、高missing、0--250、1000+、hidden-like 2面: 全7 scope改善
- by-well: 493改善 / 280悪化、p95 `+2.904594 ft`、worst `896d15b9 +18.394664 ft`
- decision: `FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`

平均・fold・固定scopeは強く改善したが、事前固定したwell-tail上限`+0.25 ft`をp95とworstの
両方で超えた。same-OOF rescueは行わず、current-test生成、downstream TVT、inference、
submissionへ進めない。

## 所見

exp490候補は行単位で高頻度に利用され、全fold・全固定scopeを改善する十分な信号を持つ。
それでもwell-tailは大きく悪化したため、candidate-long selectorの平均適合とwell単位安全性は
分離している。追加候補非top1行でも既存候補choiceが35.0%変わる点を、cross-fixed13診断の
negative evidenceとして残す。

## 実装

- `candidate_contract.yaml` / `feature_contract.yaml` / `output_contract.md`
- 9章構成のcompact self-contained train（`.py` / `.ipynb`）
- fail-closed compact inference guard（`.py` / `.ipynb`）
- `src/exp490_fixed13_candidate_cache.py`
- strict exp490 allowlist / SHA / global key / suffix-offset / truth-late validation
- Stage A schema freeze、Stage C 40-model orchestration、7-scope/tail AND gate
- H512 / whole-well oracle、incumbent reranking、feature importance、reproducibility summary
- 専用契約test 10件

compact trainをcanonical train Notebookへ採用し、CPU/private/offline packageを実行した。
canonical inference Notebookはplaceholderのまま維持する。

## Kaggle実行

- kernel: `kentookumura/exp501-exp490-fixed13-selector-train`
- valid run: version 2 / id_no `129379922` / `COMPLETE`
- runtime: `7082.113 sec`
- version 1: exp490 source種別誤りで入力解決前`ERROR`、trained booster 0
- selected output: `kaggle/output/train_v2_selected/`

大容量の49M行score parquetやcompact partitionsは取得せず、小型metrics/readoutとlogsだけを
保存した。

## 次

branchをterminal closeする。新規predictionを作らないcross-fixed13 reranking/tail原因readoutは
独立P4診断として検討できるが、exp501のgate再評価や救済には使わない。
