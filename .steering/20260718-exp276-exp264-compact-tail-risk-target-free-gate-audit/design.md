# 設計

## アプローチ

corrected exp264 Stage C v6の25 partitionをdownstream outer foldごとに読み、4つの`role=train` partitionから
well単位risk featureの分布を作る。同じfoldの`role=valid` partitionにはtrain分布だけで
empirical percentile rank、5 family等重みrisk、`q70/q80/q90` thresholdを適用する。

risk feature familyは次の5つに固定する。

1. selector score dispersion: predicted-error std、within10 std、candidate entropy。
2. candidate divergence: candidate value range/std。
3. top1-anchor distance: primary/fixed bankの2 objective top1とlast-known anchorの絶対差。
4. confidence coverage: 12候補に対するconfidence/availability不足数。
5. geometry/context: 評価長、X/Y/Zのanchor変位とstep、GRのanchor変位/step/missing率。

row系列は先頭128行、先頭512行、全評価区間でmean/p90/end等の事前固定集約を行う。
corrected Stage D v3 OOFのtarget、clean-273 control/compact-74 add-only predictionはrisk scoreと
thresholdを凍結した後だけjoinし、
bad-well rate/lift/recallとwell単位fallback gateをreadoutする。

## 実験範囲

- 対象実験: `exp276_exp264_compact_tail_risk_target_free_gate_audit`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: target-free well risk readoutとwell-level fallback auditだけ。
- 固定する変数: corrected exp264 Stage C v6 compact schema/25 partitions、Stage D v3 matched
  clean-273 control/compact-74 add-only OOF、outer 5 fold、候補bank、selector、TVT model、予測値。
- scope外: 学習、current-test port、inference、submission、quantile/weight/feature grid。

## fold契約

- downstream outer fold `f` のrisk fit wellsはStage Cの`downstream_outer_fold=f/role=train` 4 partitionだけ。
- risk valid wellsは`downstream_outer_fold=f/role=valid/source_outer_fold=f`だけ。
- train/valid well交差0、valid 773 wellsは全foldでexact-once、OOFとwell/fold一致をhard assertする。
- raw geometryはpartitionの`well_row_idx`でcompetition raw horizontal CSVへjoinし、`TVT`/`TVT_input`値は読み込まない。
- missing-tail boundaryはpartition keyを正とし、target missing maskから再決定しない。

## 判定

- 固定`q70/q80/q90`ごとに、`delta>0`と`delta>0.25`のrisk/safe rate、lift、recallを5 foldsで保存する。
- fallback gateはrisk wellだけStage D matched controlへ戻し、row RMSE、fold RMSE、改善保持率、by-well worst deltaを計算する。
- primary guardは3 quantileすべてについて、2 bad定義のrisk rate liftが5/5 folds正、gated RMSEが5/5 foldsでcontrolより良い、pooled改善保持率50%以上、pooled worst-well delta 0.25 ft以下を要求する。
- どれか1 thresholdだけの成功やfeature family別の成功ではguardを救済しない。

## 再現性設計

- seed policy: 乱数なし。well/row stable sortと決定的quantile/rankだけを使う。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging: 再実行なし。保存済みcompact値のみ読む。
- 並列処理と乱数の関係: 並列処理なし、`num_workers=1`。
- CPU/GPU runtime: Kaggle CPU、GPU off、internet off。
- train cache / test feature regeneration: Stage C manifest SHA、25 partition個別SHA、Stage D OOF SHA、raw input file manifest SHAを保存する。
- model manifest / prediction / submission: 新規modelなし。Stage D OOF prediction input SHAとrisk/gated prediction content SHAを保存し、submissionは生成しない。
- Kaggle package bootstrap: prepare後にconfig、source、metadata、bootstrap ZIP内support file SHAを照合する。

## リスク

- リークリスク: labelを見てfeature方向やthresholdを選ぶと事後gateになる。方向、family重み、scope、quantileをconfigで固定し、label joinを最後に分離する。
- CV/LB不一致リスク: current-test portを禁止し、train-side識別性だけを判断する。
- ランタイム/メモリリスク: 25 partitionを必要列だけ逐次読みし、well集約後に解放する。全18.9M行を連結しない。
- 再現性リスク: Kaggle kernel sourceのlatest outputが変わり得るため、manifestとpartition個別SHAをfail-closed照合する。

## Corrected-parent入力契約

- Stage C kernel: `kentookumura/exp264-exp263-confidence-dual-selector-train` version 6。
- Stage C compact manifest SHA: `f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c`。
- Stage C partition manifest SHA: `17930b7b50da7c783bffb8db8e34a0f69e5e583e028bde5b356d50a63bfacf66`。
- compact schema file/logical SHA: `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74` /
  `23614916c99edbbd513bcefee958d26cdfae5b83fb05c232c19736f2708dd725`。
- Stage D kernel: `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train` version 3。
- Stage D OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`。
- OOF rows / wells / worsened / over-0.25: `3,783,989 / 773 / 255 / 220`。
- 旧Stage C v3 / Stage D v2のSHAと結果は受理しない。
