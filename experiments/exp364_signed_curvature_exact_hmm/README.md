# exp364_signed_curvature_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0完了・科学gate FAIL・fail-close
- CV / LB / Submit: なし
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 利用可否: 使用不可。Stage 1へ昇格しない

## 仮説

GR 尤度が固定した上向き・直進・下向き曲率軌道を識別できるなら、persistent な符号付き曲率状態で
exp209 のconstant-rate近似を緩められる。

## 実装

- 状態候補を`(position, rate, c)`、`c∈{-1,0,+1}`へ拡張する。
- 曲率driftは512行でexp209 rate grid 1 cell (`0.005`)相当の固定値。
- geometryや学習済みprefix regressorは使わず、exp209のterminal prefix rateとGaussian GR
  emissionだけで3 pathを採点する。
- candidate path / score / input / resource projectionをtruth前にfreezeする。
- 16-well projectionで3倍state tensorのruntime/RSSをhard gateにする。

## 検証方針

- complete 512-row blockをstride 256で作り、GR scoreのtop1 / MRRをzero-first rankと比較する。
- within-well circular-shift GRをnegative controlにする。
- 5 folds、1000+、hidden-like spatial、hidden-like typewell-purgedをAND gateで判定する。
- 16-well projectionのruntime `<=30600 sec`、peak RSS `<=25 GB`を同時に要求する。
- 全gate PASSと別承認がない限りStage 1へ進まない。

## Notebook

- `exp364_signed_curvature_exact_hmm_train.ipynb`: 採用済み正規Stage 0 Notebook
- `exp364_signed_curvature_exact_hmm_compact_selfcontained_train.py`: Jupytext percent形式の編集元
- `exp364_signed_curvature_exact_hmm_compact_selfcontained_train.ipynb`: 正規版と同一内容のcompact候補
- `exp364_signed_curvature_exact_hmm_compact_selfcontained_inference.ipynb`: fail-closed候補

## 結果

- Kaggle private CPU version 1、id_no `128529795`、`224.737080 sec`
- 773 wells中772 wells、13,631 complete blocksを評価
- technical gates: `12 / 12 PASS`
- top1 `0.550143`、MRR gain `0.252574`はPASS
- real-minus-circular top1 `0.003081 < 0.03`はFAIL
- passing folds `3 / 5 < 4 / 5`はFAIL
- 1000+ / hidden-like spatial / hidden-like typewell-purgedのRMSE方向は全PASS
- projected runtime `33857.604 > 30600 sec`はFAIL
- projected peak RSS `4.880433 < 25 GB`はPASS
- 最終判定: `STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`

## 注意

- Stage 0 exact HMM well-runsは0。Stage 1の773 runsは未実装・未実行。
- magnitude、transition、sigma、emission、adaptive noise、parallelism、blendで救済しない。
- inference / submissionは未実装・未実行。

## 所見

zero-first比のtop1 / MRRと3 stress scopeのRMSE方向は良かったが、realとcircular
controlの差がほぼなく、fold再現性も不足した。さらに状態数3倍の固定runtimeが上限を
超えるため、識別力とresourceの独立した2条件でStage 1昇格を棄却する。

## 次

このbranchは閉じる。別案を検討する場合も、exp364を再昇格させず、path-ranking用
negative-controlの検出力だけを独立したtruth-free 0-HMM readoutで監査する。
