# 設計

## 仮説

exp389 Huber exact-HMMはGaussian control比でoverall `+0.085546 ft`、5/5 folds、
全required scopeを改善した一方、362 wellsで悪化してtail gateをFAILした。
corrected exp264 dual selectorへ候補とnative confidenceを追加すれば、Huberが有効な
局所だけを選び、単体tail悪化を回避してfixed12 hard selectorを改善できる可能性がある。

## アプローチ

exp388のStudent-t fixed13実装を構成参照にし、追加候補だけを
`student_t_exact_hmm`から`huber_exact_hmm`へ置換する。

1. exp263固定12候補cacheとexp389 target-free predictionをSHA固定で読む。
2. exp389は6列allowlistで読み、`well_id,row_idx` global key join後にexp263 foldへ
   repartitionする。
3. posterior std、well log-likelihood、loglik/rowをnative confidenceとして使う。
4. Stage Aでcandidate-long feature schemaを74→77列へrefreezeする。
5. Stage Cでouter 5 × inner 4 × 2 objectives = 40 CPU modelsを学習する。
6. selector scoreをfreeze後、保存済みexp264 fixed12 scoreと行単位で比較する。
7. pooled/fold/near/1000+/hidden-like/by-wellとHuber利用率を固定AND gateで判定する。
8. H512 / whole-well oracleはdiagnostic-onlyで記録する。

## 実験範囲

- 対象実験: `exp392_exp389_fixed13_dual_selector_on_exp264`
- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp389_exp209_huber_exact_hmm_emission`
- 構成参照: `exp388_exp374_fixed13_dual_selector_on_exp264`
- 変更する変数: primary candidate inventoryへ`huber_exact_hmm`を1本追加。
- 固定する変数:
  fixed12候補、fixed fallback 7候補、objective、fold、sampling、LightGBM、
  raw context、feature group、gate、threshold、weight。
- 実行量:
  1 variant / 2 objectives / outer 5 / inner 4 / 40 CPU boosters /
  parent control retraining 0 / GPU booster 0。

## 再現性設計

- seed policy:
  global seed 42。sampling keyはfold/objective/candidate等のimmutable keyから
  stable SHA256で作るexp388契約を維持する。
- stochastic処理:
  LightGBM row/column samplingとcandidate-long row sampling。
- PF/Beam:
  新規生成なし。保存済みexp389 predictionをSHA固定で読む。
- 並列処理:
  samplingをfit前に固定し、worker内global RNGを使わない。
- runtime:
  Kaggle private CPU、internet/GPU off、LightGBM
  `deterministic=true`、`force_col_wise=true`、`n_jobs=8`。
- input:
  exp389 raw gzip SHA
  `95302d547e8c49cdf67dabe6200e08e5c83f01ea158cf2fbd4f25b2fd1f74d75`、
  decompressed/logical SHA
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
  をhard checkする。
- output:
  feature schema、compact manifest、40-model manifest、candidate score、
  scientific gate、summaryのSHAを記録する。
- deterministic anchor:
  CPU deterministic flagsとSHAを記録するが、stochastic selectorであり
  submission未生成なのでdeterministic submission anchorとは呼ばない。
- Kaggle bootstrap:
  loose/package/bootstrap config、helper、contracts、metadata、3 kernel sourcesを
  push前に照合する。

## リスク

- leakage:
  exp389 truth/error/gate診断、parent exp264 score、oracleをfit前に読むこと。
  allowlistとpost-freeze境界で禁止する。
- fold:
  exp389はsource foldなし。source foldを合成せず、exp263 selector foldだけを使う。
- scientific:
  exp388では候補に補完性があってもhard selector RMSEが悪化した。Huberでも
  utilizationと選択精度が一致しない可能性がある。
- CV/LB:
  train-side selector改善はraw-test Huber候補の生成可能性を保証しない。
- runtime/memory:
  49,191,857 candidate-long score rowsと40 CPU modelsを扱う。exp388と同じ
  chunk/sample上限を固定する。
- multiple testing:
  weight、threshold、domain、candidate併用を同一OOFで救済しない。

## 次のアクション

compact train/inference、専用contract test、正規Notebookを実装・検証する。
push前に40 CPU boosters / control再学習0とpackage SHAを再確認し、承認済みscopeで
Kaggle private CPU Stage A/Cを実行する。downstream TVT、inference、submissionは
実行しない。
