# 設計

## アプローチ

各outer foldのtrain wellsで、固定deployable12をexp293 manifestのcandidate familyへ集約し、
familyごとのsuffix MAE/RMSE/best率をwell等重みで計算する。native Type Well群統計を
`k=10 wells`でglobal family統計へ縮約し、outer-validへType Well contentだけでjoinする。

prior tableとfallbackをouter-valid error結合前にSHA固定する。Stage 0ではheld-out wellの
family実績順位とのSpearman、fold安定性、hidden-like 2面、group-label shuffleとの差だけを
評価する。PASSと別承認時のみ、同じsoft priorをcorrected exp264 nested selectorへadd-onlyする。

実装ではexp293 v2の固定candidate memmap、block/fold identity、bank manifestをSHA
preflightし、exp065のType Well content由来`native_overlap/1` membershipだけをjoinする。
family errorはfamily内candidate等重みでwellごとのMAE/MSE/RMSEを作り、best-family tieは
固定family orderで解く。outer foldごとにtrain wellだけからglobal/group統計を作り、
`alpha=n_wells/(n_wells+10)`で縮約する。prior schedule SHAを固定した後だけheld-out
well-family errorをjoinする。

## 実験範囲

- 対象実験: `exp354_typewell_group_candidate_family_prior_readout`
- Route: `ml_model`
- 科学的親: `exp293_physics_only_candidate_bank_headroom_contract`
- downstream control: `exp264_exp263_candidate_confidence_dual_selector`
- 履歴参照: `exp316_typewell_group_candidate_family_error_prior`
- 変更する変数: Type Well群×candidate familyのsoft error priorだけ。
- 固定する変数: deployable12 candidate値/order/family、outer5/inner4、2 objectives、fallback、gate。
- Stage 0実行量: primary 1 + group-label shuffle 1 / 5 folds / model 0 / booster 0。
- Stage 1予約: 1 variant / 2 objectives / 5 outer / 4 inner = 40 selector models、control 0。
- Stage 0実装状態: compact self-contained train/inference候補とcontract testsを実装済み。
  2026-07-23のユーザー依頼によりtrain候補の正規Notebook採用とKaggle CPU実行を承認済み。
  inference候補の正規採用とStage 1は未承認。
- Stage 0実行結果: Kaggle CPU version 1でreal family rank Spearman `0.325789`、
  shuffle `0.327079`、差`-0.001290`となり、固定`>=0.05` gateをFAILした。
  `stage_0_failed_close_without_rescue`としてbranchを閉じ、Stage 1は不適格とする。

## 再現性設計

- seed policy: primaryはRNGなし。shuffleはfold keyとgroup content SHAからstable permutationを作る。
- stochastic 処理の有無: fixed negative-control permutationだけ。global RNGを使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存候補を読むだけで再生成しない。
- 並列処理と乱数の関係: stable permutationを先にmaterializeし、集計順とthread数を固定する。
- CPU/GPU runtime と deterministic flags: Stage 0 Kaggle CPU、GPU/internet off、上限30分。
  Stage 1は別承認時のみselector runtimeに合わせる。
- train cache / test feature regeneration の SHA 記録方針: exp293 candidate/family manifest、
  fold、prior input/output、fallback reason、Stage 0 readoutのschema/content SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: Stage 0非該当。Stage 1では40 selector modelとOOF SHAを保存する。
- Kaggle package bootstrap 確認方針: package承認後にcanonical config、candidate manifest、
  bootstrap内configのSHA一致を確認する。

## リスク

- リークリスク: outer-valid errorをprior fitやfamily/threshold選択へ使わない。prior manifestを先に凍結する。
- CV/LB 不一致リスク: Type Well群のtest coverage差をglobal/neutral fallbackとhidden-like readoutで監査する。
- ランタイム/メモリリスク: Stage 0は0-modelで低い。Stage 1は40 modelsなので別承認を必須とする。
- 再現性リスク: exp293 candidate family/orderとexp264 fold版の混同をSHA hard preflightで防ぐ。
- 解釈リスク: exp256でcandidate-family base rate寄与が大きかったため、group固有差はshuffle差を必須とする。
- 救済リスク: support、group definition、family、shrinkage、Spearman gateを同じreadout後に変更しない。
