# 設計

## アプローチ

修正版exp264の固定12候補candidate-long dual selectorをcontrolとし、保存済みexp490
`geometry_mean_reverting_hmm`をprimary domainの13番目に追加する。exp496のfixed13実装を
構成参照にして、selectorをouter 5 × inner 4で最初からcross-fitする。exp264保存scoreへ
exp490 scoreだけを後付けせず、13候補のscoreとhard choiceを同じnested split内で生成する。

これはexp499の「exp490かexp357かをwell単位で選ぶrouter」とは別仮説である。exp499の
32 well features、model selection、apply thresholdは持ち込まず、exp264既存のrowwise
candidate-relative / bank-context表現だけで評価する。

## 実験範囲

- 対象実験: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- Route: `ensemble`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 候補親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- implementation reference: `exp496_exp486_absolute_geometry_fixed13_selector_on_exp264`
- 変更する変数:
  - 全score候補数を12から13、primary domainを11から12へ変更する。
  - candidate ID `exp490_geometry_mean_reverting_hmm`を追加する。
  - exp490固有のtarget-free confidenceとして
    `geometry_mean_reverting_delta_mean`と`geometry_mean_reverting_hmm_std`を追加する。
- 固定する変数:
  - 親12候補の値、ID、順序、confidence、formula。
  - 7候補fixed fallbackとそのprediction/error。
  - exp264 raw-test-safe feature family、2 objectives、fold、sampling、LightGBM config。
  - hard selection score、tie、threshold、scope、tail gate。
  - downstream TVT、current-test、inference、submissionは無効。

## 入力契約

### exp490保存OOF

- kernel: `kentookumura/exp490-mean-revert-full-merge`, version 1
- file: `exp490_geometry_centered_mean_reverting_offset_hmm_stage1_full_oof_predictions.csv.gz`
- rows / wells: 3,783,989 / 773
- raw gzip SHA256:
  `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c`
- decompressed content SHA256:
  `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`
- feature freeze前allowlist:
  - `well`
  - `row_idx`
  - `suffix_offset`
  - `geometry_mean_reverting_hmm`
  - `geometry_mean_reverting_delta_mean`
  - `geometry_mean_reverting_hmm_std`
- feature freeze前禁止:
  - `fold`
  - `true_tvt_readout_only`
  - candidate / parent / exp226 error
  - truth由来role、episode、scope、by-well metrics、gate結果

`geometry_mean_reverting_delta_mean`は予測値のtarget-free内部状態、
`geometry_mean_reverting_hmm_std`はposterior uncertaintyとして扱う。true errorとの関係を見て
採否、変換、thresholdを変更しない。`tvt_geop`、exp357 parent prediction、exp226 predictionは
既存exp263/264候補bankとの重複を新しいnative confidenceとして追加しない。

### exp264 / exp263親

- exp263 Stage 0 manifest SHA:
  `85e60ac10b50197fa44ea29faffcbba81bd0746114bc53bae0f5cc537a26bb9e`
- exp263 catalog SHA:
  `7cd748661b719bfcfb1ed21b9fe314366b1c089cc0dd224a9e4cbd4ba7e9e6e0`
- exp264 corrected Stage C outer-valid score SHA:
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- parent fixed12 hard OOF RMSE: `8.652531955610227`
- fixed7 fallback OOF RMSE: `8.238331546485645`
- hidden-like assignment SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`

exp490は`well,row_idx`でexp263 global score-row keyへ1:1 joinし、coverageとsuffix offsetを
検証してからexp264 outer foldへpartitionする。exp490側foldはmodel featureにもsplit sourceにも
使わない。

## 候補・特徴契約

- score candidates: 13
- primary candidates: 12（親primary 11本 + exp490 1本）
- fixed fallback candidates: 7（変更なし）
- added candidate domain: primary only
- candidate ID encoding: declared-order one-hot 13
- ordinal candidate index: 禁止
- feature groups: `ctx`、`cand`、`conf`、`bank`、`formula`、`id`
- missing: native NaN + validity flag
- truth前に許可する機械的drop: all-missing、constant、exact duplicateだけ
- correlationはreport-onlyで、truth後のfeature selectionは禁止
- universal proxyは13候補へ再計算する。
  - availability / coverage
  - anchor distance / local shape
  - bank disagreement / rank context
  - family / kind one-hot
- exp498/499 feature、candidate catalog RMSE、true error、oracle、fold ID、well ID memorizationは
  使用しない。

## Validationと学習契約

1. Phase Aで入力SHA、3,783,989 keys、773 wells、13候補coverage、exp490 finite、
   fixed fallback parityを検証する。
2. feature schema/contentをtruth前にfreezeする。
3. outer 5の各outer-train内でinner 4を組み、2 objectivesを学習する。
4. outer-train compact scoreはinner OOFだけ、outer-valid compact scoreは4 inner modelの
   ensembleだけから作る。
5. 13候補hard primary prediction / choice / score / margin / entropyをfreezeする。
6. その後にだけtruth、親exp264 score、scope、hidden-like、by-well outcomeをjoinする。

実装・実行を別承認した場合の計算量は1 variant × 2 objectives × outer 5 × inner 4 =
40 CPU selector boosters、25 compact partitions、18,919,945 compact rows、
49,191,857 outer-valid candidate-long rowsで固定する。親control再学習、candidate HMM/PF/Beam
再生成、GPU、downstream TVTは0である。

## 事前固定gate

### Technical / leakage

- exp263 / exp264 / exp490 / hidden-likeのSHAが一致する。
- 3,783,989 rows、773 wells、13候補、40 models、25 partitions、全outer-valid coverageを満たす。
- exp490とexp263のglobal key / suffix offsetが1:1一致する。
- exp490追加候補値・2 confidence・全selector scoreがfiniteである。
- truth/error/oracle/fold/role/outcomeのfeature freeze前readが0件である。
- fixed7 fallback prediction/errorが親exp264と最大絶対差`0.0 ft`で一致する。

### Dual selector score

- `pred_abs_error`: outer-train candidate priorよりMAEをpooledかつ4/5 folds以上改善。
- `p_within10`: outer-train candidate priorよりloglossとBrierをpooledかつ各4/5 folds以上改善。

### Fixed13 integration

- exp490 primary top1 fraction `>= 0.005`。
- exp490 top1利用fold `>= 4/5`。
- fixed12 parent比pooled RMSE nonworse（delta `<= 0.0 ft`）。
- fixed12 parent比改善fold `>= 4/5`。
- 次の固定7 scopeをすべて`<= +0.02 ft`にする:
  raw GR observed / missing、high missing、distance 0--250 / 1000+、
  hidden-like spatial / typewell-purged。
- by-well delta RMSE p95 `<= +0.25 ft`。
- worst-well delta RMSE `<= +0.25 ft`。
- technical、leakage、dual score、integrationを全ANDでPASSする。

FAIL時は`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`として閉じ、同一OOFでの
threshold、weight、domain、feature、candidate subset、gate、hard/soft selection、downstream
TVTによる救済を行わない。PASSしてもinferenceやsubmissionへ自動昇格せず、次工程は別設計・
別承認とする。

## 診断専用readout

- h512 / whole-wellのadd-one oracle headroom
- exp490 top1 usageとwell delta
- exp490非top1行での既存12候補choice change率
- 既存choice不変行 / 変更行 / exp490 top1行のscore margin・entropy・RMSE
- exp490利用0 wellの改善/悪化数
- exp496との記述的比較

これらはpost-freezeかつreport-onlyであり、scientific gateや救済条件を変更しない。

## 再現性設計

- seed policy: exp264と同じ固定seed `42`。
- stochastic処理: CPU LightGBM学習のみ。候補HMMは保存済みで再生成しない。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。
- 並列処理: LightGBMは固定`n_jobs`、`deterministic=true`、`force_col_wise=true`。
- runtime: Kaggle private CPU、GPU off、internet off。
- input SHA: exp490 gzip raw/decompressed、exp263 manifest/catalog、exp264 score、hidden-likeを記録。
- feature/model/prediction SHA: feature schema/content、40-model manifest、compact partitions、
  outer-valid candidate score、hard prediction/choiceを記録する。
- model SHA: 実装・実行時に40 modelsを個別記録する。
- submission SHA: submissionを作らないため非該当。
- Kaggle bootstrap: package生成を別承認した場合にcanonical notebook、config、metadata、
  bootstrap内support filesの一致を確認する。
- deterministic anchor: design-only時点では主張しない。train OOF selectorであり、将来の
  current-test generation / submission再現性は別契約とする。

## リスク

- リークリスク: exp490 OOF自体はgroup-safeだが、fold、truth、error、role、outcomeを
  featureへ混ぜるとselector leakageになる。allowlist、global join、phase ledgerで防ぐ。
- rerankingリスク: 13番目を選ばない行でも再学習により既存12候補順位が変わる。exp496と同じ
  post-freeze choice-change readoutを必須にし、tail gateでfail-closeする。
- tailリスク: exp490単独のpooled改善は強いが324 wellsを悪化させ、p95/worstが大きい。
  pooled改善だけでは採用しない。
- 仮説重複: exp499のwell routerはFAIL済み。本実験はrow-local joint rankingへ限定し、
  exp499特徴やthresholdを持ち込まない。
- CV/LB不一致: train OOF PASSはcurrent test 3 wellsへの一般化やLB改善を保証しない。
- runtime/メモリ: 49.2M candidate-long score rowsを扱う。exp496と同じchunk / partition契約を
  固定し、実装前にKaggle CPU/RAM見積もりを再確認する。
