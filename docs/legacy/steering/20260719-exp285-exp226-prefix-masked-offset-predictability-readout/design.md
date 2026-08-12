# 設計

## 結論

exp281のalways-on GR offset decoderを調整せず、test-timeで再現できるknown-prefix backtestから
exp226 geometry-only residualの持続性だけを読む。処理順を
`fold-safe donor field -> pseudo mask -> pseudo geometry freeze -> masked TVT_input attach -> prefix summary freeze
-> official truth attach -> predictability readout`に固定する0-booster diagnosticとする。

## 仮説

exp226 geometry-only pathの局所形状誤差にwell固有の低周波成分があるなら、known prefix内で
test-time同等に作ったpseudo suffix residual summaryは、official suffix residual summaryとfold-stableな
正相関を持つ。逆に相関がfold-stableでなければ、prefix-derived offset correctionへ進む根拠はない。

## 実験範囲

- 対象: `exp285_exp226_prefix_masked_offset_predictability_readout`
- Route: `pf_beam`
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- negative references: exp279、exp281
- supporting diagnostic: exp280
- 変更する変数: validation wellのofficial known prefixだけを固定pseudo cutで短縮し、exp226
  geometry-only pathをfold-safeに再生して、prefix内の観測offsetとofficial suffix residualの関係を読む。
- 固定する変数: exp226 fold、donor exclusion、saved kappa、K16 geometry parameters、mask 640、
  visible minimum 512、5 x 128 pseudo blocks、summary定義、scope、permutation、guard。
- 対象外: exp226 full `tvt_pred` / `gr_delta`、GR emission、HMM/PF、model fit、補正、candidate prediction、
  current test生成、inference、submission。

## Stage 0: fold-safe入力契約

exp226保存OOFから初期段階で読む列は`well_id / row_idx / suffix_offset / fold / tvt_geop`だけとし、
`tvt_true / tvt_pred / gr_delta / error / abs_error`をforbiddenにする。fold `f`のvalidation wellを
donor field、kappa fit、ANCC surface sampleからwell全体で除外する。donor fieldはfold外wellのraw truthを
使ってexp226と同じK16 raw/smoothed fieldを再構成するが、kappaは保存済みfold kappa 12 termsを読み、
同じreadout内で再fitしない。

raw validation target readerは`well_id / X / Y / Z / MD / TVT_input`だけを許し、GR、ANCC、TVTを渡さない。
target geometryはtest-time observableなのでcut以後も使用できる。fold/well/row input manifestを作り、
対象identityと全入力SHAを固定する。

## Stage 1: 固定pseudo maskとgeometry replay

wellの最後のknown rowを`s`とし、最後のcontiguous finite `TVT_input` block長が1,152以上なら
`c = s - 640`をpseudo cutとする。generatorへ渡すtarget viewでは`row > c`の`TVT_input`を全てNaNにし、
`c+1 ... end`をunknown suffixとして扱う。pseudo score zoneは`c+1 ... s`のちょうど640行である。

unknown horizonを640行で切らずwell末尾までにするのは、exp226のK=16 segment geometryがactual
test-timeと同じ「cutから残りtrajectory全体」で定義されるためである。validation targetのraw TVTや
masked TVT_inputを参照せず、fold外donor field、saved kappa、visible anchor `TVT_input[c]`、target
`X/Y/Z/MD`からgeometry-only `pseudo_tvt_geop`を生成する。

pseudo path tableにはidentity、fold、cut、row、geometry、donor distance、finite flagだけを保存し、
schema / logical content / raw file SHAを固定する。このfreezeが完了するまでmasked 640行のTVT_inputへ
再アクセスできないreader境界を実装時に設ける。

## Stage 2: masked-known offset summary

Stage 1 freeze後に、masked 640行についてのみ元の`TVT_input`をidentity joinする。これはreal testでも
known prefix内で観測可能な値であり、raw true TVTは使わない。

```text
r_i = TVT_input_i - pseudo_tvt_geop_i
```

640行を5 x 128行blockに固定し、各block residual medianとblock-center MDを求める。

```text
offset_median = median(r_1 ... r_640)
offset_slope = OLS(block_median ~ 1 + block_center_md)
block_drift_rate = (block_median_5 - block_median_1)
                   / (block_center_md_5 - block_center_md_1)
```

first/last block median、raw drift、MD span、finite coverageも保存する。clip、winsorize、demean、
fold calibrationは行わない。pseudo summary tableをwell 1行へ集約し、schema/content SHAとcontract SHAを
固定してからStage 3へ進む。

## Stage 3: official suffix target attachment

Stage 2 freeze後にだけ保存済みexp226 OOFの`tvt_true`を別readerで読み、official cut `s`以後の
residual `tvt_true - tvt_geop`を作る。official suffixは全rowを5個の連続blockへ`array_split`し、
pseudo側と同じmedian / slope / drift-rateを算出する。

primary targetはfull official suffix residual median。固定diagnosticとして先頭H256 / H512 / H640、
distance since official cut `0-250 ft`、`1000+ ft`のresidual medianも作る。hidden-like assignmentは
exp115保存artifactをpost-freeze readoutでjoinするだけで、pseudo feature生成やwell選択へ使わない。

## Stage 4: predictability readout

対応するpseudo / official summaryについてPearson、Spearman、符号accuracy、符号balanced accuracyを
well単位で計算する。primaryはpseudo offset median対full official offset medianである。fold、scope、
hidden-like、by-wellを固定出力する。

符号はsummaryが0以上ならpositive、0未満ならnegativeに固定する。negative controlは各fold内で
pseudo summaryのwell assignmentだけを256回置換する。各permutationでは5 foldsをそれぞれ置換してから
全eligible wellを再結合し、pooled Spearmanを1値計算する。seedは
`SHA256(exp_name, 42, fold, permutation_index, "prefix_summary_shuffle")`からlocal RNGを作る。
observed primary Spearman以上のnull値数`k`に対し`(k+1)/(256+1)`をp-valueとする。real pathとsummaryは
RNGなしであり、shuffleをwell選択やthreshold決定へ使わない。

## 固定guardと判断

- Technical: canonical identity、eligible >=750、mask/path/summary coverage 1.0、fold-safe exclusion、
  forbidden access 0、input/SHA一致。
- Primary: pooled Spearman >=0.30、5/5 folds positive、4/5 folds >=0.20、pooled sign balanced accuracy
  >=0.60、permutation p <=0.01。
- Supporting: slopeまたはdrift-rateの1 family以上がpooled Spearman >=0.20かつ4/5 folds positive。
- Scope: near、1000+、hidden-like spatial、hidden-like typewell-purgedのpooled primary Spearmanが全てpositive。

PASSしても現predictionを補正しない。別実験でbase保持、固定shrink、tail safety guardを含む
prefix-calibrated candidateを設計する許可だけを与える。

## 実行契約

- active readout variants: 1
- LightGBM / model config / trained fold / booster: 0 / 0 / 0 / 0
- HMM / PF regeneration: 0 / 0
- parent/control retraining: 0
- runtime: Kaggle CPU、GPU/TPU/internet off、single process、fold -> sorted well順
- inference/submission: disabled / disabled
- Kaggle push approval: 2026-07-19のユーザー依頼「実行してください」で承認済み。

## 再現性設計

- real donor replay / pseudo summary / readoutはRNGなし。256 permutationだけstable SHA256 local RNG。
- global RNG、Python `hash()`、thread-scheduling依存を禁止し、single processで順序を固定する。
- exp226 OOF decompressed SHA `709eb726...e4c609`、fold kappa SHA
  `6cbded4c...1aeff0`、hidden-like SHA `5f9ac9fa...ca6597`をhard guardする。
- raw train donor/target manifest、pseudo path、pseudo summary、official target summaryについてschema、
  logical content、file SHAを段階別に保存する。gzipはrawとdecompressed SHAを分け、decompressedを主証拠にする。
- model、final prediction、submissionは作らないため各SHAは対象外。scientific contract、input manifest、
  target-free pseudo path / summary SHAを代替証拠にする。
- deterministic prediction anchorではなく、fixed-input deterministic diagnosticとして扱う。
- Kaggle prepare時はloose config/sourceとbootstrap内config/source、CPU/offline metadataを照合する。

## 予定生成物

- `exp285_exp226_prefix_masked_offset_predictability_readout_contract.json`
- `exp285_exp226_prefix_masked_offset_predictability_readout_input_manifest.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_mask_manifest.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_target_free_pseudo_geop.csv.gz`
- `exp285_exp226_prefix_masked_offset_predictability_readout_prefix_offset_summary.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_official_target_summary.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_overall_metrics.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_fold_metrics.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_scope_metrics.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_permutation_metrics.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_by_well_metrics.csv`
- `exp285_exp226_prefix_masked_offset_predictability_readout_summary.json`

## リスク

- リークリスク: pseudo path前にmasked TVT_input、pseudo summary前にofficial true TVTを読むとoracleになる。
  reader/API境界と段階別SHAでfail-closedにする。
- validation donor leakage: validation wellのtruth/ANCC surfaceがfieldまたはkappaへ入るとgroup-safeでない。
  whole-well exclusionとsaved fold identityをhard guardする。
- geometry contract差: pseudo horizonを640で切るとK16 segmentationがtest-timeと変わる。well末尾までを
  unknownとして再生し、scoreだけ640行へ限定する。
- offset sign imbalance: raw sign accuracyだけでは過大評価するためbalanced accuracyとfold内permutationを併記する。
- multiple testing: cut / mask / block / summary / scope / guardは1契約に固定し、結果後の最良組合せを選ばない。
- CV/LB不一致: train-side predictability diagnosticであり、current testやLBについて結論しない。
- runtime/memory: 5 fold fieldを1つずつ保持し、validation wellごとにpseudo pathを生成してrow tableを逐次保存する。

## 実装反映（2026-07-19）

- 別名compact self-contained train source/notebookを9章・20 cellsで実装した。
- exp284からgeometry replayに必要な関数だけを抽出し、exp285ではpseudo cutからwell末尾までの
  K=16 segmentationへ変更した。summary対象だけを先頭640行へ固定する。
- target-safe readerはraw `X/Y/Z/MD/TVT_input`だけを読み、監査用`id`をwell名+row indexから生成し、
  GR/ANCC/TVTをvalidation targetから拒否する。
- pseudo geometry content SHA後にmasked prefixを接続し、prefix summary SHA後にだけofficial OOF
  `tvt_true`を読む2段freezeを実装した。
- Pearson/Spearman、sign/balanced sign、H256/512/640、near/1000+、hidden-like、256 stable
  permutations、technical/primary/supporting/scope guardを実装した。
- 別名inferenceは4章・10 cellsのfail-closed実装で、正規inference stub notebookは上書きしていない。
- compact trainを正規trainへ採用し、Kaggle CPU version 2を完走した。version 1はrawにない`id`列を
  要求したinput-schema failureで、科学契約を変えずreaderだけ修正した。
- 766 eligible wellsのprimary Spearman `-0.004135`、balanced sign `0.488567`、permutation p
  `0.599222`でprimary guardは全FAIL。supporting slope/driftと1000+も負相関で、branchを閉じる。
