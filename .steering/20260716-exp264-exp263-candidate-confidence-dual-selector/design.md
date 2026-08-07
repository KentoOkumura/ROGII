# 設計

> **旧設計無効:** 旧schemaはtraining-only formation raw/delta 12特徴を含む。既存OOFとcompactを
> 再利用しない。修正版Stage A version 4でraw-test-only 88列schemaを凍結した。

## アプローチ

```text
exp263 Stage 0 OOF / Stage 1 current-test candidate cache
  ├─ 6 primitive
  ├─ 5 fixed pair
  └─ 1 fixed exp226/w500
       + exp263 native confidence / universal proxy
       + reorganized exp251 v4 raw-test-safe context
                         ↓ candidate-long
             dual-objective LightGBM selector
       pred_abs_error[12] + p_within10[12]
                         ↓ 同一fold/process内で決定変換
               compact meta-feature
                         ↓ 条件付き・別承認
               downstream TVT LightGBM
```

## 実験範囲

- 対象実験: `exp264_exp263_candidate_confidence_dual_selector`
- Route: `ml_model`
- 候補親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- selector方法親: `exp251_raw_test_safe_dual_objective_candidate_ranker`
- downstream adapter参照: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: candidate bank、candidate confidence、candidate-long feature schema、dual scoreからのcompact adapter。
- 固定する変数: outer well 5 folds、seed 42、2 objectives、初回standard LightGBM、exp263 candidate値・formula・lineage、downstream TVT anchor residual target。

## 候補bank

score対象はexp263 Stage 1でcurrent-test生成・parity確認済みの12 surfaceとする。Stage 0 OOFにあるが現行Stage 1出力に未収録のsurfaceは、生成不可能とは扱わず、Stage 1拡張とparity確認後に追加候補として再評価する。

- primitive 6: `exp226_k16`, `selfgr_hmm_a070`, `likpf_mean`, `exact_hmm`, `pf_ancc`, `beam_mean`
- pair 5: `exp226_k16__selfgr_hmm_a070`, `exp226_k16__exact_hmm`, `exp226_k16__likpf_mean`, `selfgr_hmm_a070__likpf_mean`, `likpf_mean__exact_hmm`
- fixed 1: `exp226_w500_50_50`

12本は同一modelでscoreするが、一つのhard-selectable setにはしない。top1/top2/marginは次の2 domainで計算する。

1. `primitive_pair_bank`: primitive 6 + pair 5 = 11。
2. `primitive_fixed_bank`: primitive 6 + fixed 1 = 7。

これによりcandidate情報を落とさず、pairとnamed fixedを同時selectableにしないexp263のguardも維持する。`blend_likpf_hmm_w500`はpairのaliasなので候補数へ加えない。

## 信頼度

source-native confidenceは候補ごとに種類とcoverageが違う。全候補に1個の同名sigmaがあるとは仮定しない。

- primitive: exp263が保存したsigma、loglik/row、entropy、margin、support、ESS、fallback等の実在列を使う。
- pair/fixed formula: 親別namespaced confidence、valid数、component range/std、親方向一致、親disagreementを使う。
- 全候補共通: availability/coverage、anchor距離、local shape、bank median/range/stdとの差をtarget-free universal proxyとする。
- 共通の最終尺度: selectorがouter-foldで学習する`pred_abs_error`と`p_within10`。

Stage A採用100列から逆算したcurrent-test必須native namespaceは21列に固定する。exp226は
`PredictionResult.delta`、self-GR/exact HMMは同一callのstd/loglik/self-GR診断、PF/Beamはexp073
replayのstdを使う。likPFはnative scalarなしとしてvalid falseを明示する。exp264 inferenceはこの
完全なprimitive別mappingをcandidate contractに持ち、1列でも欠ける場合はfeatureのNaN補完前に停止する。

「候補別・outer-fold内で誤差へ校正」は別のsigma calibratorを置く意味ではない。candidate identity、native confidence、universal proxy、contextを入力したdual-objective selectorが、outer-valid labelを見ずに候補誤差とwithin10確率を学ぶことを指す。

## 特徴量

exp251 v4の295列はraw-test-safe context seedとして使うが、その列集合をそのまま継承しない。

- `ctx__`: candidate非依存のrow/well/typewell/spatial context。
- `cand__`: candidate TVT、anchor差、step、slope、curvature、straightness。
- `conf__`: native confidence、valid/missing。
- `bank__`: bank内位置、spread、median差、rolling disagreement。
- `formula__`: 親値・親confidence・validity・lineage。
- `id__`: candidate ID 12 one-hot、family/kind/formula one-hot。

実装上、exp251の295列は`retain/recompute/remove/defer`分類表へ展開する。`last_known_tvt`、
`eval_len`、`md_since`は`ctx__`へ移し、候補集合依存列はexp263から再計算する。`copcf_*`は
train cross-fitとcurrent-test再生成を同じ実装で保証できる入力がexp263 cacheに含まれないため、
Stage A/B初回ではdeferする。代わりにraw horizontal/typewellからtrain/current-test同一に生成する
row、geometry、log、typewell summary contextを使う。COPCFを追加する場合は同じexp264内の別feature
schemaとしてparity監査後に有効化し、既存schemaへ事後追加しない。

旧候補固有の`sc_ens/hyb/tvt_dense*`列、ordinal `candidate_index`、model/selector output、target-derived readoutは除去する。generic multi-observationやcandidate-relative特徴はexp263 bankへ再計算する。constant/all-missing/exact duplicateはStage Aで除去する。|Pearson|または|Spearman| 0.999以上は報告し、exact/functional duplicateでない限り初回に自動dropしない。最終列数はStage AのallowlistとSHAで固定し、それ以降同一実験内で変えない。

## 学習と出力

### Stage A: feature contract audit

- booster 0。
- exp263 manifest/catalog/formula SHA、12 candidate ID、row/well/coverage、formula parity、forbidden feature 0を確認する。
- 修正版Kaggle version 4は600,000 rowsを監査し、150列から全欠損41・定数5・完全重複16を除いた
  88列をlogical schema SHA `aaef4ffd...ddd3a4`で凍結した。採用側完全重複0、高相関14組report-only、
  sparse採用特徴25列、raw context availabilityはtrain 773/773・current-test 3/3 fileでPASS。
- feature catalog、confidence coverage、duplicate/correlation audit、schema SHAを保存する。

### Stage B: selector-only outer OOF

- 1 variant × 2 objectives × outer 5 = 10 CPU boosters。
- 2026-07-18に修正済88列selector surfaceで実行完了。control/親実験の再学習は0本。
- objectiveはL1 `candidate_abs_error`とbinary `candidate_within10`。
- exp262 extra-treesは結果に依存させず、standard LightGBMを初回固定する。
- OOF candidate-long dual scoreを監査用Parquetへ保存し、同時にcompact adapterを生成する。
- hard readoutはrowwise diagnosticのみ。Viterbiは実行しない。
- version 5はexpected-error MAE 3.795801、within10 logloss/Brier 0.359972/0.112451で、
  prior比pooled・5/5 folds改善によりscore guard PASS。hard top1は8.587004でfixed比+0.348673、
  0/5 folds改善のためFAIL。scoreはcompact内部表現に限定する。
- 45,407,868 candidate-long rows、3,783,989 compact rows、10/10 model SHAを監査済み。

### Stage C: nested compact meta

- 旧2026-07-17 runと承認はfeature availability leakageにより無効。
- 修正版88列runは2026-07-18に40 CPU boostersの実行承認を受領した。
- outer 5 × inner 4 × 2 objectives = 40 CPU selector boosters。
- outer-trainはinner OOF、outer-validはouter-train内inner ensembleからcompact featureを作る。
- inner splitは各outer-trainのwellを行数でbalancedな4 groupへdeterministicに割り当てる。
  outer-valid wellはinner assignmentとselector fitから完全に除外する。
- compactは`downstream_outer_fold / role / source_outer_fold`の25 partitionへ保存する。
  5 downstream fold分で18,919,945 base rows、outer-valid score auditは45,407,868 candidate-long rows。
- 40 nested model、25 partition、outer-valid score、fold manifestのSHAとcoverage/leakage auditを保存する。

旧Stage C Kaggle version 3は2026-07-17に完了した。40 model、25 partition、18,919,945 compact rows、
45,407,868 outer-valid candidate-long rowsを実測し、score guardとnested leakage auditはPASSした。
hard top1は8.420613でfixed 8.238332より+0.182281悪いため、Stage Cの合格範囲は後段add-only用の
連続score/compact生成に限定し、hard inferenceへは昇格しない。ただしこのrun自体は旧100列schema依存のため再利用しない。

修正版Stage C Kaggle version 6は2026-07-18に完了した。88列schemaで40 model、25 partition、
18,919,945 compact rows、45,407,868 outer-valid candidate-long rowsを再生成し、40/40 model byte SHAと
manifest coverageを監査した。expected-error MAE 3.798819、within10 logloss/Brier 0.359412/0.111830は
各5/5 foldsでpriorを改善し、score guardとnested leakage auditはPASS。hard top1は8.652532で
fixed比+0.414200、改善1/5 foldsのためFAIL。Stage Dへ渡せるのは74列continuous compactだけである。

### Stage D: downstream TVT ablation

- 旧full matched ablationの実行承認と結果はavailability leakageにより無効。修正版GPU学習は
  2026-07-18に30 boostersの明示承認を受領した。
- ユーザー選択によりexp218 380列から107列をdropしたclean 273列surfaceを固定する。
- 修正版はmatched control 273列 / selector compact add-only 347列。実行する場合は
  2 variants × 3 configs × 5 folds = 30 GPU boostersで、コストを再承認する。
- selector outputはadd-only。exp257型replacement-onlyは再開しない。
- 修正版Stage Cの25 compact partitionをfold契約の正とし、control/add-onlyを同一行・
  同一clean 273 surface・同一exp218 config・同一GPU modeで比較する。
- 学習前にStage Cのmetrics/model/compact/schema SHAと25 partition SHA、clean 273 allowlistの
  file SHA・列数・列順をfail-closedで検証する。
- 修正版Kaggle T4 version 3は30/30 boostersを完走した。control 10.476169、add-only 8.460811、
  5/5 folds改善、near / 1000+ / hidden-like非悪化を通過したが、worst-well +14.482873で
  事前guard FAIL。設計どおり推論・submissionへ昇格しない。

## 評価と昇格条件

- score品質: candidate別outer-train priorに対してexpected-error MAE、within10 logloss/Brierをpooledかつ4/5 folds以上で改善する。
- calibration: candidate、fold、distance、confidence-validity別に確認する。
- ranking: rank regret、top-k oracle coverage、2 legal domainのtop1/marginを確認する。
- hard readout diagnostic: fixed `exp226_w500_50_50` 8.238331からoverall 0.02以上改善、3/5 folds改善、near/1000+/hidden-likeは各+0.02以内、worst-well +0.25以内。
- hard readoutがfailしてもscore品質guardがpassすればnested compact add-only候補には残せる。Stage Dの実行承認は自動ではない。

## 再現性設計

- seed policy: LightGBM seed 42。candidate-long samplingはsorted stable keyのSHA256で事前決定する。
- stochastic処理: LightGBM row/column samplingとlong-row samplingのみ。candidate値とconfidence生成はexp263固定artifactを使う。
- PF/Beam/likelihood-PF/seed bagging: 本実験では再実行しない。exp263 source SHAを連鎖させる。
- 並列処理: worker内global RNGを使わず、sampling後に並列fitする。
- CPU/GPU: Stage A/B/CはCPU、Stage DのみGPU。修正版Stage B 10本、Stage C 40本、Stage D 30本は
  完走により承認scopeを消化済み。追加variant/config/fold、inference、submissionは未承認。
- feature SHA: gzipはdecompressed content SHA、Parquetはlogical content/schema SHAを記録する。
- model/prediction: candidate order、one-hot order、legal domain、missing counts、model SHA、candidate score SHA、compact schema SHAをmanifestへ保存する。
- Kaggle package bootstrap: internet off、必要package/versionとkernel metadataの一致をpush前に確認する。

## リスク

- リークリスク: nestedでないselector scoreをdownstream outer-trainへ入れると漏洩する。Stage C以外ではdownstream学習を禁止する。
- 選択バイアス: exp263 pair shortlistは同じfull OOF監査で事前固定されている。独立なcandidate discovery CVとは主張しない。
- 相関リスク: primitiveとpair/fixedは強相関でmarginを過大評価し得る。2 legal domainを分け、correlation auditを保存する。
- 欠損リスク: native confidence coverageはfamily別に異なる。NaN+validityを維持し、0/medianでparityを装わない。
- CV/LB不一致: exp251/255/257でoverallとworst-wellやLBが反転したため、globalだけで昇格しない。
- ランタイム/メモリ: 3,783,989 × 12 candidate-longを全量wide保存しない。chunk predictとParquet audit、in-memory compact化を使う。
- ランタイム/メモリ: confidence/formula列はdictへ集めて一度だけ`pd.concat`し、fragmented DataFrameをStage Bへ持ち込まない。
- 再現性: confidence拡張版exp263 Stage 1 current-test parityが未成立ならinferenceへ進まない。

## 2026-07-19 OOF診断とviewer出力設計

```text
corrected Stage C v6 nested_outer_valid_candidate_score
  -> primary legal-domain top1/margin diagnostic
corrected Stage D v3 stage_d_oof_predictions
  -> final add-only OOF overlay
  -> id,tvt viewer CSV
exp072 replay cache + raw train
  -> 500 particles x 128 stable-seed LikPF paths
  -> saved likpf_mean_d exact-parity diagnostic
```

- selector-confidence notebookはexp238版のall-well/typewell順、3段plot、plot manifest、plots zip、summary JSONを
  継承する。主panelはtrue TVT、exp264 final OOF、exp264 primary selector top1、LikPF、PF ANCC、Beam、
  exp226 K16、exp209 exact HMM、exact HMM ±2sigma、-Z min-maxだけをexp238と同じ順・色・線種で描く。
- 主top1帯は`primitive_pair_bank`の`pred_abs_error`とし、primary 11候補からcandidate codeを復元する。
  selector候補集合と結果はexp264固有差分として許容する。`primitive_fixed_bank`と両domainの`p_within10`は
  summary監査には保持してもplotへは出さず、exp238にない軸・線・panelを追加しない。
- Stage C v6の`nested_outer_valid_candidate_score.parquet`をKaggle output version 6から取得し、
  SHA256 `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`、
  45,407,868 candidate-long rowsとして入力契約へ固定する。outer-foldごとにinner 4 modelのscoreだけを使い、
  downstream Stage Dを生成したstrict nested surfaceと一致させる。
- LikPF notebookはexp238版のPF replay本体を変更せず、OOF resolver、Parquet列、表示名、summary SHA契約だけを
  exp264 corrected Stage D v3へ差し替える。モデル学習、candidate選択、blend、submitは行わない。
- viewer CSVは`selector_compact_addonly__lgb_mean__pred_tvt`を`tvt`へrenameし、ID順を保持する。
  input/output SHA、行数、unique ID、finite、再計算RMSE 8.460811237612477をmanifestへ保存する。

## 2026-07-18 旧hidden-safe inference設計（無効化済み）

以下はavailability audit前の履歴であり、selector 100列の直接train-only 12列と
downstream 380列の非fold-safe 107列が判明したため、保存済みmodelによる本構成の推論は禁止する。

```text
raw competition test
  -> exp263 source-port: 6 primitive + 5 pair + fixed 1 + 21 confidence
  -> candidate-long 100 features
  -> Stage C saved selector: outer foldごと inner 4 x 2 objectives
  -> outer fold別 compact 74 features（score CSVの保存・再読込なし）
  -> exp218 current-test base 380 features
  -> Stage D saved selector_compact_addonly: outer foldごと3 models
  -> 15 model equal-weight mean residual + last_known_tvt
  -> prediction artifact + submission.csv（未提出）
```

- selectorは40本すべてをSHA検証する。各outer foldで`pred_abs_error` 4本と`p_within10` 4本を平均する。
- Stage D manifest 30本のうちmatched control 15本は使わず、add-only 15本だけをSHA検証・推論する。
- base 380列とcompact 74列の順序は保存済みTVT modelのfeature nameを正とし、全15本で一致を確認する。
- exp145 learned likelihoodはexp263 replay frameからtarget-freeに再生成し、exp218 projection/GRWRもraw testから再計算する。
- runtimeはCPU、internet off、学習booster 0。GPU学習modelもLightGBM predictionはCPUで適用する。
- 公開test用に保存済みのcandidate/selector row artifactを入力に使わないため、hidden rerunでも同じsource-portを適用できる。
- `submission.csv`生成はcompetition submitを意味しない。`submit_to_kaggle=false`をconfigとsummaryに残す。

### 推論missingness guard

version 2はcandidate-long 100列へ一律`np.isfinite`を要求し、学習時からNaNを持つ29列を誤って拒否した。
Stage A `feature_catalog.csv`をSHA固定で推論packageへ同梱し、次の契約へ修正する。

- `conf__`/`formula__`の構造的NaNは学習時欠損率と一致する場合だけ保持する。0/median補完はしない。
- 学習時欠損率0の特徴へcurrent-testでNaNが発生した場合、または任意列に`±inf`があれば停止する。
- 学習時には値があった特徴がcurrent-testで全欠損になった場合も停止する。
- feature別・candidate別missing count/rateをCSV、metrics、reproducibility manifestへ保存する。
- exp218 base 380列とcompact 74列はStage D学習時からfinite必須だったため、既存finite guardを維持する。

## 2026-07-18 availability-first再設計

### selector

旧100列からformation raw/delta 12列を削除し、`MD/X/Y/Z/GR`だけをraw horizontal allowlistにする。
Stage A/B/C notebookは、trainとcurrent-testの全horizontal CSV headerを走査し、allowlist各列が両splitの
全fileに存在しなければfeature生成前に停止する。修正版列数はStage A version 4で88列、logical schema SHA
`aaef4ffd...ddd3a4`として確定した。

### downstream

exp218 380列はcurrent-test finite coverageを通しているが、fold-safe auditでは107列を失敗とした。

- 74列: full-train FormationPlaneKNN/DenseANCCImputerへ依存。
- 27列: exp111 fold0 target-trained scoreを全trainへ適用した非nested stacking。
- 6列: 上記へのGRWR推移依存。

既存matched control 15 model/OOFを比較anchorに使わない。ユーザー選択により、107列dropの
clean 273列surfaceを正式採用し、380列の「完全復旧」はスコープ外とする。allowlistは
`artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv`、file SHA256は
`d01a73cc28485345dd86ed56ad6276f1727dca6b270d87685e1cf578afb677bf`。

Stage Dではexp218の履歴的380列名を組み立てた後、allowlistのSHA・273列・重複0・列順完全一致を
fail-closedで検証し、許可された273列だけをモデルへ渡す。比較はmatched control 273列と
selector compact add-only 347列。修正済みStage Bの10 CPU boostersは完走済みで、
修正版Stage Cの40 CPU boostersも完走済み。Stage Dの30 GPU boostersとmatched control再学習は
  2026-07-18に承認され、version 3で完走した。clean 273/347列以外を拒否する契約とlocal rerun gateは維持する。

## 2026-07-19 修正版hidden-safe inference

```text
raw competition test
  -> exp263 source-port: 6 primitive + 5 pair + fixed 1 + 21 confidence
  -> candidate-long 88 features
  -> corrected Stage C v6: outer別 inner 4 x 2 objectives
  -> outer別 compact 74 features
  -> exp218 replayからclean 273 allowlistを選択
  -> corrected Stage D v3 add-only 347 features: outer別3 models
  -> 15 model equal-weight mean residual + last_known_tvt
  -> submission.csv
  -> local submit-check PASS後だけcompetition submit
```

- 新規学習0、保存済みselector 40本・TVT model 15本だけをSHA検証して使う。
- old Stage C v3 bundle、old Stage D v2 454列model、public-test row artifactは入力禁止。
- Stage D worst-well guard FAILは提出override後も保持し、LBは参考提出として記録する。
