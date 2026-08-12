# 設計

## アプローチ

二段stageに分ける。

1. `feature_audit_only`: exp248 original-onlyのtrain long surfaceと、raw testから独立再構築したlong surfaceを比較する。297列すべてのprovenance/fallback/distributionを保存し、pass列だけのschemaとsample SHAを作る。モデル学習は0。
2. `train_after_feature_audit`: 同一実行内でstage 1を再実行し、audit pass後だけselected schemaをLightGBMへ渡す。within10 classifierとexpected-error regressorを同じouter well 5-foldで学習し、exp237固定Viterbi 1規則で評価する。

raw-test surfaceはexp073 base cache、raw horizontal/typewell、exp209 exact HMM source、exp223 self-GR HMM source、exp226 inference predictionから作る。multi-observationとcandidate-set contextはtrain/testそれぞれのraw horizontalから再計算する。

`raw_test_regenerated_copcf` variantでは、exp099 train pseudo-tail cacheをlabel source、exp065 assignmentsとtrain/test typewellをcluster source、exp114 train geometry summaryとraw-test horizontalをgeometry sourceとして使う。train側は既存exp109/114 OOF priorを維持し、raw test側だけfull-train参照priorを生成する。raw-test well IDはlabel sourceから除外し、typewell clusterはtest typewell GRのnative row-lag overlapから再割当する。

## 実験範囲

- 対象実験: `exp251_raw_test_safe_dual_objective_candidate_ranker`
- Route: `ensemble`
- 親実験: `exp248_candidate_perturbation_augmentation_for_likelihood_ranker`
- candidate親: `exp237_hmm_exp226_candidate_selector_on_exp183`
- 変更する変数: candidate-long feature schemaと`copcf_*`のraw-test再生成方法。`raw_test_safe` 130列は履歴controlとして固定し、新しい`raw_test_regenerated_copcf`だけを追加する。
- 固定する変数: 11候補、outer fold、2 objective、sampling cap、LightGBM設定、fixed Viterbi、candidate selectability、評価bucket。
- 比較control: exp248 original-only fixed Viterbi 8.421415097。control再学習なし。
- ML参照anchor: exp218 8.475793752。

## Feature contract

- 親long schemaは297列完全一致を必須にする。
- provenance allowlist:
  - `raw_test_base_cache`
  - `raw_test_hmm_regeneration`
  - `raw_test_multiobs_regeneration`
  - `raw_test_candidate_derivation`
  - `raw_test_candidate_set_context`
- `copcf_*`は`train_oof_neighbor_prior` / `raw_test_full_train_neighbor_prior`として許可し、同一schema、source exclusion、missing率、分布差を監査する。
- train-only exp226 auxiliaryとraw-test未生成列はrejectする。
- train/raw-test missing率は各5%以下、差は5%以下とする。all-missing列の0 fallbackは禁止する。
- SMD absolute 4超、PSI 0.5超はdistribution warningとして記録する。raw testが3 wellsのため、分布warningだけで列を自動除外せず、採否とwarningを分ける。

## 学習と採否

- 保存済み`raw_test_safe`は再学習しない。追加学習対象は`raw_test_regenerated_copcf` 1 variantだけ。
- within10 binary classifier 5 folds、expected-error L1 regressor 5 folds、合計10 CPU boosters。
- fixed Viterbi RMSEはexp218 8.475793752以下をprimary guardとし、exp248 original-onlyとの差を併記する。
- distance 1000+は9.234366423以下、exp115 spatialは8.958870705以下、typewell-purgedは8.909651004以下、worst-wellはexp248最大+0.25 ft以内をguardにする。
- 全guard通過前はinference/submitへ進まない。

## 再現性設計

- seed policy: audit/train row sampleはexperiment/fold/stage keyからSHA256-derived local RNGを作る。Python `hash()`とglobal RNGを使わない。
- stochastic処理: LightGBM histogram/subsample、保存済みupstream PF/Beam/HMM候補。新しいPF/Beam乱数生成やaugmentationはない。
- raw-test HMM: exp209/223の固定source/configを読み、raw testから再生成する。source SHAを記録する。
- raw-test prior: exp099 train pseudo-tail source、exp065 assignments、typewell mapping、exp114 geometryのSHA、除外したraw-test well ID、source well count、query wellごとのneighbor countを記録する。
- 並列処理: feature samplingはsingle local RNG。LightGBMはCPU設定をexp248から固定継承する。
- CPU/GPU: Kaggle CPU、GPU false、internet false。deterministic submission anchorとは扱わない。
- train cache / test regeneration: source file SHA、row/well count、selected schema SHA、train/rawtest sample decompressed content SHAを保存する。
- model/prediction: train stageだけ10 model SHA、model manifest SHA、OOF decompressed content SHAを保存する。submission SHAは対象外。
- Kaggle bootstrap: push前にcanonical metadataとbootstrap内configのstage/kernel sources/GPU/internetを再確認する。

## リスク

- リークリスク: train feature採否へtarget/error/oracleを使わず、raw testはtarget-free schema/distributionだけに使う。exp115 roleはmetrics限定。
- neighbor priorのリークリスク: raw-test well IDがexp099/exp114 train artifactに存在してもsource poolから除外する。raw-test queryには`TVT_input`既知prefixとMD/X/Y/Z/GR/typewellだけを使い、同一wellのtrain targetを使わない。
- CV/LB不一致: raw-test-safeでもtrain 773 wellsとcurrent test 3 wellsの分布差は残る。train guard通過だけでsubmission候補にしない。
- ランタイム/メモリ: feature auditはfull train surfaceを読み、20,000 base rows（220,000 long rows）を分布監査する。persist sampleは各50,000 long rowsへ制限する。
- 再現性: upstream HMM source/configとexp073/226 raw-test生成物が変わる可能性があるため、全source/schema/content SHAを毎回保存する。
