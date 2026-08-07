# exp285_exp226_prefix_masked_offset_predictability_readout 結果

## 状態

Kaggle CPU version 2を完了した。technical guardは全PASS、scientific guardはFAILし、branchを閉じた。

## 固定仮説

known prefix末尾640行をmaskしてfold-safeに再生したexp226 geometry-only pathのoffset summaryが、
official evaluation suffixのexp226 residual median / slope / block driftをfold-stableに予測できる。

## 設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 検証: 保存5 folds、validation well全体をdonorから除外したprefix-masked backtest
- mask / visible minimum / blocks: 640 / 512 / 5 x 128 rows
- primary metric: pseudo offset median対official full-suffix offset medianのwell単位Spearman
- negative control: 256回fold内stable permutation
- route / booster: PF/Beam / 0
- truth attachment: pseudo geometryとprefix summary freeze後のみ

## 変更点

exp226のgeometry parametersと保存fold kappaは固定し、validation wellのknown prefix末尾だけをmaskして
pseudo replayする。GR/HMM/PFや補正は追加せず、prefix evidenceのpredictabilityだけを新しく測る。

## 結果

- eligible / ineligible: 766 / 7 wells、5 folds
- runtime: 77.492秒
- primary pooled Spearman: `-0.004135`（guard `>=0.30`）
- primary fold Spearman: `0.055222 / 0.036900 / -0.091083 / -0.033034 / -0.026692`
- positive folds / `rho>=0.20` folds: 2/5 / 0/5
- sign balanced accuracy: `0.488567`（guard `>=0.60`）
- 256 permutation p-value: `0.599222`（guard `<=0.01`）
- supporting slope / block drift Spearman: `-0.009074 / -0.013928`
- H256 / H512 / H640 Spearman: `0.186915 / 0.153020 / 0.131063`
- near / 1000+ Spearman: `0.189776 / -0.006022`
- hidden-like spatial / typewell-purged: `0.118237 / 0.119217`

全technical guardはPASSしたがprimary / supporting / scope guardはFAILした。判定は
`close_without_parameter_rescue`。

## 実装

- train: 1,902行 / 9章 / 20 cells。fold-safe exp226 donor field、saved fold kappa、well末尾までの
  pseudo geometry、path/prefix freeze、official target、correlation/permutation/scope/guardを含む。
- inference: 135行 / 4章 / 10 cellsでfail-closed。
- compact trainを正規trainへ採用した。正規inference notebookはtemplate stubを維持した。
- version 1はraw horizontalにない`id`列を要求して失敗した。科学入力ではない監査用`id`を
  `<well>:<row_idx>`から生成するよう修正し、専用testを追加してversion 2を完走した。
- 専用test 8件、repository test 219件、Jupytext sync、`py_compile`、ruff、strict validationをPASSした。
- correction、current-test、inference、submissionは未実施。

## 生成物

pseudo geometry、prefix/official summary、overall/fold/scope/permutation/by-well metrics、contract、input
manifestをKaggle outputへ保存した。ローカル監査先は
`/tmp/kaggle-output/exp285_exp226_prefix_masked_offset_predictability_readout/train_v2`。

- summary SHA: `e1760639...eda81`
- overall / fold / scope metrics SHA: `edbe41e4...fe23` / `4c22444b...bcb5` / `94006bf4...bd7f`
- prefix / official summary SHA: `02864c8b...caf9` / `400df63e...6ca2`
- target-free pseudo geometry logical / decompressed SHA: `070a44bf...c26e` / `1d7b2dc9...15b8`

## 再現性

- deterministic anchor: いいえ。fixed-input diagnosticとして扱う。
- real readoutはRNGなし、permutationだけstable SHA256 local RNG。
- input / pseudo path / prefix summary / official target summaryを段階別SHAで記録する。
- model / prediction / submissionは生成しない。

## 解釈

exp226の局所誤差にはH256/nearで弱い持続性があるが、H512/H640で減衰し、official full suffixでは
相関が0になった。fold符号も不安定でsign accuracyはchance相当、permutationでも有意でない。
したがってknown-prefix offsetをwell全体へ外挿する補正根拠にはならない。exp280の局所GR likelihood
separabilityは、長期offset persistenceやalways-on補正可能性を意味しないことが明確になった。

## 次

parameter rescueやprefix-calibrated correctionへ進まずbranchを閉じる。後続exp284も固定horizonの
self-GR incremental recovery / safety guardをFAILしたため、短距離の弱いsignalを救済根拠にしない。
exp281のblend/selector救済、current-test生成、推論、提出は行わず、新規backlogも追加しない。
