# exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission 結果

## 仮説

fold-safe な same-typewell horizontal GR atlas を confidence-gated auxiliary emission として exp209 exact HMM に加えると、常時補正を避けながら candidate state の識別と persistent offset の早期検知を改善できる。

## 実装

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- CV: seed 42 / well 5 folds。各validation wellのatlasは他のtraining-fold wellsのみで作る。
- atlas: `native_overlap / threshold=1` group、TVT 2ft bins、GR windows `64/128/256`、16 point patch。
- HMM差分: base emissionを固定し、state-centered/clipped peer scoreをtarget-free confidenceでgateする。v2 preflightは `alpha=0.025` の1本だけを実行する。
- LightGBM config / boosters: `0 / 0`。saved exp072 cacheをbaselineとして読む。inference/submitなし。

## 結果

Kaggle CPU train v1（`kentookumura/exp231-same-typewell-gr-atlas-hmm-train`）は timeout で未完了・評価不能となった。atlas構築は約292秒後に完了したが、43,206秒時点で `589 / 773` wells の開始までしか進まなかった。2 well並列の実測から、同じ設定の全well完走には約15.8時間を要し、Kaggle runtime上限を超える。

加えて、`[1/773]` から `[588/773]` までに開始された588 wellsのログでは、575 wells が `peer_atlas_confidence_mean=0.0`、残る13 wellsも `0.000046–0.014480` に留まった。先頭を含む多くのwellでは `alpha=0.01/0.025/0.05` のRMSE・std・loglikが同一で、alpha gridはほぼbase HMMの重複再実行になった。score自体のatlas coverageは約0.96-0.98あったため、peer coverage不足ではなく match-confidence のabsolute scaleが厳しすぎることが主因と推定する。v1は採否判断に使わない。

Kaggle CPU train v2 preflight は完了した。5 fold・773 wellsでatlasを構築したまま、HMM評価対象だけを固定12 wells、`alpha=0.025` 1本へ制限した。12/12 wells が成功し、出力は55,801 rows、全体wall timeは485.437秒、HMM生成は296.884秒、成功well当たり平均48.293秒だった。`peer_atlas_confidence_mean` は全12 wellで非zero（well平均の範囲 `0.022708–0.143981`、集計平均 `0.077299`）となり、v1の「ほぼ常にgate=0」は解消した。

v2ではdirect comparisonを意図的に無効化しているため、表示された対象12 wellのHMM RMSE `6.863590` はCVではなく、採否根拠にしない。今回確認できたのはfold-safe source policyを保ったgate発火と実行時間だけである。alpha 1本・全773 target wellsに線形換算するとCPU wall timeは約5.3時間で、Kaggle runtime枠内に収まる見込みである。

Kaggle CPU train v3 の正式full runは完走した。`hmm_peer_atlas_a025` 1 variant、seed 42の5 well folds、全773 target wells、3,783,989 evaluation rowsで、HMM生成は17,435.185秒、comparisonを含む全体は17,817.683秒だった。全773 wellsが成功し、atlas source summaryは各foldで `validation_in_source_count=0` であるため、同fold validation wellのTVTがatlas sourceへ入っていない。mean `peer_atlas_confidence` は `0.086781` で、v1のinactive gate問題は解消している。

saved exp072 `likpf_mean` 比では、overall RMSEは `11.594898 → 11.569950`（`-0.024947`）、MAEは `-0.460251`、within10は `+0.014837` と小幅改善した。しかし採用guardは満たさなかった。`1000_plus` は `12.702990 → 12.719560`（`+0.016570`）と悪化し、well単位では457改善 / 316悪化、最大悪化は `b19b0395` の `+48.316178 RMSE` だった。persistent-offset onsetは AUC `0.507654`、q90 lift `1.111111` でほぼ偶然水準であり、hidden-like subgroupはrole maskを得られず未評価（出力CSVは空）である。true-state rankは top1 `0.104124`、top5 `0.348195`、top10 `0.489973` だが、比較対象との差分を示せないため採用根拠にしない。

exp209の保存済みbest blend RMSE `10.269696` も更新しない。したがって、このpeer atlas auxiliary emissionはtrain-side不採用とし、alpha再grid、raw-test port、inference、submissionには進まない。

## 判定条件

nearだけの改善では採用しない。v2 preflightでunique TVT bin単位のmargin、relative-fit + soft absolute-novelty guardによるnonzero gateは確認できたが、v3で`1000_plus`・worst-well・offset-onset guardを満たさなかった。same-typewell GR atlasをHMM emissionへ直接弱加算する枝はここで閉じ、raw-test portを行わない。

## 次の扱い

このatlas emissionに対するalpha再gridや条件緩和は行わない。次のPF/Beam候補は、peer TVTをemissionへ直接加えない既存バックログからユーザーと選ぶ。
