# exp492_huber_exact_hmm_full_replacement_on_exp264

## 状態

- ルート: `ensemble`
- 状態: `stage_c_completed_scientific_gate_failed_postreadout_error_closed`
- CV: `8.639368546`（保存exp264比`-0.013163410 ft`、3/5 folds改善）
- LB: 未提出
- 作成日: 2026-07-30
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 物理候補親: `exp389_exp209_huber_exact_hmm_emission`

## 仮説

exp389 Huber exact HMMのsaved OOFはGaussian exact HMMより`0.085546 ft`良い。
13本目として追加したexp392はparent fixed12 selectorより`0.117260 ft`悪化したが、
12本のsemantic slot置換ならcandidate count増加と13列目one-hotによる
incumbent rerankingの交絡を除いて評価できる。

## 変更点

- candidate ID、順序、domain、88列feature schemaはexp264と同一。
- `exact_hmm`の値とconfidenceをexp389 Huberへ置換。
- HMM依存の2 pairと固定3-wayを再計算。
- 4候補changed、8候補unchanged、総数12。
- GaussianとHuberは共存しない。

詳細は`candidate_contract.yaml`、`feature_contract.yaml`、
`.steering/20260730-exp492-huber-exact-hmm-full-replacement-on-exp264/`を参照する。

## 固定した評価量

- 1 variant / 2 objectives
- outer 5 x inner 4
- 40 CPU selector booster
- 保存exp264 control再学習0
- GPU / downstream TVT / inference / submission 0

## 検証方針

- exp264 corrected outer 5 / inner 4 group splitを固定する。
- saved parent hard RMSE `8.652531955610227`と比較する。
- technical、leakage、score、pooled/fold/scope/well-tailの固定AND gateで判定する。
- fixed fallback 7候補は置換影響の原因分解としてreport-onlyにする。

## 実行

別名の
`exp492_huber_exact_hmm_full_replacement_on_exp264_compact_selfcontained_train.py`
と対応ipynbを実装済み。fixed12 overlay、固定88/74 schema probe、strict nested
Stage C、saved exp264比較、feature importance、SHA記録をNotebook上で追える。

compact候補を正規train notebookへ採用し、Kaggle private CPU version 1
（`id_no=129217774`）で40/40 boosterを実行した。control再学習、GPU、
downstream、inference、submissionはすべて0。

## 結果

- hard primary: `8.652531956 -> 8.639368546`（`-0.013163410 ft`）
- improved folds: `3/5`（gateは4/5以上）
- by-well p95 delta: `+0.381470357 ft`（上限`+0.25`）
- worst-well delta: `+4.254514134 ft`（上限`+0.25`）
- decision: `FAIL_CLOSE_FIXED12_HUBER_REPLACEMENT_SELECTOR`

Stage Cと科学gate保存後、feature importanceの列名解釈ミスでNotebookはERRORに
なった。canonicalコードはlong-form `importance_type` / `importance` schemaに
修正済み。科学結論は回収済みで、追加40 boosterのrerunは未承認。

## 所見

候補数増加を除いても、Huber semantic replacementはpooled平均を小さく改善する一方、
fold一貫性とwell-tailを満たさなかった。exp392 fixed13ほどの平均悪化は消えたため
candidate-count交絡は存在したが、fixed12でもselectorのtail不安定性は残る。

## 次のアクション

branchを閉じ、同一OOFでのweight/threshold/domain/gate救済は行わない。
exp493の独立結果後、target-free continuous risk readoutを進めるか判断する。
