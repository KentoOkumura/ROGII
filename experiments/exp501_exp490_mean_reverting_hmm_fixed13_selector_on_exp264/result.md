# exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264 結果

## 現在の結論

Kaggle CPU version 2でStage A/Cを完了した。technical / leakage / selector score
guard、pooled、5 folds、固定7 scopeはPASSしたが、by-well p95 / worstの両tail gateを
FAILしたため、`FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`でterminal closeする。

## 仮説

exp490 mean-reverting HMMをexp264 fixed12 dual selectorの13番目のscore candidateへ追加し、
target-free candidate-long scoreで安全な局所だけを選べれば、強い平均改善とwell-tail安全性を
両立できるか検証した。

## 設定

- Route: `ensemble`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 候補親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 追加候補: `exp490_geometry_mean_reverting_hmm`
- validation: outer 5 × inner 4 nested selector
- execution: 1 variant / 2 objectives / 40/40 CPU selector boosters
- parent/control、HMM/PF/Beam、GPU、downstream TVT、inference、submission: 0

## Kaggle Stage A/C結果

| 指標 | fixed13 | parent fixed12 | delta |
| --- | ---: | ---: | ---: |
| pooled hard RMSE | 8.264890209 | 8.652531956 | -0.387641747 ft |
| fixed fallback RMSE | 8.238331546 | 8.238331546 | 0.000000000 ft |

fold 0--4のdeltaは`-0.524825`、`-0.148336`、`-0.856345`、`-0.183048`、
`-0.213750 ft`で、全`5/5` foldsを改善した。

| 固定scope | delta fixed13 - parent |
| --- | ---: |
| raw GR observed | -0.344513 ft |
| raw GR missing | -0.485829 ft |
| missing fraction high | -0.598803 ft |
| distance 0--250 | -0.051308 ft |
| distance 1000+ | -0.438503 ft |
| hidden-like spatial | -0.664168 ft |
| hidden-like typewell-purged | -0.605512 ft |

固定7 scopeは全て`+0.02 ft`上限をPASSした。exp490 top1率は`55.335335%`、
全5 foldsで正。selector scoreのexpected-error MAE、within10 logloss、Brierもpooled / 5 foldsで
全てpriorを改善した。

一方、by-wellは493改善 / 280悪化で、delta p95は`+2.904593926 ft`、worst
`896d15b9`は`+18.394664149 ft`。固定上限`+0.25 ft`を両方でFAILした。
pooled・fold・scope改善でtail FAILを救済しない。

## technical / leakage

- input: 3,783,989 rows / 773 wells、exp490 raw gzip / decompressed SHA一致
- feature freeze前に読んだexp490列: key 3列、prediction 1列、native confidence 2列だけ
- forbidden truth/error/role/episode/fold/scope/gate列の読込: 0
- global key / suffix offset / exp263 selector fold repartition: PASS
- Stage A: 155候補featureから92列を固定、compact meta 77列
- Stage C: 40 models、25 partitions、18,919,945 compact rows、49,191,857 score-long rows
- outer-valid exclusion、inner well disjoint、inner OOF / four-model ensemble契約: PASS

## post-freeze診断

- H512 add-one oracle headroom: `0.272805 ft`、strict unique-best groups `2331/7787`
- whole-well add-one oracle headroom: `0.355756 ft`、strict unique-best wells `250/773`
- exp490非top1行でincumbent choiceが変わる割合: `35.007153%`
- usage-delta Pearson / Spearman: `-0.172649 / -0.203881`
- exp490利用0の2 wells: 1改善 / 1悪化

exp490自体の局所noveltyと直接利用率は大きいが、candidate bank追加に伴う既存候補rerankingを
含めてwell-tailを安全化できなかった。診断はprediction/gate freeze後に実施し、学習や判定の
救済には使っていない。

## 再現性 / 実装確認

- canonical train: compact候補と同一SHAで採用
- canonical inference: placeholder維持
- Kaggle kernel: `kentookumura/exp501-exp490-fixed13-selector-train` version 2、
  id_no `129379922`、`COMPLETE`
- runtime: `7082.113 sec`
- dedicated tests: `10 passed`、exp496込み回帰: `20 passed`
- Jupytext roundtrip、strict experiment validation: PASS
- feature / model / compact / score SHA:
  `2eb780b9...63e96e` / `3adb894d...cabbc` / `32317a71...9c257` /
  `1641b9cb...1599e`
- 小型metrics / readoutと完全logsだけを`kaggle/output/train_v2_selected/`へ保存した。
  49M行score parquetと25 compact partitionsを含む大容量output archiveは取得していない。

version 1はexp490 Notebook outputを誤ってdataset sourceへ指定し、入力解決前・booster 0で
`ERROR`となった。科学設定を変えずsource種別だけをkernel sourceへ訂正したversion 2が
上記の有効実行である。

## 次

same-OOFのweight / threshold / domain / feature / candidate subset / gate調整を行わずbranchを閉じる。
parent/control再学習、HMM/PF/Beam/GPU、downstream TVT、current-test候補生成、inference、
submissionは0のまま。後続は新規予測を作らないcross-fixed13 reranking/tail原因readoutを
既存P4候補として扱い、本expの救済には使わない。
