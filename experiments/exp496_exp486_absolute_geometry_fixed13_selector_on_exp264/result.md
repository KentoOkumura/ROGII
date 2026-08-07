# exp496_exp486_absolute_geometry_fixed13_selector_on_exp264 結果

## 現在の結論

Kaggle CPU version 1でStage A/Cを完了した。technical / leakage / selector score
guardはPASSしたが、well-tail gateをFAILしたため、
`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`でterminal closeする。

## 仮説

exp486 Absolute版の大きなpooled改善を、exp264 fixed12 selectorがtarget-free
confidenceから安全な局所だけ選ぶことで利用できるか検証する。

## 設定

- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- added candidate: `exp486_absolute_geometry_likpf`
- candidate source: exp486 Stage 1 frozen prediction
- validation: outer 5 × inner 4 nested selector
- execution: 1 variant / 2 objectives / 40/40 CPU selector boosters
- parent/control、PF/HMM/Beam、GPU、downstream TVT、inference、submission: 0

## 事前根拠

| prediction | RMSE | saved exp404比 | by-well p95 | worst |
| --- | ---: | ---: | ---: | ---: |
| exp486 Absolute | 9.726938029 | -1.187584044 ft | +10.069321492 ft | +44.021977054 ft |
| exp486 Residual | 11.139812021 | +0.225289948 ft | +4.795182565 ft | +32.921501347 ft |

Absoluteだけを13本目に追加する。Residual、HMM blend、新規pairは除外する。

## 判定契約

technical/leakage、selector score、candidate利用率、parent fixed12対比の
pooled/fold/scope/by-wellを全てAND判定する。tail上限はp95 / worstとも
`+0.25 ft`。1項目でもFAILならbranchを閉じ、same-OOF rescueを行わない。

## Kaggle Stage A/C結果

| 指標 | fixed13 | parent fixed12 | delta |
| --- | ---: | ---: | ---: |
| pooled hard RMSE | 8.461357622 | 8.652531956 | -0.191174334 ft |
| fixed fallback RMSE | 8.238331546 | 8.238331546 | 0.000000000 ft |

fold 0/1/2/4のdeltaはそれぞれ`-0.242410`、`-0.029654`、`-0.677940`、
`-0.104244 ft`。fold 3だけ`+0.106449 ft`悪化し、改善foldは`4/5`。

| 固定scope | delta fixed13 - parent |
| --- | ---: |
| raw GR observed | -0.188158 ft |
| raw GR missing | -0.198086 ft |
| missing fraction high | -0.193662 ft |
| distance 0--250 | -0.042128 ft |
| distance 1000+ | -0.218029 ft |
| hidden-like spatial | -0.370388 ft |
| hidden-like typewell-purged | -0.362970 ft |

全7 scopeは固定上限`+0.02 ft`をPASSした。exp486 top1率は`11.104974%`、
全5 foldsで正。selector score 3指標もpooled / 5 foldsすべてpriorを改善した。

一方、by-wellは416改善 / 357悪化で、delta p95は`+1.109359862 ft`、worst
`14fee784`は`+9.361781278 ft`。固定上限`+0.25 ft`を両方でFAILした。
pooled改善や全scope改善でtail FAILを救済しない。

## post-freeze診断

- H512 add-one oracle headroom: `0.097475 ft`、strict unique-best rows `10.2992%`
- whole-well oracle headroom: `0.066659 ft`、strict unique-best wells `74/773`
- exp486非top1行でincumbent choiceが変わる割合: `34.789662%`
- usage-delta Pearson / Spearman: `0.020819 / -0.000958`
- exp486利用0の38 wells: 24改善 / 14悪化

exp486自体に局所noveltyはあるが、利用率とwell deltaはほぼ無相関で、追加候補を
直接選ばない行の既存候補rerankingも大きい。診断は科学gate後に実施しており、
学習や判定の救済には使わない。

## 再現性

exp486 prediction / absolute ledger、exp226 geometry、exp263 cache、exp264 parent
scoreをSHA固定した。feature schema/content、40-model manifest、candidate score、
compact、gate、summary、Kaggle versionのSHAもversion 1生成物へ記録した。

## 実装確認

- compact train: 9章 / 647行。exp392 fixed13参照の8章 / 540行に対して、
  exp486二入力監査、reranking、feature importance、再現性summaryを追加した。
- compact inference: current-test候補、downstream TVT、submissionをfail closedにした。
- 専用test: `10 passed`
- 関連回帰test: `41 passed`
- Jupytext roundtrip、`py_compile`、`ruff`、strict experiment validation: PASS
- canonical train: compact候補を採用
- canonical inference: placeholder維持
- Kaggle package / completed version / trained boosters: `1 / 1 / 40`
- technical checks: `10/10 PASS`
- runtime: `3945.563 sec`
- model / compact / candidate-score SHA:
  `d4ac1528...36a977` / `7bacee14...6b2ac` / `07588f6d...5003e`
- 選択取得した小さいmetrics / manifestと完全logsを
  `kaggle/output/train_v1/`へ保存した。巨大score parquet / compact partitionsは取得していない。

## 次

same-OOFのweight / threshold / domain / feature / gate調整を行わずbranchを閉じる。
parent/control再学習、PF/HMM/Beam/GPU、downstream TVT、current-test候補生成、
inference、submissionは0のまま。原因確認が必要な場合だけ、既存のcross-branch
incumbent-reranking readoutへexp496を加える。
