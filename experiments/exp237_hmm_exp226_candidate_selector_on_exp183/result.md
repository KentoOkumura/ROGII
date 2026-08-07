# exp237_hmm_exp226_candidate_selector_on_exp183 結果

## 状態

Kaggle CPU train v1と、ユーザー承認済みraw-test inference v2が完了。global CV は支持したが、near / worst-well guard は不通過である。raw-test artifactの整合性は確認したものの、feature parity gapも残るため、Kaggle competition submitと次段ML化はしない。

## 目的

exp183 selectorへexp209 HMM blend、exp223 self-GR HMM、exp226 K16 geometryの3候補を加え、候補別の予測絶対誤差で選択する。row-wise選択は、exp183で支持された固定Viterbi continuity ruleでwell内を連続化する。

## 比較基準

- exp183 continuity: RMSE 10.601481774
- exp226 K16単体: RMSE 9.427109597
- exp072 likPF mean: RMSE 11.594897672

## 実行前ガード

- 11候補のID/well/row/target/last-known TVTを一対一照合する。
- candidate oracleの改善、unique-best rate、残差相関を学習前に確認する。
- `pf_z`は候補に入れない。
- 5 boostersのみを学習し、parent/controlは再学習しない。

## 結果

### 実行契約とコスト

- Kaggle kernel: kentookumura/exp237-hmm-exp226-candidate-selector-exp183-train v1
- CPU / internet off、3,783,989 rows / 773 wells、runtime 3,051.086 sec。
- candidate-error regressor 1 config x 5 GroupKFold = 5 boosters。parent/control の再学習なし。
- exp209 / exp223 / exp226 は各 3,783,989 rows / 773 wells で ID join coverage 100%、missing rows 0。

### 候補集合

- 11候補 oracle: RMSE 2.883510。既存8候補 oracle 4.564605 から -1.681096。
- 追加候補の単体RMSEは exp226 9.427110、exp209 blend 10.269697、exp223 self-GR HMM 11.349943。
- unique-best rate は exp223 22.27%、exp226 14.73%、exp209 7.47% で、追加候補に実用的なoracle headroomがある。

### Selector CV

| variant | RMSE | MAE | within10 | exp183比 | exp226単体比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| row-wise error ranker | 8.545228 | 4.998193 | 0.850156 | -2.056254 | -0.881882 |
| fixed Viterbi | 8.545093 | 4.991819 | 0.849475 | -2.056388 | -0.882016 |

fixed Viterbi は 7,640 switches、2.019 / 1,000 rows。row-wiseからの改善は -0.000134 に留まる。

### Guard readout

- exp183比の distance bucket は 050_100 / 100_250 / 250_500 / 500_1000 / 1000_plus で改善したが、000_050 は 0.508182 -> 0.675755（+0.167573）で悪化。
- exp115 hidden-like は spatial valid 12.593127 -> 8.637572、typewell-purged valid 12.479252 -> 8.598143 と改善。
- worst RMSE は 57.642328（fb03ae90）。exp183 worst 57.581365をわずかに上回る。well別の最大回帰は 70925e23 の 6.588074 -> 32.227080（+25.639006）。
- 主な回帰wellでは geometry / dense / PF path の誤選択が残る。fb03ae90では geometryとdenseが大半を占め、選択行のMAEは約60 ftだった。

### 判定

exp183とexp226単体に対するoverall CVは明確に改善し、11候補のcandidate coverage仮説は支持された。しかしnearとworst-well guardが不通過である。ユーザーがこのリスクを理解したうえでraw-test artifact生成を承認したため、fixed Viterbi一択・competition submitなしでinferenceを実行した。rank-slot add-only follow-upは実施しない。

### Raw-test inference artifact

- Kaggle inference v2はCPU / internet offで213.659 sec、14,151 rows / 3 wellsを完走した。competition submitは0件。
- `submission.csv`（`id,tvt`）は重複ID・欠損なし、finiteで、selected predictionと完全一致した。SHA256: `8e188be763a761965b9cfa1f3b26991b8093b447e5198ee01864209fb4d1c2a0`。
- HMM 2候補・multiobsはraw testで再生成した一方、exp109/114 OOF-only cluster/prior由来のlong feature 320本はraw-test median / all-missing時0で補完された。selectorは全14,151行で`pf_ancc`を選び、predicted errorは一定だった。
- よって本artifactは実行・ファイル整合性の確認用であり、提出候補ではない。

## 次アクション

新規候補を使う後続案は、well-riskを外側foldで推定してunsafe well / near-rowで候補集合を縮退できるかを、まずtrain-sideのみで検証する。safety guardとraw-test feature parityを両方満たすまで、hard selectorのcompetition submitやrank-slot ML化は再開しない。
