# exp503 結果

## 状態

Kaggle private CPU version 3完了。technical PASS。固定fadeは小さく有効、
well-adaptive / masked-prefix replayは不採用。inference / submissionなし。

## 実行

- kernel: `kentookumura/exp503-exp490-strength-prefix-fade-readout-train`
- Kaggle id_no / version: `129477630` / `3`
- runtime: private CPU、internet off、`57.844874 sec`、peak RSS `1.549519 GiB`
- scope: 3,783,989 rows / 773 wells / 5 folds / target-free 32 features
- evaluation: 29 fixed fade profiles、outer 5 profile選択、prefix/context tree各5 fits、
  descriptive KMeans 1 fit
- LightGBM / control再学習 / new prediction / HMM / PF / Beam / GPU: 全て0
- inference / submission: 0 / 0

version 1は凍結特徴の`rows`列と再集計`rows`列がmerge時にsuffix化した実装不整合で停止。
科学条件を変えずrow数一致check後に重複列を落とし、version 2を完走した。version 3は
事前grid内のtau 85 / 500についてwell別tailを追加しただけで、grid、tree、outer選択、
truth-aware結果はversion 2と同一。

## 結論

exp490の本質は「普通のwellを少し改善するモデル」より、**exp357がwhole-well biasで
大きく外れた少数wellを救う平均回帰補正**だった。反対にexp357が既に正しいwellへ
逆向きの大きな補正を入れると、suffix全体に持続するcatastrophic biasになる。

- pooled: exp357 `9.737195157` → exp490 `8.480155260 ft`。
- well: 449改善 / 324悪化、median delta `-0.057105 ft`。
- positive gainの36.9231%を上位10 wellが占め、negative harmの57.8542%をworst 10
  wellsが占めた。平均値は少数の大改善・大悪化に強く支配される。
- 729-wellの主archetypeではRMSE改善は`-0.275596 ft`に留まり、32+3 wellsの
  strong archetypeが`-17.97 / -16.13 ft`、7+2 wellsのweak archetypeが
  `+21.54 / +45.24 ft`だった。

## 強いwell / 弱いwell

### 正解を使った最も明瞭な違い

| 指標（well median） | exp490が弱い324 wells | 強い449 wells |
| --- | ---: | ---: |
| exp357 RMSE | 3.084 | 6.035 |
| exp490 RMSE | 5.034 | 3.977 |
| correction-required alignment | -0.214 | 0.757 |
| truth-optimal direct alpha | -0.340 | 1.587 |
| 最長連続悪化rows | 1,353 | 619 |
| 累積gainが正のsuffix比率 | 0.264 | 0.830 |
| 累積gainが最後に非正となるrow | 4,771 | 1,241 |

correction alignmentとbenefitのSpearmanは`0.820838`、truth-optimal alphaは
`0.652327`だった。bottom alignment quintileは155/155 wellsが悪化してRMSE
`+3.408885 ft`、top quintileは98.06%が改善して`-9.579776 ft`。つまり、弱さは
posterior分散より「補正方向が正解方向か」の問題である。

exp357 RMSE quintileでも、下位4 quintilesはpooledでexp490が`+1.42`から`+4.02 ft`
悪化し、最難関top quintileだけ`-6.992134 ft`改善した。強い代表wellは
`8a3da6d1 -42.6903`、`b19b0395 -42.5201`、`4caa7289 -29.7540 ft`。
弱い代表wellは`389ae58f +49.6026`、`1b1eba53 +43.7719`、
`fb03ae90 +40.4397 ft`だった。軌跡図では、いずれも局所noiseではなく数千row続く
whole-well offsetとして現れた。

### suffix深度

| suffix rows | exp490 - exp357 RMSE |
| --- | ---: |
| 0--128 | +0.0483 ft |
| 128--256 | +0.0850 ft |
| 256--512 | +0.0685 ft |
| 512--1024 | -0.0507 ft |
| 1024--2048 | -0.5973 ft |
| 2048--4096 | -1.4825 ft |
| 4096+ | -1.9433 ft |

平均回帰はprefix直後512 rowsまでは逆効果で、約512--1024 rowsから効き始め、long
suffixほど強い。相対深度でもQ1 `-0.0161`に対してQ4 `-2.1037 ft`だった。
ただしfold 0は全depth帯で悪化し、4096+は`+2.6819 ft`。fold 3は同帯で
`-4.8350 ft`であり、fold間のwhole-well regime差が大きい。

### 観測可能なtarget-free signal

- 最強は`mean abs(exp357-exp226)`でbenefit Spearman `0.240706`、beneficial AUC
  `0.591912`、AUC範囲`0.564386--0.642165`、正方向5/5 folds。
- 同特徴の上位quintileはexp490が`-3.501481 ft`、下位3 quintilesは
  `+0.026--+0.103 ft`で、parentが別物理面と大きくずれるとexp490が効きやすい。
- `prefix_gr_sigma`は逆方向AUC `0.573077`。低sigma quintileほど改善しやすいが、
  quintile単位のRMSEは全て改善でありhard good/bad分離にはならない。
- posterior/state uncertaintyは単調でなく、`posterior_std_mean`のbenefit Spearman
  `0.048379`。HMM自身の分散はcatastrophic wrong directionを検出できない。

これはexp499のAUC `0.521151` / router RMSE `8.514311`と整合し、**難しいwellは
ある程度分かるが、間違った補正になるwellは事前に安全に分けられない**。

## 公開notebook型prefix処理

### 1. warm-up fadeは有効か

`pred = exp357 + alpha * (1-exp(-md_since/tau)) * (exp490-exp357)`を評価した。

| policy | RMSE | exp490比 | 改善fold | by-well p95 / worst（exp490比） |
| --- | ---: | ---: | ---: | ---: |
| 公開値 alpha=1, tau=85 | 8.479654 | -0.000502 | 5/5 | +0.000889 / +0.043500 |
| alpha=1, tau=500 | 8.447033 | -0.033123 | 5/5 | +0.080156 / +1.195616 |
| outer-selected strong fade | 8.098662 | -0.381493 | 4/5 | +2.644062 / +20.372558 |

公開値tau=85の効果は実質ゼロ。tau=500は5/5 foldsで小改善し、version 3のtail
characterization gateもPASSした。ただしtau=500は29 profile結果を見た後に選んだ
exploratory候補で独立validationではなく、exp490自体のexp357比tail
`+7.252786 / +49.017836 ft`はほぼ残る。**warm-upは弱い安全regularizerとして有効だが、
catastrophic wellの解決ではない。**

alphaを0.5--0.75へ縮めるstrong fadeはpooledを大きく改善するが、fold 3を
`7.928528 → 8.831610 ft`へ悪化させ、well tailも大きい。平均改善だけでは採用しない。

### 2. prefixから処理を変えるのは有効か

prefix-only treeは主にlog1p prefix GR sigmaとvisible-prefix長からalpha
`約0.45--0.75`を出し、outer-held pooledを`7.911445 ft`まで改善した。context treeも
`7.967391 ft`。ただし実態はgood/bad routingではなく全wellへの可変shrinkである。

- prefix tree: 380 wells改善 / 393悪化、fold 3は`+0.620514 ft`、p95 / worstは
  exp490比`+3.458136 / +20.766347 ft`。
- context tree: 384改善 / 389悪化、fold 3は`+0.502289 ft`、p95 / worstは
  `+2.889768 / +16.542903 ft`。

事前の平均・4/5-fold gateはPASSしたがtail-safeではなく、inferenceへ昇格しない。

さらに、early suffix truthで29 profilesからwell別に選ぶ楽観監査でも、128 / 256 rowsは
後半RMSEを`+0.161322 / +0.166882 ft`悪化。512 rowsでようやく`-0.021980 ft`だが、
early-late benefit Spearmanは`0.150234`、sign一致`55.57%`だけだった。正解を見ても
前半から後半を十分予測できないため、公開notebookのmasked-prefix backtestをexp490で
完全再生する追加HMM runは正当化されない。

## 再現性

- output: `kaggle/output/train_v3`
- exp490 raw / decompressed SHA: `99030b33...61b72c` / `e020e82e...e9a07`
- exp499 target-free feature SHA: `54c7e1da...7bb0d4`
- well readout SHA: `d87629a1...f8be2`
- depth metrics SHA: `5b030a19...ed9d`
- feature associations SHA: `48268a5b...fcc9`
- policy OOF SHA: `0458a783...fd90`
- fold metrics SHA: `e9b0109e...1f2c`
- model manifest SHA: `ae4d0950...37d8`
- summary SHA: `36c18e8c...1056`
- submission SHA: not applicable

## 判断

1. exp490の強みは、exp357のpersistent whole-well biasを大きく戻すこと。
2. 弱みは、既に良いparentへ逆向き補正を入れて別のpersistent biasを作ること。
3. target-free hard routerとprefix-adaptive alphaはtail-safeでなく、branchを閉じる。
4. tau=500固定fadeはexploratoryな小改善候補として保存するが、exp490 standalone LB
   `9.680`とcatastrophic tailを変えないため、inference / submissionは行わない。
5. early-truth transfer FAILにより、cutoff別masked-prefix HMM replayは実行しない。

## 参考資料

- [fle3n-rogii-v5](https://www.kaggle.com/code/fleongg/fle3n-rogii-v5):
  `1-exp(-md_since/tau)` warm-up dampingと公開設定tau=85の出典。
- [ROGII LB 7.201 Public GOLD — Conservative](https://www.kaggle.com/code/curvecowboy/rogii-lb7201-public-gold-conservative):
  既知prefixをmaskして候補をbacktestし、低alphaで保守的に適用する発想の出典。
- [Working Note: Our Solution & the Failures Behind It](https://www.kaggle.com/writeups/daulettoibazar/working-note-our-solution-the-failures-behind-it):
  whole-well bias、confidence/routingの難しさに関する外部解法メモ。exp503の証拠ではなく、
  ローカルOOF結果との独立した整合性確認にだけ使用した。
