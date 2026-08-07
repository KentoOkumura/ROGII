# exp244_bidirectional_prediction_start_pseudotail_augmentation 結果

## 仮説

early/original/lateのprediction-start multi-viewが、learned likelihood・ranker・small calibratorの
official-start long-tail頑健性を改善するか検証する。

## 設定

- 親: `exp239_distribution_matched_multicut_pseudotail`
- Route: `ml_model`
- 初期stage: deterministic manifest / prefix feature audit
- start offset: `-1000/-250/0/+250/+1000` rows
- 学習: 0 config / 0 fold / 0 booster
- シード: 42（乱数処理なし、SHA256 stable keyのみ）

## 結果

Kaggle CPU train audit v1は完了した。

| 項目 | 値 |
| --- | ---: |
| wells | 773 |
| views | 3,854 |
| early / original / late | 1,537 / 773 / 1,544 |
| materialized rows | 3,850,880 |
| feature columns（identifier/target込み） | 33 |
| late view share | 0.400623 |
| reported audit elapsed | 61.300 sec |

全leakage/materialization guardはpassした。official-start OOF、CV、LBは未取得。

current-test calibration audit v1は3 wells、6 requests、3,750 known-prefix rowsを生成した。
offsetは各well`-1000/-250`で、全rowがactual prediction start以前に収まった。unknown-tail TVT、
full-model fine-tune、submission predictionはすべて禁止されたままguardをpassした。

## 再現性

- well/view順はsortする。
- request IDとfold keyはimmutable keyからSHA256で作る。
- feature content SHAはcanonical CSV内容を対象とする。
- Kaggle kernel: `kentookumura/exp244-bidirectional-pseudotail-augmentation-train` v1。
- feature decompressed content SHA: `3ad67ca37800b28e6a77f8a25fbbf8167dbe60bfbaa7922ab19c03847706b444`。
- feature schema SHA: `19631f7c8e7a7cfcfbe36f698fa53e7ba0f2d1508cf328ee71fdeb74bf627d24`。
- inference kernel: `kentookumura/exp244-bidirectional-pseudotail-inference` v1。
- calibration feature content SHA: `6cce096380821bf7df2ab6ba0a22b11d31ff6808db57f810a21014e1e7370fdd`。
- calibration request manifest SHA: `37825f74fb2148740f4e169b5efbe8a1188ad7309270a92fe52c4c291f41c588`。
- rerun SHAは未取得のためdeterministic anchorとは扱わない。

## 解釈

view生成とleakage contractは成立した。late-startはtrain-onlyであり、current-test unknown tailへ
適用しない。これは実装・coverage監査のpassであり、予測性能の支持ではない。

## v2 parity preflight

exp218 frozen OOFを取得したローカル事前監査では、fold規則の再構成値が155 / 773 wells一致だった。
ただし等weight tie順序に環境差があるため、この値は採用せず、Kaggle v2でexp244 v1実fold manifestを
直接比較した。v1 foldはsource-well leakageを防いでいるが、exp218との差分を同じouter foldで評価する
用途には使えない。v2はraw official-tail row数で重み付けしたexp218互換GroupKFoldへ修正し、OOF identity、
raw official surface、model manifestをCPU 0 boosterでhard auditした。

Kaggle guard v2は全guardをpassした。exp218 frozen OOFは3,783,989 rows / 773 wells、
RMSE 8.475793978で記録値8.475793752と丸め範囲で一致し、decompressed SHAも一致した。
model manifestは15 boosters、fold 0..4、全model fileとSHAが一致した。raw trainのofficial-tail行数と
OOF first/last indexも全wellで完全一致した。

exp244 v1実fold manifestとの直接比較では174 wellsのみ一致し、599 wellsがexp218互換foldへ変更された。
v2 foldの評価行数は757,738 / 756,650 / 756,255 / 757,101 / 756,245で均衡している。

- kernel: `kentookumura/exp244-frozen-anchor-parity-guard` v2
- parity fold manifest SHA: `33694a894b4c616b8ac187d9cb752171c70815a5e607c3c27beccfb896883c0f`
- fold report SHA: `3b3ee689fe3a3324de0d1b7b8aebca896f1360d268440f519df40ecf0633ca3c`
- 学習 / 予測 / 提出: なし

### 解釈

exp218を凍結anchorとして使うためのidentityとofficial-start評価面は確立した。一方、v1で作ったfoldは
source-well leakageこそないが、exp218との差分評価には使えない。後続はv2 parity fold manifestへ固定する。

### v2時点の次

v2 fold上でsingle-variantのknown-prefix confidence-shrink meta-validationを実装する。exp218を再学習せず、
official-start OOF、1000+、hidden-like、worst-well、start間安定性を採用guardにする。

## v3 dual-start confidence-shrink meta-validation

`-1000/-250`のknown-prefix内で、各pseudo start以前128 rowsのTVT対MD local linear modelをfitし、
official startまでの既知区間でbacktest RMSEを測った。2 startの両方が悪いwellだけ、事前固定式で
exp218 residualを最大5% anchorへ縮めた。formulaはofficial-tail truthやfold結果からfitしていない。

| surface | raw | calibrated | delta |
| --- | ---: | ---: | ---: |
| overall | 8.475793978 | 8.477243182 | +0.001449204 |
| 000_050 | 0.957634003 | 0.957150028 | -0.000483975 |
| 050_100 | 1.310174744 | 1.308530058 | -0.001644686 |
| 100_250 | 2.094430711 | 2.099381820 | +0.004951108 |
| 250_500 | 3.315458688 | 3.321248307 | +0.005789619 |
| 500_1000 | 4.800746572 | 4.805099949 | +0.004353377 |
| 1000_plus | 9.295197870 | 9.296398072 | +0.001200202 |
| hidden-like spatial | 9.661607648 | 9.662210581 | +0.000602933 |
| hidden-like typewell-purged | 9.636010264 | 9.636612330 | +0.000602066 |

772 wellsでdual-startを計算できたが、実際にshrinkしたのは33 wells（4.269%）。alpha meanは
0.998726208、min 0.95だった。使用wellは15改善 / 18悪化、最大悪化は`a959858c` +0.925075。
改善foldは1 / 5だけだった。start RMSE gap medianは273.45 ftと大きく、`-1000`と`-250`の
local-linear proxyは安定していない。riskとwell deltaの相関も-0.018で、exp218 confidenceを識別しなかった。

### 判断

`adoption_supported=false`。near 100 rowsだけは微改善したが、primary overall、1000+、hidden-like 2面、
fold stabilityがすべて不合格なので不採用とする。threshold / max-shrink grid、current-test推論、提出は行わない。

- kernel: `kentookumura/exp244-dual-start-confidence-shrink` v1
- OOF decompressed SHA: `e55fec0351f2eaef727ffdb47ce5a296633ead2a152ea606017871564646c747`
- calibration feature SHA: `96dc811153e31af02b7778cd344852a05c1204073ac321e1a44f58dd09f1571a`
- 学習 / test prediction / submission: なし

### 次

simple local-linear proxyを使うtest-time calibrator枝は閉じる。再検討する場合は、保存済みexp218 boosterと
full feature stackをpseudo startで再生成し、まずoriginal-start prediction parityを通すreadoutに限定する。

## v4 early / original / late直接統合学習

v3は本来の位置変更学習ではなかったため、中心仮説を直接検証するsingle variantを実装した。
originalはexp239 official 380-feature cacheの全3,783,989行、early/lateはrawからstart別に
380特徴を再生成した3,081 views / 770,157 sampled rowsを使う。weightはofficial 1.0、pseudo 0.5。

| offset | kind | views | sampled rows |
| ---: | --- | ---: | ---: |
| -1000 | early | 764 | 191,000 |
| -250 | early | 773 | 193,250 |
| +250 | late / train-only | 773 | 193,157 |
| +1000 | late / train-only | 771 | 192,750 |

各viewは5距離帯から各50行、最大250行を決定的に選ぶ。outer-valid source well由来pseudo rowを
trainから除外し、validationはofficial-start全行だけとする。保存済みexp218 OOFをcontrolとして、
overall、1000+、hidden-like 2面、5 folds、by-well、worst-wellを同じsurfaceで比較する。

学習量は1 variant / 3 LightGBM configs / 5 folds / 15 boosters。parent/control再学習はない。
4本のCPU cache notebookと1本のGPU streaming train notebookを実装し、raw 773 wells上の期待件数、
Jupytext、ruff、py_compile、strict experiment/package validation、pytest 15件を通した。

### Kaggle GPU train v1結果

`kentookumura/exp244-bidirectional-multiview-train` v1は正常完了した。1 variant、3 configs、
5 folds、15 boostersを学習し、親/controlは再学習していない。validationはofficial-start
3,783,989行だけである。

| surface | raw exp218 | integrated | delta |
| --- | ---: | ---: | ---: |
| overall | 8.475794 | 8.472380 | -0.003414 |
| 000_050 | 0.957638 | 0.952745 | -0.004894 |
| 050_100 | 1.310177 | 1.304418 | -0.005759 |
| 100_250 | 2.094429 | 2.101476 | +0.007048 |
| 250_500 | 3.315459 | 3.358468 | +0.043009 |
| 500_1000 | 4.800747 | 4.863575 | +0.062828 |
| 1000_plus | 9.295198 | 9.286063 | -0.009135 |
| hidden-like spatial | 9.661607 | 9.245771 | -0.415836 |
| hidden-like typewell-purged | 9.636010 | 9.230900 | -0.405110 |

fold deltaは`-0.470272 / +0.909638 / -0.033011 / +0.132699 / -0.601773`で、改善は
3 / 5 foldsだった。overall、1000+、hidden-like 2面、minimum fold数はpassした。

by-wellでは387 wells改善、386 wells悪化と件数は拮抗したが、14 wellsが+2 ftを超えて悪化した。
最悪`059c8f24`は7.655552から24.306119へ+16.650567悪化し、worst-well guardだけがfailした。
上位悪化は`d90aa14c` +11.742932、`7987f2f2` +10.403484、`b37fd114` +9.339129だった。

### 判断

`adoption_supported=false`。集約OOFの改善は0.0034 ftと小さい一方、特定wellとfold 1の回帰が
非常に大きい。事前固定したworst-well +2 ft guardを緩和せず、不採用とする。inference、submission、
mixed pseudo weightの事後gridは行わない。

実装契約は成立している。official/pseudo cacheの380-feature schema、全manifest、outer-valid source-well
除外、15 model、OOF/artifact SHAはpassしており、Tracebackもない。したがって実装失敗ではなく、
uniform weight 0.5でearly/late shifted-start rowsを混ぜる学習目的が、hidden-like/deep surfaceを改善する一方、
一部wellのdecision boundaryを大きく崩した結果と解釈する。ただしearlyとlateを同時投入したため、どちらが
補償・悪化を生んだかはこの実験単独では識別できない。

### 再現性と生成物

- runtime: 17,201.149 sec、peak RSS 19,957.77 MB。
- feature schema SHA: `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`。
- model manifest SHA: `d93612c1c80d382f099892f08a34b4153b58554feb44cd17a69580256ccdb830`。
- OOF prediction decompressed SHA: `3c4600562f385b80be1de7279d4bd52fb3de6f2e6db0570ba339d7b0e422e98b`。
- metrics SHA: `384640c0002486830c00a3caae67dd9c2ee40d95ce5ae63c9a6b798c48592f80`。
- by-well SHA: `5b31141ea4fb415bd8b1536eb5dc2e374511eb431a4cb39500bb2a398f01cede`。
- full output archiveは取得せず、原因監査に必要なmetrics / by-well / summaryだけを
  `kaggle/output/integrated_train_v1/`へ取得した。

### 次

このbranchを再開する場合は、exp244と同じcache・sampling・weightでearly-onlyとlate-onlyを分離する
matched attributionを先に行う。lateが独立にhidden-like/deep改善を維持しworst-well guardも通る証拠が
出るまでは、再weight、risk gate、inferenceへ進まない。追加GPU学習は別途明示承認を必要とする。
