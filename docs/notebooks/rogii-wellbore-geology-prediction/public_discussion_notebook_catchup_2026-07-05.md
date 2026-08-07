# ROGII public notebook / discussion catch-up 2026-07-05

調査日: 2026-07-05

## Source

- Competition: `rogii-wellbore-geology-prediction`
- Discussion listing: `kaggle competitions topics list --sort-by recent/top`
- Notebook listing: `kaggle kernels list --sort-by dateRun/scoreAscending/voteCount -v`
- Pulled notebooks:
  - `docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent_20260705/`
- 主な本文確認:
  - Kaggle discussions `716289`, `716699`, `717445`, `717573`, `718088`, `718670`, `719235`, `719389`
  - Kaggle writeup pagesはHTML shellとmeta descriptionまでは取得できたが、本文はKaggle JS側でレンダリングされ、CLI/curlでは本文抽出できなかった。

## Executive Summary

1. 新しい公開notebookの多くは、6/27時点で整理済みの `7.159/7.2/7.295` 系 public fork の再実行、Gold/profile wrapper、または提出ラッパーだった。直接コピーしても新規性は小さい。
2. 新しい有用シグナルは、提出notebookよりも diagnostic / working note 側に多い。特に「heel/prefix GR calibration」「二峰性 datum の posterior mean」「自分の予測を測る ruler harness」「同一コード rerun variance」が再確認された。
3. `lucifer19/rogii-bedrock-forge-contact-reconstruction` の non-overlap contact override は、hidden test horizontal に `ANCC/EGFDU/...` がある前提のコードだった。このrepoでは formation列を train-only と扱うため、hidden-safe候補にはしない。
4. `yusuketogashi/rogii-lb7156-baseline` の差分は、public 7.159 source の末尾に `000d7d20` だけへ超小幅 anti-posterior move を入れる public micro-tune。一般化手法ではない。
5. TabM/ResNet notebook は Ravaghi/PF feature table上の深層tabular差し替えで、保存された出力スコアは確認できなかった。非タブラー単体が強いというdiscussion信号はあるが、既存の exp179/182/184 heatmap route の継続が優先。

## Highest-Value Ideas

### 1. Heel-calibrated GR likelihood landscape as confidence features

Signal:

- `pilkwang/working-note-target-free-tvt-geosteering` は、GR/typewell matchingを hard label ではなく likelihood landscape として扱うべきと整理している。
- `georgymamarin/fork-the-ruler-not-the-model` は、raw/global calibrationではなく heel/prefix calibration後に datum localizationが大きく改善する、という診断主張を置いている。
- Discussion `716289` でも pointwise GR ではなく constrained sequence/path likelihood が本筋という整理。

Repo status:

- 既に `exp133_gr_bimodal_match_ambiguity_detector` で direct midpoint / mode commit は壊れている。
- `exp161/166/172` 系に affine calibration / prefix crop feature はあるが、exp148 add-only global featureとしては弱かった。

Next experiment candidate:

- direct replacementではなく、`exp148` または exp158/176/191 selector系へ、heel-calibrated shift landscapeの `best_delta`, `second_delta`, `margin`, `entropy`, `zero_rank`, `calibration_residual_scale` を compact confidence featureとして追加する。
- 成功条件は global RMSEではなく、まず worst-well / longtail / GR ambiguity bucket の改善と、feature importanceでの利用確認。

Risk:

- exp133 の direct proxy failureを繰り返しやすい。必ず add-only / confidence-only から始める。

### 2. Two-mode posterior mean, but only after mode probability calibration

Signal:

- Georgy notebookは、二峰性wellでは hard mode commit より `p*a + (1-p)*b` の posterior mean が平方損失で自然と整理している。
- Discussion `719235` は `000d7d20` のような typewell mismatch例を再度問題にしている。
- `yusuketogashi` の 045 cell は逆符号の超小幅 posterior moveを `000d7d20` にだけ試しているが、public-only candidate CSV依存。

Repo status:

- exp133 は midpoint/averaging proxyが大破しており、単純な中点化は不採用。
- exp177/173 の beam topK posterior direct replacementも悪化済み。

Next experiment candidate:

- `p` の推定を先に評価する no-training auditを作る。候補は calibrated GR shift landscape、PF seed posterior dispersion、Beam topK gap、exp184 heatmap topK probability。
- posterior mean自体を提出値にしない。`p` / mode gap / expected-error を selector confidence featureに入れる。

Risk:

- 二峰性は「悪いwell flag」ではない。exp133でも ambiguous=1 bucketが単純に悪いとは限らなかった。

### 3. Ruler harness for current anchor diagnostics

Signal:

- Georgy notebookは `oracle_ceiling`, `tail_concentration`, `wall_test` を強調。
- Discussion `719389` では、CVが低くなるほどLB相関が崩れる、CV wellsの一部だけがLBと相関する、というコメントが出ている。
- Teardown discussion `718670` は、同一コードrerunでも public LBが 0.089-0.381 RMSE動く事例を整理している。

Repo status:

- exp086 / metric map / by-well readoutはあるが、現行 exp148/193/184/191 系の「どのwell群だけがpublicと相関しやすいか」の整理は弱い。

Next experiment candidate:

- 新モデル学習ではなく study として、現行 anchors (`exp148_cpu`, `exp193`, exp184/191 trainが完了したらそれら) の by-well residualを、tail concentration、prefix shift landscape、distance bucket、known-prefix shape、PF/Beam disagreement、public-LB-like subsetに分けて読む。
- public rerun varianceを考え、Public LB差 0.02-0.05 程度の候補は seed/rerun band内として扱う。

### 4. Non-tabular signal should continue through heatmap / sequence path features

Signal:

- Discussion `717573`: non-tabular single methodで 7.098、非タブラーCV 5台というコメント。ただし詳細は非公開。
- TabM/ResNet notebooksは深層tabularであり、raw GR sequence modelではない。Ravaghi/PF feature tableに対する model swapに見える。

Repo status:

- exp179/182で CNN/SDF/MTP heatmap のGR signalは確認済み。
- exp184は heatmap path features を selector featureとして支持したが、direct heatmap TVT replacementはしない方針。

Next experiment candidate:

- 新しいTabM/ResNet routeを切るより、exp184 compact heatmap selector features の raw-test parity / sparse coverage / fallback behaviorを優先。
- その後にやるなら、sequence modelは直接TVTではなく候補 path probability / uncertainty headとして使う。

### 5. Contact reconstruction claims are not hidden-safe as written

Signal:

- `lucifer19/rogii-bedrock-forge-contact-reconstruction` は public rebuildからGold overlayを削り、non-overlap wellsにprefix-anchored contact reconstructionを入れると主張。
- 実コードの non-overlap branchは `hw_te[ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA]` を読む。これらがtest horizontalにない場合は発火しない。

Repo status:

- このrepoでは公式説明に従い formation columns は train-only として扱う。
- exp150 / exp193 / exp176 / exp192/196 で formation/typewell priorは既に guarded feature / cache replacementとして検証中。

Decision:

- 直接採用しない。もし使うなら、hidden testにformation列が本当に存在することが公式/実行ログで確認できた場合だけ、同じexp内で raw-test schema guardを通す。

## Notebook Readout

| Ref | Local path | Readout |
| --- | --- | --- |
| `pilkwang/working-note-target-free-tvt-geosteering` | `date_run_recent_20260705/pilkwang__working-note-target-free-tvt-geosteering/` | 直接提出notebookではなくworking note。target-free geosteering、情報境界、public-aggressive / private-safe profile分離、GR likelihood landscapeが有用。 |
| `georgymamarin/fork-the-ruler-not-the-model` | `date_run_recent_20260705/georgymamarin__fork-the-ruler-not-the-model/` | 実装より診断harness。recoverable 3-5ft、worst 10%がSSE 40%、bimodal datum、posterior mean、seed bandの整理。 |
| `yusuketogashi/rogii-lb7156-baseline` | `date_run_recent_20260705/yusuketogashi__rogii-lb7156-baseline/` | public 7.159 lineage +末尾の `045` anti-posterior `000d7d20` micro-tune。一般化候補ではない。 |
| `lightningv08/rogii-lb-7-168` / `hujile/rogii-maybe-you-like` / `mattiaangeli/rogii-reproducible` | `date_run_recent_20260705/...` | 既存 public fork lineage。reproducibility / seed固定の価値はあるが、新手法は薄い。 |
| `lucifer19/rogii-bedrock-forge-contact-reconstruction` | `date_run_recent_20260705/lucifer19__rogii-bedrock-forge-contact-reconstruction/` | Gold no-op削除は実務的。non-overlap contact overrideはformation列前提でhidden-safeではない。 |
| `omidbaghchehsaraei/tabm-rogii-wellbore-geology` | `date_run_recent_20260705/omidbaghchehsaraei__tabm-rogii-wellbore-geology/` | TabM BatchEnsemble。LightGBMで上位110 features選択、GroupKFold 5、Ravaghi/PF features依存。出力スコアなし。 |
| `omidbaghchehsaraei/resnet-rogii-wellbore-geology` | `date_run_recent_20260705/omidbaghchehsaraei__resnet-rogii-wellbore-geology/` | Tabular ResNet。TabMと同じfeature table route。出力スコアなし。 |
| `busyaprime/persistence-is-the-geosteering-baseline-to-beat` | `date_run_recent_20260705/busyaprime__persistence-is-the-geosteering-baseline-to-beat/` | carry-forward baseline分布、dip extrapolationが危険、error grows like sqrt(distance)の整理。既存方針と一致。 |
| `paulodmayra/rogii-geologia-v92-geographic-restoration` | `date_run_recent_20260705/paulodmayra__rogii-geologia-v92-geographic-restoration/` | `kaggle kernels pull` で取得したコード本体が0行。評価対象外。 |

## Discussion Readout

| Topic | Signal |
| --- | --- |
| `717573` Score Without Tabular Models | 非タブラー単体 7.098、現best 6.798はtabularという報告。コメントで非タブラーCV 5台の話。詳細非公開なので、既存heatmap routeの継続材料。 |
| `719389` Does CV correlates with LB? | CV/LB相関は一部well群に限られる、低CV域では相関が崩れる、public LBは約50 wellsでnoisyというコメント。 |
| `718670` top-kernel teardown | 同一コードrerunでDWT 0.381、LB7295 0.149、pixiux/lightning 0.089 RMSE幅。public title scoreを保証値として扱わない。 |
| `717445` FOYSAL writeup thread | last-known anchor、GR/typewell alignment、ungated matching failure、guarded geosteeringのworking note紹介。既存方針と一致。 |
| `716289` Pointwise GR Makes No Sense | pointwise GRではなくsequence/path likelihood、prefixをanchor / calibration / local templateとして使う整理。 |
| `718088` Typewell and horizontal. How to match? | typewell matching featuresのablation推奨。既存exp群でも direct matchingよりfeature/confidence寄せが妥当。 |
| `719235` geologists' analysis | `000d7d20` のtypewell mismatch例。特定public wellへのmicro-tuneには注意。 |
| `716699` writeup submission thread | 新規writeupリンク集。curlでは本文取得不可だったが、meta descriptionからPF honest pipeline / unobservable datum系の追加調査候補はある。 |

## Recommended Next Actions

1. 直近は exp184 / exp191 / exp200 の進行を優先し、公開forkの再実行には寄らない。
2. 次に小さく切るなら、`heel_calibrated_shift_landscape_confidence_features_on_exp148` のような add-only confidence feature実験。ただし exp133のdirect failureを明記し、direct replacementは禁止。
3. その前に study として、現行 anchors の by-well residualを `oracle ceiling / tail concentration / public-like subset / seed band` で読み直すと、次の候補選別精度が上がる。
4. `lucifer` contact override、`yusuke` anti-posterior public micro-tune、Gold/profile wrapperは hidden-safe methodとしては採用しない。

