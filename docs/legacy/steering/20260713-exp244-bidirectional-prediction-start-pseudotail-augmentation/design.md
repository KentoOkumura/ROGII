# 設計

## 仮説

prediction startの前後viewを同一source-well foldで学習候補化すると、early long-tailだけの
augmentationよりもre-anchor後回復を含むprefix条件の幅が広がり、official-startで使う
learned likelihood、ranker、small calibratorの頑健性を改善できる可能性がある。

## アプローチ

exp239のraw-train official-start metadata、source-well fold、replay requestという契約を
引き継ぐ。ただしcutoff候補のdistribution matchingは変更せず、初版は原因分離しやすい
固定row offsetだけを使う。各wellについてofficial cutoffを`0`とし、負offsetをearly、`0`を
original、正offsetをlateとして扱う。prefix長と残tail長guardを満たすviewだけを生成する。

各viewはraw horizontal wellから`TVT_input`を作り直す。全行を欠損化した後、start以前だけ
train true TVTで埋める。feature builderへtail true TVTを渡さず、初版ではanchor、prefix GR、
trajectory、距離統計をmaterializeする。PF/Beam、learned likelihood、GRWR、PF/HMM初期状態は
同じrequestから後続段階で再生成する契約とし、生成済みfull-prefix cacheのsliceは禁止する。

late viewはtrain-only augmentationである。current-test向けrolling calibrationはactual start
以前のknown prefixだけを使う別request kindとしてmanifest schemaを用意するが、初版では
calibrator学習やtest-time fine-tuneを行わない。

## 実験範囲

- 対象実験: `exp244_bidirectional_prediction_start_pseudotail_augmentation`
- Route: `ml_model`
- 親実験: `exp239_distribution_matched_multicut_pseudotail`
- 比較親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: prediction startの方向とofficial startからの固定row offset
- 固定する変数: source-well GroupKFold、official-start validation、target、metric、親model/control、feature family contract
- 初期grid: early `-1000/-250`、original `0`、late `+250/+1000` rows。prefix/tail guardで不成立viewはskipする。
- cap: wellあたり最大5 views、materializeはviewあたり最大1,000 rows、start kind別件数とrow比率を記録する。
- 学習量: active audit 1、LightGBM config 0、fold学習0、booster 0、親/control再学習なし。

## 再現性設計

- seed policy: view IDとfold keyはexperiment、source well、official cutoff、start offsetからSHA256で生成する。
- stochastic 処理の有無: 初版のview生成、fold、sampling、feature materializationは乱数を使わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 初版は実行しない。後続ではsource well、view ID、familyからstable seedを作る。
- 並列処理と乱数の関係: well/view順をsortし、global RNGとthread schedulingへ依存しない。
- CPU/GPU runtime と deterministic flags: 初版はKaggle CPU audit想定、GPU学習なし。
- train cache / test feature regeneration の SHA 記録方針: manifest、schema、materialized featureをcanonical CSV contentでhashする。train-only lateとtest-compatible requestを別集計する。
- model manifest / prediction / submission SHA 記録方針: 初版はmodel/prediction/submissionなし。後続学習時に追加する。
- Kaggle package bootstrap 確認方針: prepare後にbootstrap内configのoffset、cap、seed、route、実行stageを正本と照合する。

## リスク

- リークリスク: late prefixへ追加したtrue TVTをvalidation/testへ流す危険。`target_usage=train_only`、source-fold assertion、outer-valid exclusion contractで防ぐ。
- CV/LB 不一致リスク: late short-tailだけが改善しofficial long-tailへ転移しない。採用主指標をofficial-start OOFとし、start kind別結果は補助に限定する。
- ランタイム/メモリリスク: 5 views/wellでPF/Beam再生成が膨らむ。初版はanchor/prefix統計と1,000 rows/view capだけをmaterializeする。
- 再現性リスク: view順、同点、CSV gzip metadata差。stable sort、SHA256 key、decompressed/canonical content SHAで防ぐ。

## 次

Kaggle CPU auditでmanifest、view比率、materialization coverage、content SHAを確認し、guard通過後に
learned likelihood / ranker / small calibratorのどれを最初のdownstreamにするか決める。

## v2 frozen-anchor parity preflight

exp218 train v1 のOOFは3,783,989 official-tail rowsを対象に、row-levelのgroupsを渡した
`GroupKFold(n_splits=5)`で作られている。一方、exp244 v1は773 wellsを1行ずつ並べた
GroupKFoldでfoldを作っていた。どちらもsource-well分離は守るが、fold attributionは同一ではない。

v2はexp218 OOFのwell別行数とraw trainのofficial-tail行数が一致することを確認し、sklearnの
non-shuffled GroupKFoldと同じ「大きいgroupから最軽量foldへ割当」の規則でexp218 foldを再構成する。
v1 foldとの一致率、変更well数、fold別well数・行数を保存する。これにより、後続calibratorの
official-start OOFをexp218凍結OOFと同じouter fold上で評価できるようにする。

preflightではexp218 modelを再学習・再推論しない。保存済みOOFとmanifestのidentity、raw official
surface、fold attributionだけを監査し、LightGBM 0 config / 0 boosterを維持する。

## v3 dual-start confidence-shrink meta-validation

exp218 OOFを使ってouter-train wellsのoptimal alphaを学ぶ通常のmeta modelは採用しない。保存済みOOFは
各well自身にはfold外だが、meta outer-train側のOOF modelがouter-valid wellを学習に含み得るため、
厳密な二重OOF attributionにならない。この段階では性能より原因分離を優先し、target fittingのない
事前固定式を使う。

各wellのofficial startから`-1000/-250` rowsへ戻り、そのstart以前の最後128 rowsだけでTVT対MDの
local linear extrapolatorをfitする。pseudo startからofficial startまでの区間はactual prediction時には
既知prefixなので、ここでRMSEを測ることはcurrent-test compatibleである。2 startのRMSEの小さい方を
conservativeな一致riskとし、10 ft以下はalpha=1、10〜30 ftをlinear ramp、30 ft以上はalpha=0.95とする。
片方だけ悪いstartでは縮めないため、start不安定性による誤発火を抑える。

official tailでは保存済みexp218予測のanchor residualだけを`alpha`倍する。alphaはofficial-tail truth、
他well、fold集計から一切fitしない。v2 parity foldはfold別readoutにのみ使い、formula選択には使わない。
exp115 hidden-like 200-well 2面を固定stress subgroupとしてjoinする。

再現性はraw train、exp218 OOF decompressed SHA、v2 parity manifest SHA、exp115 assignment SHA、
calibration feature content SHA、OOF prediction decompressed SHAを記録する。乱数、GPU、PF/Beam再生成、
model trainingはない。

## v4 early / original / late 統合学習

v3はearly known-prefixをconfidence proxyへ使っただけでlateを学習へ入れておらず、中心仮説の
検証ではなかった。v4は保存済みexp218と同じ380特徴・3 model configを使い、official全行に
early/late pseudo rowを加えた新規variantを直接学習する。保存済みexp218 OOFをcontrolとし、
control自体は再学習しない。

position view数は`-1000:764 / -250:773 / +250:773 / +1000:771`、合計3,081。full 1,000
rows/viewでは3,078,562 pseudo rowsとなり、exp239 cacheの約3.85倍かつrequest単位のPF/likelihood
再生成が12時間上限を超え得る。このため各viewは`0-49 / 50-249 / 250-999 / 1000-2499 /
2500+`から各50行、最大250行へ固定samplingし、pseudo合計を770,157 rowsへ抑える。offsetを
削らず、距離帯coverageとposition方向を両方保持する。

pseudo feature cacheはoffset別4 CPU kernelに分割する。各kernelはraw horizontalからそのstartの
`TVT_input`を作り直し、exp072/PF/likPF、learned likelihood、U projection、GRWRを再生成する。
full-prefix cache sliceは使わない。offset別期待値は`-1000: 764 views / 191,000 rows`、
`-250: 773 / 193,250`、`+250: 773 / 193,157`、`+1000: 771 / 192,750`とする。

GPU trainはexp239 official 16 shardsと4 pseudo cacheをrow/content/file SHA付きで順次memmapへ読む。
各foldでouter-valid source wellのpseudo rowを除外し、official train weight 1.0、pseudo weight 0.5で
3 configsを学習する。validationはofficial-start全3,783,989行だけとし、保存済みexp218 OOFとの
差分を採否根拠にする。GPU学習のbitwise一致は仮定せず、input/schema/model/prediction SHAと
kernel versionを記録する。
