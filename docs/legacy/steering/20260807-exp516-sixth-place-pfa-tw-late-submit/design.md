# 設計

## アプローチ

公開submission Notebookから、GR-free anchor生成、learned-emission similarity、`pf_banks_v95`の`pfA`、whole-interval smootherに必要なセルだけを抽出してJupytext percent形式のself-contained inference Notebookへ持ち込む。参照Notebook全体の91候補・最終融合はコピーせず、公開componentとして数値が報告された`pfA × twGR`のsmoothed meanだけをsubmissionへ整列する。

参照kernel outputはKaggle inferenceのread-only `kernel_sources`として付け、5本の`stageA_enccapaug_f*.pt`を動的探索する。PF configは公開outputから取得済みの`v96_art/pf_banks_config.json`をvendor copyし、runtimeでもSHAを照合する。GR-free test anchorはcurrent hidden testから再生成し、visible test固有prediction/cacheを使わない。

## 手法忠実性

- 実装区分: `faithful`
- 参照sourceとの一致点: `pfA`全parameter、typewell GR表現、GR-free anchor mult 20、learned emission weight 0.01、600 particles、32 seeds、full ancestral smoothing、seed log-likelihood soft weighting。
- 参照sourceからの変更 / 省略点: 単体PF契約の数値機構は変更しない。変更はNotebook orchestration、runtime path resolver、ID contract guard、manifest/SHA保存だけ。91候補fusionは対象外。
- input tensor / feature: per wellのhidden suffix `MD, Z, GR`、typewell TVT-grid上のGR、prefix終端の`TVT+Z`とrate、row-level GR-free anchor mean/std、anchor中心±45 ftのlearned similarity band。
- target / objective: state `position=TVT+Z`とformation rateのsequential Bayesian approximation。anchor GRU以外に学習objectiveはない。
- output representation: particle/seed posteriorをwhole-interval smoothingしたrow-level TVT mean、seed dispersion、run log-likelihood。
- loss: PF本体なし。GR likelihoodは`|GR-observed - GR-typewell(TVT)| / sigma`のpower likelihood、anchor Gaussian kernel、learned similarity exponential factor。anchor GRUはmasked Huber loss（delta 8 ft）。
- decode / postprocess: final particle weightsを全祖先へ逆伝播し各seedのsmoothed pathを得て、seed log-likelihoodでsoft averageする。追加blend、gain、projectionなし。
- context unitと予測範囲: known-prefix境界で初期化したhidden suffix whole-well。anchor field構築はtrain field context。
- この実験が支持 / 棄却できる主張: 公開`pfA × twGR` componentをdynamic code submissionとして再生成し、late-sub scoreを監査できる。工程別runtimeとartifact SHAは公開commit runで監査し、Kaggle APIから取得できないhidden runtime / well数 / output SHAは主張しない。
- この実験では判断できない主張: 91候補のdiversity、row-level bagging、TCN/GBM、Submission A/B、6位最終scoreの再現。
- 実験名と実装機構の整合: `sixth_place_pfa_tw_late_submit`は6位解法の単体`pfA × twGR`とpost-competition submissionだけを表し、full 6th-place solutionとは呼ばない。

## 探索幅とpivot判定

- 変更class: `mechanism` + `representation`
- 同じ親 / familyで連続した小改善実験数: 既存PF/Beam実験は多数あるが、今回は公開上位解法の指定componentを固定再現する新source replayで、既存親の3件目micro-tuneではない。
- positiveなoracle headroom / coverage / 誤差非相関性: writeupは`twGR-prior PF alone`をCV 7.8 / Public 7.88 / Private 7.78と報告し、物理anchor追加でCV/Public/Private整合が改善したとしている。
- 比較したrepresentation-change案: `pfA × twGR`単体、91候補bank、91候補+learned fusion。ユーザー確認で単体PFを選択した。
- 小改善の継続またはpivotを選ぶ根拠: parameter tuningではなく、公開されたphysical-anchor PF mechanismのsource-faithful replayである。
- `kaggle-idea-forge`の実行要否と根拠: 不要。次案生成ではなく、ユーザー指定済み手法の再現である。

## 実験範囲

- 対象実験: `exp516_sixth_place_pfa_tw_late_submit`
- Route: `pf_beam`
- 親実験: scientific sourceはKaggle discussion 733226と公開kernel `k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`。repo内の予測parentは持たない。
- 変更する変数: 既存repo anchorから公開`pfA × twGR`componentへ置換し、単体componentを直接decodeする。
- 固定する変数: bank/representation/particles/seeds/smoother/config/checkpoints/submission回数を実行前に固定する。

## 再現性設計

- seed policy: 参照実装のPF seedを固定し、well順・chunk境界・device数をmanifestへ記録する。anchor GRUはfold/seedごとの`torch.manual_seed`を参照実装どおり使う。
- stochastic 処理の有無: PF transition/resampling、anchor GRU学習にあり。
- PF/Beam / likelihood-PF / seed bagging の有無: PFあり、Beamなし。32 PF seedsをrun log-likelihoodでsoft baggingする。
- 並列処理と乱数の関係: 公開実装はchunkごとにgeneratorをresetし、GPU device分配でchunk構成が乱数列に影響しうる。late-submit fixed pathはKaggle T4 x2、`PF_NGPU=2`、well sort/chunk policyを固定し、別device構成のbyte一致は主張しない。
- CPU/GPU runtime と deterministic flags: Kaggle T4 x2、internet off。PyTorch float32。GPU bitwise determinismは未保証のため、初回は`stochastic replay candidate`として扱う。
- train cache / test feature regeneration の SHA 記録方針: anchor manifest、similarity schema/content、well/row counts、PF config、candidate content SHAを保存する。visible prediction cacheはhiddenで利用しない。
- model manifest / prediction / submission SHA 記録方針: public encoder checkpoint 5本のpath/bytes/SHA、anchor model count、source/config SHA、prediction logical content SHA、submission SHAを保存する。
- Kaggle package bootstrap 確認方針: self-contained Notebook、vendor config、project configを埋め込み、生成packageのmetadata、bootstrap config、source SHA、kernel source、T4 x2、internet off、canonical slug/titleをpush前に照合する。

## リスク

- リークリスク: GR-free anchor生成時にvalidation/target mixingを起こさないよう公開5-foldコードをそのまま使う。hidden inferenceはcompetition train targetだけを使い、test true TVTを読まない。
- CV/LB 不一致リスク: writeup報告splitとlate hidden test、GPU、kernel imageが異なる。外部報告値との一致を技術gateにしない。
- ランタイム/メモリリスク: 600 particles × 32 seeds × full ancestry × hidden suffixをT4 x2で保持する。公開可変chunk VRAM budget 8 GB/device、長さsort、well逐次chunkを保持する。
- 再現性リスク: public sourceはchunk/device依存RNGとGPU reduction差を許容している。manifest固定とrerun SHAで測定し、未確認でdeterministic anchorと呼ばない。
- 手法忠実性リスク: checkpoint kernel sourceのmount pathやoutput versionが変わる可能性。5 checkpoint SHAをfail-closeで固定する。
- 過度な縮小 / proxy化リスク: particles、seeds、whole-interval smoothing、anchor、emissionの縮小/offを禁止する。debug smokeだけはデータ量を縮小しても手法契約を変えず、official late submitはfull設定だけを使う。
