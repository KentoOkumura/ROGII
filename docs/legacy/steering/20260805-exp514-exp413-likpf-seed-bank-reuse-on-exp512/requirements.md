# 要件

## 依頼

`exp512_hjyact_v2_final_10pct_hedge_on_exp413`で二重生成しているlikelihood-PFを、
exp413のwell別stable seed・Numba実装へ統一し、1 wellにつき1回だけ生成した128-seed bankを
SP45とexp413の両方へ渡す新実験を設計する。

今回の作業範囲は、バックログ、steering、実験ディレクトリの作成と設計確定までとする。
PF実装、Notebook実装、Kaggle package/run、出力取得、提出は行わない。

## 2026-08-05 実装承認による追加範囲

ユーザーの`exp514を実装してください`により、設計段階の後続作業だったPF実装、別名の
compact self-contained inference候補、Stage A fixed32専用Notebook、契約test、静的検証までを承認範囲へ追加する。
正規Notebook採用、Kaggle package/run、output取得、fixed200、hidden inference、提出は引き続き含めない。

## 2026-08-05 Stage A実行承認による追加範囲

ユーザーの`Stage Aを実行してください`により、Stage A fixed32専用NotebookのKaggle package/run/monitorと、
SHA/ledger判定に必要なreport・metrics取得を承認範囲へ追加する。実行量は32 wells × thread 1/4 × 各2 run、
合計128 well-bank生成、各500 particles × 128 seeds。Stage B/C、正規Notebook、hidden inference、提出は含めない。

## 固定する共有手順

1. exp413 likelihood-PFをSP45より前に一度だけ実行する。
2. 同じ500粒子×128軌跡とlog-likelihoodからscale `3/5/8/12`をすべて集約する。
3. SP45用にknown prefixを`TVT_input`から連結し、full-length配列へ変換する。
4. raw seed軌跡とlog-likelihoodからSP45のseed-branch統計を生成する。
5. scale 5をexp413へ、全scaleとbranch統計をSP45へ渡す。

## 2026-08-05 Stage B well数変更

ユーザー指示により、Stage B paired精度監査は200 wellsから32 wellsへ縮小する。Stage Aでtruthを読まずに
固定した同じ32 wellsとselection SHAを再利用し、wellの選び直しは禁止する。精度gateの閾値は変更しないが、
結果は小規模screeningであり200-well一般化の証明とは扱わない。

## 2026-08-05 Stage C不要化とruntime見積もり変更

ユーザー指示により、raw-only 200-well shadowを実行するStage Cは不要とし、実装・package・実行しない。
9時間上限の見積もりは、Stage Dのvisible test実行で得る工程別runtimeを200 wellsへ外挿して行う。
親exp512と同じく、4-way well並列工程は`visible batch秒 × 200/4`を下限、
`visible batch秒 × 200/visible wells`を上限とし、逐次工程はvisible throughputで外挿する。
固定費とI/Oはwell比例工程から分離する。この値はhidden 200 wellsの実測保証ではなく、
visible 3 wellsの長さ・欠損率・CPU競合に依存する高不確実性の見積もりとして扱う。

## 仮説

exp413のstable per-well seed bankを唯一のlikelihood-PF生成源にすれば、exp413のscale-5入力を
exactに維持しながらSP45の重複128-seed PFを除去できる。scale追加集約とbranch統計は既に生成済みの
seed bank上の軽量処理なので、200-well推論を9時間制限へ近づけられる。

SP45 legacy PFとexp413 PFはRNG、seed namespace、初回MD差分、typewell補間が異なるため、
SP45側は同値置換ではない。runtime改善だけで採用せず、Stage Aでtarget-freeに固定したtrain-like 32 wellsで
paired精度screeningを行う。

## 制約

- 対象実験: `exp514_exp413_likpf_seed_bank_reuse_on_exp512`
- Route: `ensemble`
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- likelihood-PF科学契約: exp413 `x1.0`、500 particles、128 seeds、temperature 5、
  `SHA256(likpf::<split>::<well>)` stable seed。
- exp413 consumerへ渡すscale 5は、親exp413 raw-test replayとcontent exactでなければならない。
- SP45 consumerにはscale 3/5/8/12、full-length配列、branch統計を渡す。
- shared bankの入力はdynamic raw splitだけとし、train/test overlap、visible well ID、固定3 well、
  14,151 rowsを使った分岐を禁止する。
- 親exp512のvisible physical overrideはこの実験で変更しないが、精度・runtime・再現性gateの
  positive evidenceには使わない。
- learned likelihood-PF `gs×1.3`、Goldのmasked-prefix PF、`pf_ancc`、`pf_z`、Beamは共有しない。
- exp413の保存済みmodel、SP45 selector/Beam/hold係数、learned、Gold、guarded overlap、
  seed-branch hedge、最終`0.50/0.50`式、model-package無効化は固定する。
- weight、temperature、scale、particle、seed数、selector、branch thresholdの探索を禁止する。
- LightGBM新規学習0、親/control再学習0、inference-time booster training 0。
- visible testの全体runtimeを単純に`200/3`倍しない。Stage Dで工程別timingを記録し、
  固定費、4-way well並列工程、逐次工程を分けて200 wellsへ外挿する。
- `docs/06_reproducibility.md`に従い、well別seed、thread schedule独立性、schema/content SHA、
  prediction/submission SHA、Kaggle versionを記録する。

## 受け入れ基準

### 今回の設計段階

- `docs/legacy/steering/20260805-exp514-exp413-likpf-seed-bank-reuse-on-exp512/`に
  `requirements.md`、`design.md`、`tasklist.md`がある。
- 実験ディレクトリにroute、lineage、共有bank契約、consumer契約、精度/runtime/reproducibility gate、
  禁止事項、実装未承認状態が記録されている。
- `KAGGLE_DIRECTION.md`へP0のdesign-frozen候補として追加されている。
- 実装コード、実行可能Notebook、Kaggle package/run、submissionを作成していない。

### 将来の実装・技術gate

- dynamic eligible wellごとのlikelihood-PF core実行回数がexactly 1で、二重生成が0である。
- exp413 scale 5は親replayとID、row order、float32 content SHAがexact一致する。
- scale 3/5/8/12は同じ`preds[128, n_eval]`と`liks[128]`から集約される。
- SP45 full-length配列はknown rowsで`TVT_input` exact、suffixで対応scale exact、finite coverage 1.0。
- branch統計は同じseed bankとtemperature 5から生成され、raw seed bankはwell処理後に解放される。
- shared nodeはtrain overlapやvisible/static sidecarを読まず、well並列順序を変えても出力SHAが一致する。

### 将来の精度gate

- Stage Aでtarget-freeに事前固定したtrain-like 32 wellsについて、legacy SP45とshared-bank SP45を
  paired生成してからtruthを読む。
- pooled selector RMSEのcandidate-control差が`+0.02 ft`以下。
- 5 reporting folds中4 fold以上で差が`+0.02 ft`以下。
- raw-GR observed/missing、high-missing、suffix 1000+、hidden-like 2面の最大悪化が`+0.05 ft`以下。
- by-well RMSE差p95が`+0.25 ft`以下、worstが`+5.0 ft`以下。
- いずれかFAILならscale/seed/selector等で救済せず終端する。

### 将来のruntime見積もり・再現性gate

- Stage C 200-well shadowは実装・実行しない。
- Stage D visible testでcandidate end-to-endと主要工程のwall time、well数、row数、effective worker数を実測する。
- 4-way well並列工程は`stage秒 × 200/4`から`stage秒 × 200/visible wells`、逐次工程は
  `stage秒 × 200/visible wells`、固定費は実測固定費として合算し、200-well lower/upper estimateを保存する。
- runtime判断は推定upperが9時間以下ならestimated PASS、超過ならestimated FAILとする。
  いずれもhidden実測PASSや9時間保証とは表記しない。
- SP45 legacy重複bank実行回数0、exp413 consumerの追加bank実行回数0。
- fixed32 technical replayの2回でaggregate/branch/content SHAが一致する。
- hidden-like dynamic cardinalityに固定行数・固定well数fallbackがない。
- deterministic anchorと呼ぶ場合だけ、input/source/model/feature/prediction/submission SHAと
  Kaggle kernel versionをすべて記録する。

## 次

ユーザーの追加承認により、Stage B scientific gateが未評価でもStage D visible-test package/runへ先行する。
Stage Dの結果はruntime/readiness証拠であり、Stage Bの精度gateを代替しない。
Stage Cは実行せず、competition submissionへはさらに別承認なしに進まない。

## 2026-08-05 Stage B version 2修正

ユーザー指示により、v1のpost-freeze採点ERRORを修正して同じfixed32契約で再実行する。変更は
`metric_bundle`の重複列防止だけとし、well、予測、selector、branch hedge、gate閾値は変更しない。

## 2026-08-05 Stage D version 2修正

ユーザー指示により、v1ログで200 wellsを先に暫定評価してからStage Dを修正・再実行する。親exp512
outputのexact一致はexp514の科学変更と両立しないため、v1で観測したexp514候補のvisible HJYACT SHAを
再現性witnessとして固定する。科学パイプラインとruntime外挿式は変更しない。

## 2026-08-05 Stage D version 3 runtime-only最適化

ユーザー指示により、Gold visible-prefix calibrationをwell単位4-processへ変更し、SP45が生成済みの
決定論test featureをHJYACTへ再利用する。likelihood-PF、`pf_ancc`、`pf_z`、Goldのseed・particle・cut、
保存model、selector、weight、最終式は変更しない。Stage A/B専用sourceは変更せず、再実行対象はStage Dだけとする。

- Goldは入力well順を固定し、各workerの結果を親processで同順にmergeする。process数は`min(4, well数)`。
- SP45/HJYACT共有対象はBeam、multi-scale NCC、formation/dense geometry、GR rolling、座標・傾斜などの
  決定論列とtrain由来imputer instance。HJYACT固有のstochastic `pf_ancc` / `pf_z`列は従来どおり別生成する。
- HJYACTのPF依存派生列（`sig_*`、`pf_vs_*`、`tdpf*`）はHJYACT側PFから再構成し、SP45側PFを流用しない。
- Stage D v2のHJYACT、Gold profile、exp413、最終submissionのcontent SHAをv3のfail-close witnessにする。
- いずれかのcontent SHAが不一致ならruntime改善があっても採用しない。
- 200-well見積もりではGoldを4-process並列工程へ移し、visible batch時間から既存式で外挿する。

## 2026-08-05 Stage D version 4 OOM修正

ユーザー指示により、hidden rerunの約40分時点の未処理例外とvisible parent peak RSS
`25,124.602 MiB`を受け、Stage Dだけに次のmemory-lifetime修正を加える。PFの数値契約、stable seed、
particles、seed数、4-thread並列、保存model、weight、最終式は変更しない。

- Ridge予測生成後、train feature、OOF、保存trainer、Ridge trainer、予測用matrixを明示解放する。
- shared likelihood-PF生成とSP45 consumerをwell単位の同じ4-thread workerへ統合し、SP45消費後に
  全scale full-length配列とrow-level audit配列を直ちに解放する。全200 wellsをfull payloadのまま保持しない。
- exp413まで保持するのは各wellの`likpf_scale_5` / `likpf_mean` float32とmanifest/ledger情報だけとし、
  exp413 adapter消費時にそのcompact frameも解放する。
- SP45 Ridge用の一時frameを必要列だけに縮小し、SP45決定論frameはdeep copyせずHJYACTへ所有権移譲し、
  HJYACT frameもdeep copyせずexp413へ所有権移譲する。外部helper自身がcopy/mergeする呼び出し前の
  重複`pf_frame.copy()`は除去する。
- Stage D v2 visible出力SHA 5件はvisible sample ID SHA一致時だけfail-close検査し、hidden dynamic sampleでは
  `SKIPPED_HIDDEN_DYNAMIC`として記録する。
- 修正後は静的検証とcontract testを行う。Kaggle再実行・competition再提出はこの実装依頼には含めない。
