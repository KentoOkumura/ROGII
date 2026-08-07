# 設計

## 仮説

exp512では、同じlikelihood-PF familyをSP45とexp413が別々に500粒子×128 seedで生成している。
exp413のstable per-well seed・Numba kernelを唯一のseed-bank producerに固定し、全temperature集約と
branch統計を同一bankから派生させれば、exp413入力を維持したままSP45の重複計算を除去できる。

SP45 legacyのPCG64 seed 0--127をexp413 seed bankへ変えるため、SP45予測は変化する。
したがって本実験はsource parity実験ではなく、runtime最適化を伴う新しいscientific variantとして扱う。

## 実験範囲

- 対象実験: `exp514_exp413_likpf_seed_bank_reuse_on_exp512`
- Route: `ensemble`
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- PF producer: exp413 / exp073 replayの`x1.0` likelihood-PF kernel
- 比較control: exp512 SP45 legacy `default_rng(seed=0..127)` bank
- 変更する変数: SP45が読む128-seed bankだけをexp413 stable-seed bankへ統一する。
- 固定する変数: particles、seed count、transition/noise/resampling、scale集合、SP45 selector/Beam/hold、
  branch hedge式、learned/Gold、exp413保存model、最終0.50/0.50式、model-package無効化。
- 学習量: model config 0、fold training 0、booster 0、親/control再学習0。

## Producer / consumer DAG

```text
dynamic raw well
  -> exp413 x1.0 stable-seed Numba PF (500 particles x 128 seeds, exactly once)
      -> preds[128, n_eval] + loglik[128]
          -> aggregate scale 3/5/8/12 + arithmetic mean
          -> branch summary at temperature 5
          -> release raw preds/loglik after both summaries are materialized
      -> SP45 adapter
          -> known TVT_input + suffix scale arrays = full-length arrays
          -> all scales + branch summary -> existing SP45 selector/hedge
      -> exp413 adapter
          -> suffix scale 5 / delta -> existing exp413 feature graph and saved models
```

`pf_ancc`、`pf_z`、learned `gs×1.3` likelihood-PF、Gold masked-prefix PF、Beamは別producerのままにする。

## Shared producer契約

- 入力: raw horizontal `MD/Z/GR/TVT_input`、raw typewell `TVT/GR`。
- well列挙: dynamic sample IDから導出し、nonempty suffixを持つ全wellを処理する。
- seed base: `stable_seed("likpf", split, well)`。
- seed index: `seed_base + 0..127`。
- particles: 500。
- GR sigma: known prefix zero-fill population std、clip `[10, 60]`、multiplier `1.0`。
- typewell grid: step `0.2 ft`。
- init spread / dynamics: `4.5 / 0.998 / 0.002 / 0.005 / 0.1 / 0.001 / 0.5`。
- aggregation temperatures: `[3.0, 5.0, 8.0, 12.0]`。
- branch summary: temperature 5のseed weightと各seed suffix medianから、SP45と同じ二分割SSE式で
  `center_low/high`、`mass_low/high`、`weighted_center`、`eval_rows`を作る。
- dtype: kernel内部float64、consumer frameはexp413契約に合わせてfloat32。SHAはdtypeを含めて記録する。
- memory: raw `preds/liks`を全well cacheへ保存しない。well内でaggregateとbranch summaryを作成後に解放する。
- generation ledger: producer call、core call、consumer hit、fallbackをwell単位で記録する。

## SP45 adapter契約

- `TVT_input.notna()` rowは入力値を変更せずfull-length配列へコピーする。
- suffix indexへ各scaleのaggregateを配置する。
- 既存selectorが要求する`pf_scale_3/5/8/12`と`pf_mean`を同じshapeで供給する。
- branch summaryを既存`PF_SEED_BRANCH_STATS` schemaへ渡す。
- selector bin、Beam、hold weight、bimodal detector、後段branch hedgeの係数は変更しない。
- fallbackはproducer failureを隠さずfail-closeする。last-known複製によるscientific continuationは禁止する。

## exp413 adapter契約

- `likpf_scale_5`、`likpf_scale_5_d`、semantic `likpf_mean`だけをexp413 model consumerへ渡す。
- 親exp413 replayを同じraw inputで単独実行したscale 5とID/order/content exactに一致することを必須にする。
- scale 3/8/12はSP45専用であり、exp413 feature/model schemaへ追加しない。
- exp413後段で`build_likpf`を再実行しない。producer ledgerに2回目のcore callがあればFAILする。

## Legacy SP45との差と解釈

固定hyperparameterとGR sigmaはほぼ同じだが、次は意図的に変わる。

- RNG: PCG64 `default_rng`からNumba seeded RNGへ変更。
- seed namespace: 全well共通0--127からwell別SHA256 baseへ変更。
- 初回MD propagation: last-known MDからの実差分ではなくexp413 kernelのfirst-eval 1 ft契約。
- typewell lookup: raw pointsへの`np.interp`から0.2 ft grid lookupへ変更。
- output dtype: SP45 full float64中心からexp413 suffix float32 consumer契約へ変更。

この差を隠して「高速化のみ」や「exact parity」と表記しない。

## visible / hidden分離

- shared producerと両adapterはtrain well集合を読まず、raw splitとdynamic sampleだけを使う。
- 親SP45のvisible physical overrideは本実験の変更対象外として固定する。
- ただしvisible override後のfinal一致はPF置換品質を観測できないため、precision gateや
  deterministic gateの根拠に使わない。runtimeはStage Dの工程別実測だけを高不確実性の外挿元に使う。
- 科学比較はStage Aと同じtrain-like masked suffixのfixed32で行い、truth/errorはcontrol/candidate freeze後だけ読む。
- fixed32はStage Aのtarget-free strata / stable SHA selectionをselection SHAごと再利用し、well ID、error、
  Stage B結果で選び直さない。

## 段階gate

### Stage A: fixed32 technical / determinism

- target-free stable SHAで32 eligible wellsを固定する。
- exp413 scale5 standalone parity、all-scale aggregation、full-length adapter、branch summary、ledger、finite、
  memory解放、thread count `1/4`のcontent parityを確認する。
- 同一package 2回でaggregate/branch SHA一致を確認する。
- truth、RMSE、fold、hidden-like roleはこの段階で読まない。
- 1条件でもFAILならStage Bへ進まず終了する。

### Stage B: fixed32 paired scientific screening

- Stage Aでtarget-freeに固定済みの同じ32 wellsについて、legacy SP45 bankとcandidate shared bankを
  同じraw prefix/typewellから生成する。
- 両predictionと設定・source・content SHAをfreezeしてからsuffix TVTとreporting foldをjoinする。
- primaryは既存SP45 selector出力のpooled RMSE差`candidate-control`。
- gateはpooled `<= +0.02 ft`、fold nonworse `>=4/5`、固定scope max `<=+0.05 ft`、
  by-well p95 `<=+0.25 ft`、worst `<=+5.0 ft`のall-AND。
- branch hedge適用前後を両方reportするが、採用判定は後段契約を含む適用後をprimaryとする。
- threshold、seed、scale、selector、well subsetを同じtruthで選び直さない。
- 32 wellsのPASSは小規模screeningの通過であり、200-well accuracy generalizationの証拠とは表記しない。

### Stage C: ユーザー指示により削除

- raw-only 200-well end-to-end shadowは実装・package・実行しない。
- `observed <=8.5h`とbootstrap p95 `<=9.0h`の実測gateは廃止する。
- Stage CをPASS済みとは扱わず、`not_required_by_user_override`として履歴に残す。

### Stage D: hidden inference readinessとvisible runtime外挿（実行承認済み）

- Stage A PASSを前提とし、ユーザー追加判断によりStage B scientific gate未評価でもcompact inference候補の
  visible-test Kaggle package/runを先行する。Stage DはStage Bの精度根拠を代替しない。
- dynamic sample ID one-to-one、finite、fallback 0、source/model/input SHA、producer ledgerを確認する。
- hidden stochastic pathを同一package 2回で再現できるまでdeterministic anchorと呼ばない。
- visible test実行の工程別wall time、visible well/row数、effective worker数を記録する。
- 4-way並列工程は`stage秒 × 200/4`をlower、`stage秒 × 200/visible wells`をupper、
  逐次工程はvisible throughput、固定費は一回分として合算する。推定upper `<=9h`をestimated PASSとする。
- runtime結論には必ず`visible-stagewise estimate / hidden runtime not observed`を付記する。
- submit-check、competition submission、監視はさらに別承認とする。

## Notebook実装

最初の実装対象は正規Notebookではなく、Jupytext percent形式の別名
`exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.py`とする。
章立ては次を最低限含む。

1. Imports, source identity, and configuration
2. Dynamic input and hidden-cardinality checks
3. Shared exp413 likelihood-PF producer
4. SP45 and exp413 consumer adapters
5. Existing exp512 component orchestration
6. Generation ledger, runtime, memory, and reproducibility audits
7. Prediction and submission outputs

正規Notebookは実装・静的検証・採用の別承認までplaceholderを維持する。

実装時には上記のfull inference候補に加え、Stage Aだけを単独実行する
`exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.py` / `.ipynb`を生成する。
このNotebookはtarget-free属性による32 well選定、thread 1/4、各2 run、aggregate / branch / ledger SHAだけを扱い、
suffix truth、full exp512 inference、submission生成を実行しない。

## 再現性設計

- stochastic component: shared likelihood-PF、learned PF、Gold seed bank、残存PF trackers。
- shared PF seed policy: feature family / split / wellのSHA256からstable baseを作り、seed indexを加算する。
- parallel policy: wellごとにprivate Numba RNG stateを持ち、thread scheduleから独立させる。
- fixed order: well、row、scale、consumer merge orderを明示sortする。
- SHA: source/config/input、ID order、aggregate frame、branch summary、generation ledger、model manifest、
  component prediction、final prediction、submissionを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- Kaggle package作成時はbootstrap内config/source SHAとmetadataをreadbackする。

## リスク

- 精度: SP45 seed/RNG/first-step/grid変更でwrong branch分布が変わる。
- visible blind spot: visible finalはphysical overrideでPF差を隠すため、visible parityはscientific evidenceにならない。
- runtime:重複PFを消してもGold、learned、exp413 HMM/K16が残り、上限側は9時間を超え得る。
  Stage Cを削除したため、Stage D visible外挿はhiddenのwell長・row数・欠損率・競合を直接観測しない。
- memory: raw128 pathをwell横断cacheすると増大する。aggregate/summary後即解放を必須にする。
- reproducibility: Numba RNGをthread間共有するとschedule依存になる。per-well private callとSHA rerunを必須にする。
- contract drift: exp413 consumerへscale 3/8/12を追加すると保存model schemaが変わるため禁止する。
- scope creep: visible-only branch除去、learned/Gold PF共有、weight/profile変更は別仮説として本実験へ混ぜない。

## Stage D v3 runtime-only設計

### SP45/HJYACT決定論feature共有

SP45の`build_dataset(test)`を唯一の決定論feature producerとし、そのframeをHJYACTへ渡す。HJYACTでは
`pf_ancc` / `pf_z`だけを従来のHJYACT kernelでwell別に再生成し、PFに依存する列だけを上書きする。
これにより、2回目のBeam、NCC、formation/dense KNN、rolling/geometry生成と、2回目のtrain 773-well
imputer構築を除去する。共有列とPF再生成列はmanifestへ列名・content SHA・生成回数を保存する。

### Gold 4-process

Goldの1 well calibrationをpureなworker単位へ切り出し、Linux/Kaggle上のjoblib multiprocessing backendで
最大4 process実行する。各well内の3 cut、PF seed 0..23、final seed 0..47、350 particles、候補選択式は固定する。
worker結果は`order`付きで返し、親processが元のwell順へsortしてreport・candidate mapを構成する。
重複ID、欠落well、worker exception、backend/effective worker不一致はfail-closeする。

### 検証境界

Stage A/Bの対象であるshared likelihood-PF producerとpaired accuracy sourceは変更しない。Stage D v3だけを
再実行し、v2のcomponent/profile/final content SHA完全一致、finite/ID、worker contract、runtime reportを確認する。
visible 3 wellsの同値性はhidden 200 wellsのprocess memory・throughput保証ではないため、runtime推定の不確実性はhighのままにする。

## Stage D v4 memory-lifetime設計

### shared PF / SP45 bounded lifetime

shared likelihood-PF producerとSP45 Beam/selectorを同じwell worker内で連続実行する。workerは
`_shared_likpf_one_well -> shared_likpf_sp45_adapter -> existing SP45 selector`の順に処理し、SP45結果を
row辞書へ縮約した直後に`sp45_full`、`row_index`、`evaluation_index`、`known_mask`をrecordから除去する。
joblibの同時worker数は4のままなので、raw 128-seed pathと全scale full payloadの同時保持上限は4 wellsである。
全worker完了後に保持するcompact bankはexp413用2列float32、branch summary、SHA、audit、ledgerだけとする。

exp413 adapterはcompact frameを余分にdeep-copyせずconcatし、concat完了後に各recordの
`exp413_frame`を除去する。final manifestはfull/compact row payloadが残っていないことをfail-close確認する。

### DataFrame ownership transfer

- SP45 Ridge一時予測frameは`id/well/md_since`と`pred`だけを持つ。
- Ridge出力確定後、train/OOF/model/matrix変数を明示解放し、GCとLinux `malloc_trim`をbest-effort実行する。
- SP45 deterministic frameはHJYACT refreshでin-place更新し、旧global参照を除去する。
- HJYACT frameはcandidate reuse SHA確定後、`globals().pop`でexp413へ唯一所有権を渡す。
- exp413入口ではdeep copyせず受領frameを使用し、外部helperが内部copy/mergeする2箇所のcaller-side copyを除去する。

各変更は列値、dtype、ID順、well順、stable seedを変えない。visible sampleでは従来の5出力SHAを検査し、
hiddenではvisible固定SHAを評価しない。
