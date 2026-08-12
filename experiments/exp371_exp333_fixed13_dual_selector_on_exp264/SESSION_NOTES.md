# exp371_exp333_fixed13_dual_selector_on_exp264 セッションノート

## 目的

exp333 Stage 1 OOFをcorrected exp264 deployable12へ1本だけ追加し、同じ
candidate-long dual selectorをfixed13として再学習する。

## 現在の状態

- Route: `ml_model`
- 状態: Stage D canonical Kaggle T4 version 1完了、平均改善 / tail gate FAILで終了
- 親: corrected exp264 Stage C v6
- 比較基準: 保存済みfixed12 outer-valid selector score
- Stage D active variant / configs / outer folds: `1 / 3 / 5`
- trained CPU boosters: `40`
- parent/control retraining: `0`
- trained GPU / downstream TVT / inference / submission: `15 / true / false / false`
- `execution.run_approved`: `false`（version 1 push後に承認を消費）
- 正規notebookへの採用: 2026-07-24承認済み

## 実行承認（2026-07-24）

- ユーザー指示: 「実行してください」
- 承認scope:
  `fixed13_stage_a_plus_stage_c_1_variant_2_objectives_5_outer_4_inner_40_cpu_boosters_no_control_retraining`
- active variant: `1`
- LightGBM config / objective: `2`
- outer / inner folds: `5 / 4`
- 合計CPU booster: `1 × 2 × 5 × 4 = 40`
- parent/control再学習: `0`
- GPU booster / downstream TVT / inference / submission: `0 / false / false / false`
- compact self-contained train / inferenceを同名の正規notebookへ採用する。
- この承認はKaggle CPU train v1の1実行だけに使用し、retryや後続実行へ流用しない。

## 設計

- `.steering/20260724-exp371-exp333-fixed13-dual-selector-on-exp264/`を先に作成した。
- exp263 fixed12の候補順序・formula・fixed fallback 7本を固定した。
- `exp333_segment_offset`を13本目、primary domainの12本目へ追加した。
- exp333にsource-native confidenceはないため、`confidence_valid=false`とし、
  candidate value、anchor差、32/128/512 shape、bank disagreement、
  candidate/family/kind one-hotだけを使う。
- corrected exp264と同じ`pred_abs_error` / `p_within10`、outer 5 × inner 4、
  sampling、LightGBM設定を使う。
- exp333 saved-exp226 source foldはOOF provenanceとしてだけ保持する。
  全行を`well_id,row_idx`でglobal key index化し、exp263 selector foldへ再partitionする。
  source foldとselector foldの一致は要求せず、source foldはmodel featureにしない。
- Stage Aを同じrunのfit前に実行し、13候補用feature schemaをfreezeする。
- compact metaは74列から77列。追加はexp333候補別2 scoreとprimary top1 one-hot。
- Stage C後、保存済みexp264 scoreと行単位pairingし、candidate usageと
  pooled/fold/near/1000+/hidden-like/by-wellを評価する。

## 事前gate

- technical:
  - `3,783,989 rows / 773 wells / 13 candidates`
  - exp333 file/decompressed SHA、exp263とのglobal key parity
  - source/selector fold 5×5 overlap、missing key 0、source fold feature利用0
  - truth/error pre-freeze load 0
  - 40 models / 25 compact partitions / 18,919,945 compact rows
  - 49,191,857 outer-valid candidate-long rows / compact 77列
- score:
  - 2目的ともouter-train candidate priorをpooledと4/5 folds以上で改善
- integration:
  - exp333 primary top1使用率 pooled `>=0.5%`、4/5 foldsで正
  - parent fixed12 hard selectorをpooledで非悪化、4/5 folds改善
  - near / 1000+ / hidden-like delta `<=+0.02 ft`
  - by-well p95 / worst delta `<=+0.25 ft`

## 再現性

- seed: 42
- sampling: stage/fold immutable keyからstable SHA256 seed
- LightGBM: deterministic / force_col_wise / n_jobs 8
- runtime: Kaggle CPU、internet/GPU off
- exp333 OOF file SHA:
  `70b623d4c839c4f7eb11fb2134aa214ca8f0ce8d6ebe65e723d2fffa95dcc2dc`
- exp333 OOF decompressed SHA:
  `f2ebc6f6ea243b45fdb785342b8815b3b04947f96d787d3017e5e2be7ff92e5a`
- parent exp264 score SHA:
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- exp333 current-test candidate decompressed SHA（将来inference契約、train入力ではない）:
  `7571c6281bd2ab484e7bf536a876b8072407b272a0ef0ec5112ca06897a717cd`
- pandas round-trip logical hashをhard contractにせず、gzipはdecompressed SHAを主証拠にする。
- GPU bitwise reproducibilityは主張しない。Stage Dではmodel / OOF / artifact SHAを記録する。

## 実装内容

- `candidate_contract.yaml`: fixed12 + exp333の13候補、primary 12 / fixed 7を固定。
- `feature_contract.yaml`: 74 -> 77 compact差分とraw-test-safe契約。
- `experiments/exp371_exp333_fixed13_dual_selector_on_exp264/exp333_fixed13_candidate_cache.py`:
  - exp333 target-free allowlist loader
  - file/decompressed SHAとpost-read prediction content SHA
  - global key join後のexp263 selector-fold-safe add-one
  - exp333 source fold provenanceと5×5 overlap audit
  - parent/new candidate score Parquetのrow-group readout
  - fixed12比較、candidate usage、安全surface、scientific gate
- compact self-contained train source:
  - 8章、483行。親exp264 trainは7章、465行。
  - 入力、実行量、Stage A、Stage C、paired readout、SHA/summaryをnotebook上で追える。
- inference source: Stage C gateとdownstream TVT未完了のためfail-closed。
- 専用test:
  - fixed13 contract / compact77
  - base fixed12 contract parity
  - exp333 allowlist / decompressed SHA
  - row-order alignment
  - fixed13 vs fixed12 readout
  - gzip metadata非依存

## コマンドログ

- `make new-steering EXP=exp371_exp333_fixed13_dual_selector_on_exp264`
- `make new-exp EXP=exp371_exp333_fixed13_dual_selector_on_exp264`
- `py_compile`: helper / train / inference source PASS
- `ruff --select F821,F401,E9`: PASS
- `pytest -q experiments/exp371_exp333_fixed13_dual_selector_on_exp264/tests/test_exp371_exp333_fixed13_dual_selector.py`: 初版`6 passed`
- Jupytext compact train / inference notebook生成: 完了
- 2026-07-24: compact train / inferenceを正規notebookへ採用。
- 正規採用後: Jupytext test / py_compile / Ruff PASS。
- 正規採用後の回帰bundle: `43 passed`。
- `make validate-exp EXP=exp371_exp333_fixed13_dual_selector_on_exp264`: strict PASS。
- Kaggle train package:
  - kernel: `kentookumura/exp371-exp333-fixed13-selector-train`
  - private / CPU / internet off / run-on-push: `true / true / true / true`
  - package notebook SHA256:
    `f7f97716d8b97a7a6ccc3eb7373e4b3349c521edb5740304344013df580e3fcb`
  - embedded support ZIP SHA256:
    `a0a100634a14dac2b9103e8532b810637ca955693422f2172193e0f7970ad74c`
  - embedded support: 23 files、manifest path/size/SHA検証付き
  - 同一slug既存kernel: なし
- Kaggle push:
  - version: `1`
  - id_no: `128372803`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp371-exp333-fixed13-selector-train`
  - push直後status: `KernelWorkerStatus.RUNNING`
  - pull-back metadata: private / CPU / TPU off / internet off / 3 kernel sources一致
  - pull-back metadata SHA256:
    `8c3c33f13383d58f05796fa3364e0287a99289fd4b20b0119724d3eacea351f5`
  - version 1はmodel fit前に停止したため、同scopeのversion 2 technical retryへ承認を引き継ぐ。

## Version 1 技術停止と修正

- 最終status: `KernelWorkerStatus.ERROR`
- 停止位置: 入力path解決、model fit前。trained booster `0`。
- 原因: `_expand_paths()`が絶対path patternを`Path.glob()`へ渡し、
  Python 3.12で`NotImplementedError: Non-relative patterns are unsupported`。
- 修正: 絶対patternは直接存在確認だけに使い、search rootのglob対象から除外した。
- 回帰test: Kaggle絶対patternと`**/candidate.bin`相対patternを同時に渡すcaseを追加。
- 修正後: py_compile / Ruff / `44 passed` / strict validation PASS。
- 科学的変更: なし。候補、特徴、fold、sampling、LightGBM、gate、40 booster scopeは同一。
- version 1で予定学習が始まっていないため、同じcanonical slugのversion 2を
  承認済みscopeのtechnical retryとして実行する。追加variant/control再学習は0。
- version 2 package notebook SHA256:
  `d0ab6225e76f228a6b23ea6ab66351ee8d7cf6b4ef9c0142a5a1237c44e3a6b7`
- version 2 embedded helper SHA256:
  `90eb0070b5c8360cb76af74c446a41ab5bc386e56dd3031f11d6723610819364`
- version 2 push: 成功。実行承認は消費済み。

## Version 2 fold guard停止

- 最終status: `KernelWorkerStatus.ERROR`
- path解決、exp333 file/decompressed SHA、target-free allowlist、
  `3,783,989 rows / 773 wells / 5 source folds`はPASS。
- 停止位置: fixed13 cacheのfold parity preview、model fit前。trained booster `0`。
- exp263 selector fold row counts:
  `757,738 / 756,650 / 756,255 / 757,101 / 756,245`
- exp333 saved-exp226 source fold row counts:
  `742,514 / 770,907 / 746,011 / 746,131 / 778,426`
- 例: well `000d7d20`はexp263 selector fold `0`、exp333 source fold `3`。
- 両者は全体key coverageが同じでもwell単位outer-fold assignmentが異なる。
- exp263 builderはsource artifactのfoldを固定せず、全candidateをcanonical keyへ揃えた後に
  row-count-balanced foldを独自生成する。一方exp333は保存exp226のsource foldでOOFを生成した。
- したがって承認時の`exp263_outer_fold_equal_to_exp333_outer_fold` hard contractは事実と不一致。
- 追加retryは停止。次のいずれかは結果に影響する設計判断なのでユーザー確認が必要:
  1. parent exp264と同じbank semanticsでexp333をkey joinし、exp263 selector foldへ再partitionする。
  2. strict alignmentのためexp263 foldsでexp333 candidate OOFを再学習する。
  3. fixed13 routeを閉じる。
- version 1 / 2ともselector booster `0`、control再学習 `0`、GPU/inference/submission `0`。

## Version 3 方針・実行承認（2026-07-24）

- ユーザー指示: 「それで実行してください」
- 採用方針:
  `global_key_join_then_exp263_selector_fold_repartition`
- exp333再学習: `0`。保存済みStage 1 OOFを使う。
- source fold: saved-exp226 OOF provenanceとして保存、model feature利用 `false`。
- selector fold: 親exp264と同じexp263 row-count-balanced outer fold。
- active variant / objectives / outer / inner: `1 / 2 / 5 / 4`
- 合計CPU selector booster: `40`
- parent/control再学習 / GPU / downstream TVT / inference / submission:
  `0 / 0 / false / false / false`
- 新technical artifact:
  `exp371_exp333_selector_fold_repartition.json`
- 修正後検証:
  - dedicated tests: `8 passed`
  - exp371 / exp333 / exp264 / notebook regression bundle: `45 passed`
  - py_compile / Ruff / Jupytext / strict validation: PASS
- version 3は同じcanonical kernel idへ追加する。version 3以後のretryは別判断とする。
- version 3 package監査:
  - notebook SHA256:
    `bf89c2a81621edd8a34c5730aa5f179ed12fe2b4ab5e8d829d9c2e42a0c5edbb`
  - config SHA256:
    `7cef7ab24c6d770be7da3036691261daa7444ad246e797c729368a48cb034a91`
  - embedded helper SHA256:
    `fc2899b538e9cf362dcfcc83324cb7f662b5a34733b9feff0dd1a16233244b58`
  - support ZIP SHA256:
    `5e62b955ce5b55e147e2e013b052fd8ad30793adb7243b00a8994000b118e298`
  - 23 embedded filesのpath / bytes / SHA: PASS
  - metadata: private / CPU / TPU off / internet off / run-on-push
  - kernel sources: exp263 / exp333 / exp264の3件
  - embedded config: version 3 fold policy / 40 boosters / control 0 / inference 0 / submission 0
  - pre-push pull: canonical id_no `128372803`、version 2存在、metadata一致
- version 3 push: 成功。canonical URL:
  `https://www.kaggle.com/code/kentookumura/exp371-exp333-fixed13-selector-train`
- version 3 pull-back metadata: id_no `128372803`、private / CPU / TPU off /
  internet off / exp263・exp264・exp333 sources一致。
- push直後status: `KernelWorkerStatus.RUNNING`
- version 3はglobal key join / repartition guardとStage Aを通過し、
  Stage C LightGBM学習へ到達した。Stage Aは153特徴を監査し90特徴をfreeze、
  compact 77列、feature schema SHA256
  `4665ca7317ddcb993326e66ee19aa908f4aeff5fe88b2b16bac3db12c35b665f`。
- ユーザー指示によりCLI follow監視を停止。Kaggle kernel本体は停止せず継続する。
  完了連絡後にstatus/logs/gateを取得して記録する。

## Version 3 完了結果

- ユーザー完了連絡後、canonical kernel version 3の通常logsを取得した。
- final status:
  `kaggle_cpu_stage_a_stage_c_completed`
- decision:
  `FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`
- runtime: `6761.965850462 sec`
- 実行量:
  - selector models: `40 / 40`
  - compact partitions: `25 / 25`
  - compact rows: `18,919,945`
  - outer-valid candidate-long rows: `49,191,857`
  - parent/control retraining: `0`
  - GPU / downstream TVT / inference / submission: `0 / false / false / false`
- Stage A:
  - audit rows: `650,000`
  - feature count: `153 -> 90`
  - all-missing / constant / exact duplicate drop: `41 / 5 / 17`
  - compact meta: `77`
  - feature schema SHA:
    `4665ca7317ddcb993326e66ee19aa908f4aeff5fe88b2b16bac3db12c35b665f`
- exp333入力・fold:
  - `3,783,989 rows / 773 wells`
  - truth/error pre-freeze load: `0`
  - global key join / selector-fold repartition: PASS
  - missing key: `0`
  - source fold row-count保存: PASS
  - source foldのmodel feature利用: `false`
- selector score:
  - expected-error MAE: `3.757694865`、prior `5.699947515`
  - within10 logloss: `0.357103106`、prior `0.507452097`
  - within10 Brier: `0.111087716`、prior `0.163875249`
  - pooledと5/5 foldsで3指標改善、score guard PASS
- leakage audit:
  - outer-validをinner assignmentから除外: PASS
  - inner train/valid well disjoint: PASS
  - outer-train compactはinner OOF、outer-validは4 inner model ensemble: PASS
- fixed13 vs saved parent fixed12:
  - pooled: `8.419997371 vs 8.652531956`、`-0.232534584 ft`
  - fold改善: `4 / 5`。fold 3のみ`+0.116619033 ft`
  - near 0--250: `-0.023116065 ft`
  - 1000+: `-0.263067757 ft`
  - hidden-like spatial / typewell-purged:
    `-0.406777133 / -0.403270188 ft`
  - fixed fallback `8.238331546`比: `+0.181665825 ft`
- exp333 candidate usage:
  - pooled primary top1: `6.267989%`
  - positive usage folds: `5 / 5`
- well safety:
  - improved / worsened: `400 / 373 wells`
  - median delta: `-0.006019365 ft`
  - p95 delta: `+0.861529323 ft`、上限`+0.25 ft`をFAIL
  - worst `a48640d9`: `3.349108 -> 14.107105`、
    `+10.757996620 ft`、上限`+0.25 ft`をFAIL
  - worst wellのexp333 top1率: `42.961165%`
  - 全wellのexp333使用率とdeltaのPearson: `-0.070004`
- scientific gate:
  - pooled / folds / usage / near / 1000+ / hidden-like: PASS
  - by-well p95 / worst well: FAIL
  - overall: FAIL
- 取得方針:
  - CV詳細を確定するため、Kaggle output archive全体ではなく評価CSV/JSON 7件だけを
    `kaggle kernels output --file-pattern`で選択取得した。
  - 保存先:
    `kaggle/output/train_v3/artifacts/`
  - logs:
    `kaggle/output/train_v3/logs.json`
- 主要SHA:
  - selector model manifest:
    `c5b70f32d698056336fe98eddd87f9f5fb041adea3797b56ede16b756d44396d`
  - compact manifest:
    `534e8278ad0e0dddc04a94236e949e9a5680342138bdebfdbf22fd8fb4f08956`
  - outer-valid candidate score:
    `5601b369704d36b4e8e8fba342a153ce09a8016f465a1a0bc88bb8beccecd9df`
  - scope metrics:
    `d9076e584f451a11815164bed3c7670e428ec18afa9bcf170b3e0b48def7adc4`
  - candidate usage:
    `35ffd93939457bf55d89acf938b758e17cecef23ab78eabd8ace7fd2772d2576`
  - by-well:
    `2783e57fd9728b75a30be818975822ab16f987323f0daf30c768b12aa8bb248c`
  - scientific gate:
    `7577796820679986adce7c98a264077ce2fb6276d16bfe6bdfce70b3ef47d786`

## 未実行

- downstream TVT
- current-test inference
- submission

## 次のアクション

1. fixed13 selector branchを閉じ、downstream TVT、inference、submissionへ進めない。
2. candidate weight、使用率threshold、domain、gateを同じOOFで救済調整しない。
3. exp333固有の原因確認を再訪する場合は、保存済みOOFの0-booster attributionに限定し、
   予測変更とは別実験・別承認にする。

## Stage D 明示例外進行（2026-07-24）

- ユーザー指示: 「平均で改善しているのなら次に進みましょう。」
- 解釈:
  - fixed13 selectorのpooled改善`-0.232534584 ft`を下流TVTで検証する。
  - 元のStage C scientific gate FAIL、by-well p95 `+0.861529323 ft`、
    worst `+10.757996620 ft`は変更・再分類しない。
  - gate閾値、candidate weight、使用率threshold、selector OOFは調整しない。
- 承認scope:
  `full13_compact350_addonly15_three_configs_five_folds_15_gpu_boosters`
- GPUコスト確認:
  - active variant: `1`（`selector_compact_addonly`）
  - LightGBM configs: `3`（indices `0,1,2`）
  - folds: `5`
  - 合計GPU booster: `1 × 3 × 5 = 15`
  - runtime: Kaggle T4 / internet off / GPU double precision /
    deterministic / force_col_wise / threads 8
  - saved exp264 parent/control再学習: `0`
  - selector/candidate/PF/HMM再生成: `0`
  - inference / submission: `false / false`
- feature surface:
  - exp218 source: `380`
  - clean allowlist後: `273`
  - exp371 fixed13 compact: `77`
  - final: `350`
- 比較対象:
  - saved exp264 Stage D v3 parent12 compact add-only:
    `8.460811237612477`
  - saved matched clean273 control:
    `10.476169179272501`
- Stage D gate:
  - pooled改善
  - 改善fold `>=3/5`
  - near / 1000+ / hidden-like最大悪化 `<=+0.02 ft`
  - by-well p95 / worst最大悪化 `<=+0.25 ft`
- SHA固定したexp371 Stage C入力:
  - `nested_selector_metrics.json`:
    `201cc87eff38fcb8c82c712046c3ca795fdf84aba3c07af31503e010bcaa5edb`
  - `nested_selector_model_manifest.json`:
    `c5b70f32d698056336fe98eddd87f9f5fb041adea3797b56ede16b756d44396d`
  - `nested_compact_manifest.json`:
    `534e8278ad0e0dddc04a94236e949e9a5680342138bdebfdbf22fd8fb4f08956`
  - `compact_meta_schema.json` file:
    `3ad0c88b0f7b4597e07f4a5fbfeb359c897ff8cc9f0c6365351358bbaf6ffb34`
  - compact77 logical schema:
    `76242ae603c1afa99813e8d84bad67f6663b698181732ee8697339f66b10235e`
- 実装:
  - readable Jupytext source
    `exp371_exp333_fixed13_dual_selector_on_exp264_tvt_train.py`を追加した。
  - fixed13 compact Stage D共通runnerを実験code対応へ一般化し、
    exp286の既存契約を維持した。
  - by-well p95を設定時のみ追加判定する下流gateを実装した。
  - exp371 + exp286 dedicated tests: `30 passed`。

## Stage D package監査（2026-07-24）

- canonical kernel:
  `kentookumura/exp371-exp333-fixed13-selector-tvt-train`
- title:
  `exp371 exp333 fixed13 selector tvt train`
- notebook:
  - 19 cells（markdown 10 / code 9）
  - output / execution_count: `0 / 0`
  - 全code cell compile: PASS
  - Jupytext round-trip test: PASS
- metadata:
  - private: `true`
  - GPU: `true`
  - machine: `NvidiaTeslaT4`
  - TPU / internet: `false / false`
  - run-on-push: `true`
  - kernel sources: exp371 selector v3 / exp072 cache / exp145
- embedded config:
  - Stage C historical scientific gate: `false`
  - selector gate reclassified: `false`
  - explicit Stage D override: `true`
  - variant / configs / folds: `1 / 3 / 5`
  - planned GPU boosters: `15`
  - control retraining: `false`
  - inference / submission: `false / false`
- SHA:
  - canonical Jupytext source:
    `4b7720db56fc5666ccaddef0417224ddce192b4dafe75fead1f066759cd810f1`
  - canonical ipynb:
    `789c510b97b28b568db9452aae2f1a81e0fbafea2972616fd13819a53d8bc025`
  - packed ipynb:
    `955f90b8b1684351a5ef0fea509495e087d3caf8ba1aa172df730b5464c3a63b`
  - packed config:
    `8eff4af10459a41d8276aef066f03c9692e4b9714ed996f115babe516a2827bd`
  - packed exp371 helper:
    `f6e9fc6a11c807355e38d9832f288b3c9ff3982317c968ae8e338a7e2bc7aeb2`
  - packed fixed13 Stage D runner:
    `db35cba51324241bc7ebe3001527478e7d45c9069c7bb923a4fb73f39ad1c648`
- validation:
  - structure validation: PASS
  - Ruff / py_compile: PASS
  - exp371 + exp286 + Kaggle package tests: `34 passed`

## Kaggle Stage D push拒否（2026-07-24）

- command:
  `kaggle kernels push ... --accelerator NvidiaTeslaT4`
- result:
  `Maximum weekly GPU quota of 45.00 hours reached.`
- Kaggle notebook version作成: `false`
- Stage D booster学習開始: `0 / 15`
- 保存済みcontrol再学習: `0`
- 実行承認の消費: `false`
- 安全処置:
  - local `execution.run_approved=false`
  - local `execution.run_downstream_train=false`
  - 再push前にconfig再有効化とpackage再生成を必須にした。
- fallback監査:
  - Colab Drive rootの既存運用は
    `/content/drive/MyDrive/Kaggle/ROGII`。
  - exp371 Stage Cの25 compact partitionはKaggle kernel outputにあり、
    Colabへ大容量移送が必要。
  - 350 features × 3,783,989 rowsのmatrix構築は通常T4 RAMでOOMリスクが高く、
    現実的fallbackはL4 high-memoryとなる。
  - GPU/runtime変更は結果へ影響し得るため、Kaggle quota reset待ちとの選択を
    ユーザー確認せず実行しない。

## quota回復後の再実行指示（2026-07-25）

- ユーザー指示: 「quota回復しました。実行してください。」
- 実行基盤: canonical Kaggle T4を維持する。Colab fallbackは採用しない。
- 再確認した固定scope:
  - active variant: `1`（`selector_compact_addonly`）
  - LightGBM configs: `3`（indices `0,1,2`）
  - outer folds: `5`
  - 合計新規GPU booster: `1 × 3 × 5 = 15`
  - saved exp264 parent/control再学習: `0`
  - Stage C selector/candidate/PF/HMM再生成: `0`
  - inference / submission: `false / false`
- 元のStage C safety gate FAILは保持し、Stage D独自gateで評価する。
- 前回pushはversion作成前のquota拒否だったため、実行承認は未消費。

## Stage D Kaggle version 1 push（2026-07-25）

- 再package前 validation:
  - structure validation: PASS
  - Jupytext round-trip: PASS
  - py_compile / Ruff: PASS
  - exp371 + exp286 + package tests: `34 passed`
- 再生成package:
  - metadata SHA:
    `8016f95a423373b1265e0cde672b6c256022906aa4dbe42572e72c70d62493e2`
  - embedded config SHA:
    `f8ec43dab853fababcf8c232fe2790ece125386781c196ee29da6ae9a9ec17ba`
  - packed notebook SHA:
    `28b051737f2aefb9f522a11863a00476773a9efdf39155240edb21967e00f9c4`
  - notebook: 19 cells / code 9 / outputs 0 / execution counts 0
- quota回復後の初回push:
  - result: `Kernel push error: Notebook not found`
  - target status: 404
  - target pull: 500
  - own-kernel list: refなしの`[Private Notebook]`、version/session/outputなし
  - 3 kernel sourcesはすべてmetadata pull成功
  - 原因: 2026-07-24のquota拒否でKaggle側に残ったversionless empty shell
- recovery:
  - 削除対象:
    `kentookumura/exp371-exp333-fixed13-selector-tvt-train`
  - 削除内容: version/session/outputを持たない空shellのみ
  - 同じcanonical slugで再作成し、科学contract・notebook・計算量は変更なし
- push:
  - kernel:
    `kentookumura/exp371-exp333-fixed13-selector-tvt-train`
  - version: `1`
  - id_no: `128524177`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp371-exp333-fixed13-selector-tvt-train`
  - pushed at: `2026-07-25 09:07:17 JST`
  - status: `KernelWorkerStatus.RUNNING`
- pull-back:
  - private / T4 / TPU off / internet off: PASS
  - competition source: ROGII: PASS
  - kernel sources: exp371 Stage C / exp072 / exp145: PASS
  - pack / pull cell source SHA:
    `bfc24bbfdf6ef2d01941c53e1ba5614407560e22720ddfd220f6a977db3e9f93`
  - 19/19 cell source一致、output 0: PASS
- push後:
  - local `execution.run_approved=false`
  - local `execution.run_downstream_train=false`
  - 実行承認を1回分消費
  - 約45秒後もstatusは`KernelWorkerStatus.RUNNING`
  - CLI logsは実行中仕様どおり空。空ログを失敗根拠にせず、再pushしない。

## Stage D Kaggle version 1 完了結果（2026-07-25）

- ユーザー完了連絡後、canonical kernelのstatus、通常logs、小容量評価artifactを確認した。
- kernel:
  `kentookumura/exp371-exp333-fixed13-selector-tvt-train`
- version / id_no: `1 / 128524177`
- final status: `KernelWorkerStatus.COMPLETE`
- runtime: `13619.488220 sec`
- 実行量:
  - active variant: `1`（`selector_compact_addonly`）
  - LightGBM configs / folds: `3 / 5`
  - GPU boosters: `15 / 15`
  - saved parent/control retraining: `0`
  - selector/candidate/PF/HMM再生成: `0`
  - inference / submission: `false / false`
- feature surface:
  - clean base: `273`
  - fixed13 compact: `77`
  - final: `350`
  - rows / wells: `3,783,989 / 773`
- pooled:
  - saved parent12 compact add-only: `8.460811237612477`
  - fixed13 compact add-only: `8.369996236751339`
  - delta: `-0.090815000861138 ft`
- fold:
  - 改善: `3 / 5`
  - fold 0--4 delta:
    `+0.077892843 / +0.047410496 / -0.295060606 / -0.091289173 / -0.197859075 ft`
- distance / hidden-like:
  - near 0--250: `-0.061537485 ft`
  - mid 250--1000: `-0.056552715 ft`
  - 1000+: `-0.098307744 ft`
  - spatial / typewell-purged:
    `-0.303915188 / -0.303477245 ft`
- well safety:
  - improved / worsened: `389 / 384`
  - median delta: `-0.002905860 ft`
  - p95 delta: `+1.179312073 ft`、上限`+0.25 ft`をFAIL
  - worst `e25f1537`: `4.706826985 -> 9.344426420`、
    `+4.637599435 ft`、上限`+0.25 ft`をFAIL
- Stage D gate:
  - pooled / 3 folds / near / 1000+ / hidden-like: PASS
  - by-well p95 / worst well: FAIL
  - overall: FAIL
- historical Stage C:
  - safety gate: FAILのまま
  - reclassified: `false`
- final decision:
  `STAGE_D_MEAN_IMPROVED_TAIL_GATE_FAILED_CLOSE_NO_INFERENCE`
- 取得方針:
  - 104 MB OOFと15 model本体を含むoutput archive全体は取得しなかった。
  - metrics、fold、bucket、hidden-like、by-well、feature importance、
    model / reproducibility manifestだけを選択取得した。
  - 保存先:
    `kaggle/output/tvt_train_v1/artifacts/`
- 主要SHA:
  - Stage D metrics:
    `b1c9335ac0c4f5558ee43741fc14c8f2bc5e7921ba6803624494d4380d82b0de`
  - model manifest:
    `151d7a06eeb0dcce157faf36f4613b3f3704a34d167c6e39f3eaa267426ba8e1`
  - OOF prediction:
    `272325effac930cab0ff944ec9ed493a3ff2dceb4ae2a4844d482f99fc20ad3c`
  - reproducibility manifest:
    `e2f39868333c3c17d81ab82c20541ad004f4971dc12b3e1d151d98c8b76a4238`
  - fold metrics:
    `1c65b6d189510fff7734c6c575f3465e80d74562c458d1491857d8a0894fc59f`
  - parent fold comparison:
    `850eaf40c25dbfcb83d3cf1c5abe2595e78701e16ce9141f77a0d412aac30661`
  - bucket comparison:
    `43fea4396cfc927d0009feef1ee1974ab92bf8980d920c0c4ab2fce64cb0323b`
  - hidden-like comparison:
    `55a0263f0a95700221dab132d70a4df2b84d77eade332629164e30714af500f3`
  - by-well comparison:
    `b26daa7700fad05618435095a196238c310f3ebd23f95b8fc170b24cc71f87e7`
- 結論:
  - exp333由来compact signalは下流TVTの平均を改善した。
  - ただしStage Cと同様、well間tail regressionを安全上限内に抑えられなかった。
  - 同一OOFでweight、threshold、feature、gateを救済調整しない。
  - current-test inferenceとsubmissionへ進めず、fixed13 branchを閉じる。
