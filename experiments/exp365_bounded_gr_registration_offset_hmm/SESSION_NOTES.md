# exp365_bounded_gr_registration_offset_hmm セッションノート

## 目的

物理位置とGR登録ずれを分離したbounded-offset HMMが、known-prefix上で実GR固有の
予測信号を持つかを判定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_gate_failed_closed`
- CV / LB: なし
- compact self-contained train / fail-closed inference: 実装済み
- 正規train / inference Notebook: compact候補を採用済み
- Kaggle Stage 0: private CPU version 2 `COMPLETE`、technical PASS /
  scientific FAIL。
- Stage 1 / inference / submission: 不適格・未実行。

## コマンドログ

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp365_bounded_gr_registration_offset_hmm
make new-exp EXP=exp365_bounded_gr_registration_offset_hmm
```

### 2026-07-25 Stage 0実装

- ユーザーの`exp365を実装してください`を、設計済みStage 0だけの実装承認として記録した。
- Jupytext percent形式のcompact self-contained trainを実装し、正規train Notebookへ採用した。
- inferenceはStage 1未実装を明示して停止するfail-closed構成に置換した。
- この実装時点では`config.yaml`のKaggle package / push / execution、
  `run_stage_0`、`run_stage_1`、inference、submissionをfalseのままにし、
  Kaggle実行は行わなかった。
- 実行量はdiagnostic 1、offset state 5、reporting fold 5、
  resource projection well 16、exact-HMM well-run 0、LightGBM config 0、
  trained fold 0、booster 0、parent control rerun 0。

### 2026-07-25 Kaggle Stage 0実行承認

- ユーザーの`実行してください`を、正規train Notebookのprivate Kaggle CPU
  package / push / run承認として`2026-07-25 11:49:15 JST`に記録した。
- 実行対象はdiagnostic 1、offset state 5、reporting fold 5、
  resource projection well 16。
- exact-HMM well-run 0、LightGBM config 0、trained fold 0、booster 0、
  parent control rerun 0、GPU 0。
- Stage 1 exact HMM、inference、submissionは未実装・未承認。
- canonical kernelは
  `kentookumura/exp365-bounded-gr-registration-offset-hmm-train`
  (`exp365 bounded gr registration offset hmm train`)。
- push前のcanonical pullは403、kernel list照会は`Not found`。別slugは作らず、
  canonical id/titleを正として初回pushする。
- 実行承認フラグをtrueへ切り替えた直後、未承認を期待する旧専用test 1件がFAILした。
  科学contractではなくlifecycle expectationの不一致なので、承認済み3 flagsと
  Stage 1/inference無効を検証するtestへ更新した。
- `task prepare-kaggle-notebooks`は`task: command not found`で未実行。
  repoの同等`make prepare-kaggle-notebooks`へ切り替える。
- 更新後の専用testは`9 passed`、`make validate-template`と
  `make validate-exp EXP=exp365_bounded_gr_registration_offset_hmm`はpass。
- canonical packageは正規22 cellsへsupport bootstrap 1 cellを加えた23 cells。
  outputs / execution countは0。
- metadataはprivate CPU、GPU / TPU / internet off、run-on-push true、
  competition source 1、追加kernel/dataset/model source 0。
- support bootstrap 24 filesを監査し、source / loose package / bootstrap内
  `config.yaml` SHAは
  `ec2d4b2fcdc246d77287a536dc8fa7c8ed55ba35c9259afe25b24921bd90f78a`
  で一致した。
- package Notebook SHA:
  `4233f8a1f683cd65efb465ca94e53ea15e47e3b5a6cea05be1f68b10cbe9b108`
- kernel metadata SHA:
  `f16cb301b4f7be412b05dbff32a1ffdf11ad1d3a236daa9766951d9bd84fcef5`
- bootstrap ZIP SHA:
  `8280be20d750d6929bef5a2d028bc6dca12fb088bce2df6f3e4585d4c70fa9b4`
- canonical kernel version 1をpushし、URL
  `https://www.kaggle.com/code/kentookumura/exp365-bounded-gr-registration-offset-hmm-train`
  を取得した。
- push後pullでprivate CPU / GPU・TPU・internet off、competition source 1を再確認した。
  Kaggle kernel id_noは`128537562`。

### Kaggle version 1

- status `COMPLETE`、Stage 0本体`352.181967 sec`。
- 773 wells / 18,465 rolling windows / 915,301 observed held-out rowsを評価した。
- suffix truth read、physical prediction row、exact-HMM runはすべて0。
- real predictive NLL gainは`0.054304`で1% gateをPASSしたが、circular control側が
  `0.153114`改善し、real-minus-circularは`-0.098810`でFAIL。
- fold別real gainは`0.050687`から`0.056661`で全fold正だったが、
  real-minus-circularは全fold負のためpassing fold `0 / 5`。
- nonzero posterior mean `0.489435`、boundary mass `0.182671`はPASS。
  adjacent-window sign agreement `0.580771 < 0.60`でFAIL。
- projected runtime `56429.34 > 30600 sec`でFAIL、projected peak RSS
  `7.358320 GB <= 25 GB`はPASS。
- `posterior_in_unit_interval`だけtechnical false。取得したposteriorではmass値は
  全て`[0,1]`、個別posteriorも全値`[0,1]`で、CSV確率和は
  history `[0.99999999983, 1.0000000001322]`、final
  `[0.999999999869, 1.000000000159]`。in-memoryのmachine-epsilon超過を
  strict `between(0,1)`が拾ったvalidator不具合と判定した。
- 科学判定は`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`で、Stage 1不適格。
- 科学設定・閾値を変えず、probability interval technical checkに`atol=1e-12`
  だけを追加し、同じcanonical slugのversion 2でtechnical判定を正す。

### Kaggle version 2

- 同じcanonical kernelをversion 2としてpushし、status `COMPLETE`まで監視した。
- canonical URL:
  `https://www.kaggle.com/code/kentookumura/exp365-bounded-gr-registration-offset-hmm-train`
- packageは23 cells、outputs / execution count 0。private CPU、GPU / TPU /
  internet off、run-on-push true、competition source 1を維持した。
- source / loose package / embedded bootstrap内`config.yaml` SHAは
  `f3f3459de6a98f096a62a34dddb1b31e95be3c15cda220dbe0960c71d8047cb9`。
- package Notebook SHA:
  `dbbf118f4dcf59b4955ced0e397a47c9b261e74a21596a9f8796a18490ca0c8c`
- kernel metadata SHA:
  `f16cb301b4f7be412b05dbff32a1ffdf11ad1d3a236daa9766951d9bd84fcef5`
- embedded support ZIP SHA:
  `4ba32df78fa16674d4b60dfc817effbd982f6baa7c5a9667863e997132803b8a`
- compact train source SHA:
  `a42a07369491ef842b20afb144d887d0f92a67689e54f29527e180f848725833`
- Stage 0本体`408.720069 sec`、773 wells / 18,465 windows /
  915,301 observed held-out rows。
- technical gateは全項目PASS。`posterior_in_unit_interval`もtrueへ復旧した。
- scientific値はversion 1と同一。real NLL gain`0.054304`はPASSしたが、
  circular NLL gain`0.153114`、real-minus-circular`-0.098810`、
  passing folds`0 / 5`、sign agreement`0.580771`、projected runtime
  `56429.34 sec`のためscientific FAIL。
- contract、input manifest、rolling ledger、delta posterior、resource projection、
  fold metricsはversion 1とraw SHAが全て一致した。technical gate reportと
  elapsed timeを含むsummaryだけが変更された。
- 取得したfreeze対象5ファイルはraw/content SHAがmanifestと一致した。
  contract SHAは
  `83a15a82a966a44837be5f7c22dece5160c2324e112949bab66895f39b7225d9`、
  rolling content SHAは
  `58e0db25485cef614f002b200e8123922a0d50156c968168d4fbd550d52d3896`、
  posterior content SHAは
  `0b6715000c1a8ff74df41340594c0e0a49b4e2dac7b5467ceb5c5497c6e7f711`。
- posterior CSVは18,465 rows、全値finiteかつ`[0,1]`。history確率和は
  `[0.99999999983, 1.0000000001322]`、final確率和は
  `[0.9999999998690001, 1.000000000159]`。
- 最終判定は`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`。Stage 1は未実装・不適格、
  inference / submissionは未実行。

## 実装した固定契約

- Stage 0はraw trainの`MD/Z/GR/TVT_input`とtypewell `TVT/GR`だけを読む。
  horizontalのsuffix truth `TVT`はusecolsに含めない。
- known prefix 256行以上を対象に、128 history / 64 held-out / stride 64でrollingする。
- historyだけからexp209と同じstd式のsigmaを計算し`[10,60]`へclipする。
  history内missing GRはhistory内interpolation、held-outはraw finite GRだけを逐次予測・更新する。
- deltaは`[-6,-3,0,3,6] ft`、priorは`[.05,.15,.60,.15,.05]`。
  隣接cellへ方向ごと`1/512`、境界の無効massはstayへ戻す。
- normalized Gaussian predictive NLLをdelta=0と比較する。emission lookupは
  `TVT_input + delta`、物理predictionは生成しない。
- circular controlはwell内のobserved known-prefix GRだけを64要素rotateし、
  missing maskと観測値multisetを固定する。
- Stage 0生成物はrolling ledger、history/final delta posterior、safe input manifest、
  16-well resource projection。suffix truthなしでcontent SHAをfreeze・readbackする。
- foldはstable well orderのGroupKFold 5分割。fold passはNLL gain`>=1%`かつ
  real-minus-circular gain`>=0.5%`。
- exact-HMM runtimeは採用済みexp209 v5 `11285.868 sec`を5 offset statesで
  `5.0x`する固定projection。投影値`56429.34 sec`は実行前から30,600秒上限を超えるが、
  係数・state数・gateを変更して救済しない。
- peak RSSは実際のposition/rate/suffix shapeを16 workload分位wellで見積もる。

## Notebook構成比較

- 親exp209にはcompact self-contained版がないため、通常train sourceを比較対象にした。
- 親は6章・174行、exp365 compact trainは10章・約1,500行。
  Stage 0のinput、filter、rolling、resource、freeze、gate、保存処理をNotebook上へ展開した。
- 同一exp helper importと`__file__`は使用していない。

## 静的検証

- `py_compile`: train / inferenceともpass。
- `ruff --select F821`: train / inference / testともpass。
- `pytest -q experiments/exp365_bounded_gr_registration_offset_hmm/tests/test_exp365_bounded_gr_registration_offset_hmm.py`:
  version 2修正後`10 passed`。
- `jupytext --to ipynb --test`: train / inferenceともpass。
- `make validate-exp EXP=exp365_bounded_gr_registration_offset_hmm`:
  strict pass。
- `make validate-template`: pass。
- `make test`: exp365を含む`1009 passed / 7 skipped`、既存契約test 3件だけFAIL。
  exp296の実行後status / run flagと旧test期待値の不一致2件、exp393のGPU session
  blocked statusと旧test期待値の不一致1件であり、exp365変更対象外。

## 実装SHA

- compact train source:
  `a42a07369491ef842b20afb144d887d0f92a67689e54f29527e180f848725833`
- compact train Notebook:
  `2d07b7ded0b1b8891354f300ec832628a6d24a723d3eb655df6626f25ee9c8e9`
- 正規train Notebook:
  `275c7063f75732bd627b49beec5e90868b8a29be01aea8a4417410cb4ece4cf4`
- fail-closed inference source:
  `739f5e034ae2f2f72fc72e5e08105af0904c0b1244d8b256721b389497017fca`
- compact inference Notebook:
  `f76b1b90f47d21f991c3e18585a5965da343afe2171213c8895319025e16371c`
- 正規inference Notebook:
  `2207e921fd05dd6a8c9c9cf4fce5eae5b1efc4ab673afdf2eb5c873a83a2054b`
- dedicated test:
  `b11bca4ee65e7c69ea9a365a6b40c47dbce768aecac1b7f0c7b962e037f24cd7`
- config:
  `57d222a96dda96190b6b426c4e239054657329166a15befc4f724b60507c8e5e`
- train Notebookはcompact / 正規とも22 cells、inferenceは8 cells。outputs /
  execution countはいずれも0。

## 変更点

- delta 5状態、zero-centered prior、adjacent transition、emission/outputの役割を固定した。
- Stage 0をsuffix truth不要のknown-prefix rolling-originに固定した。
- Stage 1は1 variant / 5 folds / 773 HMM runs / booster 0 / control rerun 0。

## 再現性メモ

- seed policy: RNGなし、well / row / delta順を固定。
- stochastic components: なし。
- CPU/GPU: Kaggle CPU single worker、GPU off、上限30,600秒 / 25GBで実行。
- SHA: rolling-window ledger、delta posterior、prediction content SHA。gzipはdecompressed SHA。
- kernel version 2。physical prediction / submissionは未生成。

## 次のアクション

1. 本branchは救済せず閉じる。
2. Stage 1 exact HMM、inference、submissionは実装・実行しない。
3. 次候補は同じdelta grid / transition / sigma / controlの閾値調整ではなく、
   独立仮説から選ぶ。
