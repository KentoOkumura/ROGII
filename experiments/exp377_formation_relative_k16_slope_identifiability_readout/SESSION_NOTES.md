# exp377_formation_relative_k16_slope_identifiability_readout セッションノート

## 目的

formation-relative K16勾配がouter-validへ外挿可能かを、HMM/PF/MLを入れる前に切り分ける。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU v2実行承認済み・共通support report-only修正・push前検証中
- CV: Stage 0 fail-closedによりtruth scoringなし
- LB: まだなし
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 負の比較参照: `exp376_exp226_formation_conditioned_k16_donor_kernel`

## 承認境界

2026-07-24のユーザー指示「exp377を実装してください」を、以下のimplementation-only承認として受領した。

- Stage 0 integrityと、Stage 0 PASS時だけ動くStage 1 identifiabilityを実装する。
- Jupytext起点の別名compact self-contained train/inference候補を作る。
- 合成データtest、Jupytext、構文、Ruff、実験validatorまで確認する。
- 既存の正規train/inference Notebookは上書きしない。
- Kaggle package/push/run、current-test生成、inference、submissionは行わない。

2026-07-24のユーザー指示「実行してください」を、以下の追加承認として受領した。

- compact self-contained train候補を正規train Notebookへ採用する。
- scientific diagnostic 1 / reporting surfaces 6 / reporting folds 5をKaggle CPUで1回実行する。
- model config / trained fold / LightGBM booster / HMM / PF / parent-control再実行はすべて0のまま固定する。
- Kaggle package、push、完了監視、train-side Stage 0/1評価、必要な小規模成果物の確認と実験記録更新まで行う。
- inference、submission、current-test生成、exp378以降の自動実装は承認範囲に含めない。

正規kernel id/titleはKaggleの50文字slug上限を踏まえて
`kentookumura/exp377-formation-relative-k16-slope-readout-train` /
`exp377 formation relative k16 slope readout train`とする。完全な実験名由来slugは65文字で上限を超えるため、
`identifiability`だけを省略し、実験番号・formation-relative K16 slope・readoutの意味を保持した
49文字のslugに短縮する。

## 固定した変更点

- 構造座標を`S=TVT+Z`とする。
- exp226と同じouter 5-fold、K16 `numpy.linspace + searchsorted(side="left")`、方位`118.4°`、`|projection|>0.3`、XY最近傍50、bandwidth 500 ft、ridge 1を固定する。
- outer-train donorの各K16区間で、finiteかつ正の`ΔMD` step rateのmedianとして`dS/dMD`と6種類の`d(S-F_f)/dMD`を計算する。
- outer-train坑井ごとのXY・6地層面medianから`FormationPlaneKNN(k=10)`を作り、outer-valid K16区間両端の面差から`dF_f/dMD`を計算する。
- 各面の復元rateは`XYKernel[d(S-F_f)/dMD / projection] * target_projection + dF_f/dMD`とする。
- 面順は`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`、primaryは6系列すべてがfiniteな行のmedianに事前固定する。
- best formationのpost-hoc選択、formation/K/bandwidth/grid救済を禁止する。

## leakage / freeze境界

- exp226 OOFは`well_id,row_idx,suffix_offset,fold`だけをpre-freezeで読む。保存OOFのdecompressed SHAは`709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`に固定する。
- outer-validのtarget-free objectは`X,Y,Z,MD,TVT_input`だけを持ち、`TVT`と6地層列を持たない。
- foldごとにouter-train donorだけの`TVT`と6地層列を読む。source/valid集合は分離する。
- fold manifest、role read ledger、donor relative field、segment schedule、primary pathを保存し、logical/decompressed SHAをfreezeする。
- Stage 0が不合格ならtruth readerを呼ばない。合格時だけfreeze SHA一致確認後に`TVT`を別readerでlate joinする。
- raw horizontal file SHAは入力同一性のため記録するが、pre-freezeの数値処理でvalid `TVT`/formation列をparseしない。

## scope定義

設計scaffoldの`h512 / long401 / clean273`には実装可能な定義が揃っていなかったため、次のように明示した。

- `h512`: 未知suffix先頭512行。
- `long401`: 未知suffix長が401行以上の坑井の全未知suffix行。
- `clean273`: このrepoでは273特徴allowlistであり行集合ではない。本readoutはML特徴を使わないため、架空の坑井集合を作らずpooled契約の別名とする。

判定処理は`config.yaml`のscope kindを読む。target由来のscopeは作らない。

## 実装内容

- train候補:
  `exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_train.py/.ipynb`
- inference候補:
  `exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_inference.py/.ipynb`
- 専用test:
  `tests/test_exp377_formation_relative_k16_slope_identifiability_readout.py`
- train候補は同一exp helper importなしのself-contained構成で、2,484行・9章・20セル。
- inference候補はdiagnostic-only契約を確認後、必ず`RuntimeError`で停止し、生成物を作らない。
- 正規train Notebookはユーザーの実行指示に基づきcompact self-contained候補から採用した。
- 正規inference Notebookは変更せず、config上もinference/submissionをfail-closedのまま維持する。

## 固定gate

### Stage 0 integrity

- 3,783,989 rows / 773 wells / 12,368 K16 segments / 5 fold runs。
- outer-valid truth read 0、formation read 0、source/valid overlap 0。
- FormationPlaneKNN fallback率`<=5%`。
- primary finite coverage`>=98%`。
- XY kernel effective donor数p05`>=10`。
- 全freeze artifactのlogical SHA一致。

### Stage 1 identifiability

- primary segment rate RMSEのdirect baseline比relative gain`>=5%`。
- primary cumulative path RMSE gain`>=0.50 ft`。
- rate/pathとも4/5 folds改善。
- `h512 / long401 / clean273` path悪化が各`<=0.02 ft`。
- by-well delta p95`<=0 ft`、worst`<=+0.25 ft`。
- 1条件でも不合格ならexp378/379/380を停止し、surface/kernel/scope救済を行わない。

## 実行量

- scientific diagnostic: 1
- reporting surfaces: 6
- reporting folds: 5
- model configs / trained folds / LightGBM boosters: `0 / 0 / 0`
- HMM / PF runs: `0 / 0`
- parent/control再実行: 0
- GPU: false
- inference / submission: false

## Kaggle CPU v1 preflight

- execution start: `2026-07-24T08:19:57Z`
- kernel: `kentookumura/exp377-formation-relative-k16-slope-readout-train`
- title: `exp377 formation relative k16 slope readout train`
- metadata: private / CPU / internet off / `run_on_push=true`
- competition source: `rogii-wellbore-geology-prediction`
- notebook source: `kentookumura/exp226-k16-kappa-repro-train`
- 正規train Notebook: 20 cells、packaged Notebook: bootstrap込み21 cells
- packaged Notebookのbootstrap後20 cellsは正規train Notebookと一致した。
- source / packaged `config.yaml` SHAは一致した。
- source / packaged compact self-contained train `.py` SHAは一致した。
- bootstrapを一時ディレクトリへ展開し、埋め込み`config.yaml`とtrain `.py`のSHAがpackage側と一致することを確認した。
- source train `.py` SHA: `bcb2bffc50768cf2db728c1ce63a03a1ca4a5b8fa2abfe3a15aa6c6fec337c74`
- preflight時点のconfig SHA: `1acb42dba68b6f8a452cbd5c1ffca1786753de217eff6a8b2f71f960ade5f10e`
- push時の最終config / package / bootstrap SHA: `e12b2fd9f6d2d0ad360cdaa369d164b38d918cf55b21a3d6943f11134f3b92aa`
- packaged Notebook SHA: `58ef3296e259c22d50578c605c1a502b0062e48e7babde04b2b3004a02c89bd8`
- `py_compile`: PASS
- Ruff F821: PASS
- Jupytext round-trip: PASS
- 専用test + scaffold + Kaggle Notebook test: `19 passed`
- strict experiment validation: PASS
- template validation: PASS
- ローカルNotebook実行: 未実施

実行開始状態をconfigへ記録した後にpackageを再生成し、最終config SHAとbootstrap parityを再確認してからpushした。

## Kaggle CPU v1 push

- push: 成功
- kernel version: `1`
- kernel id_no: `128452991`
- URL: `https://www.kaggle.com/code/kentookumura/exp377-formation-relative-k16-slope-readout-train`
- push直後に同じkernel idを`kaggle kernels pull -m`で取得し、存在を確認した。
- Kaggle側metadataでもtitle/slug一致、private、CPU、internet off、competition source、exp226 kernel sourceを確認した。
- pull metadataの`machine_shape`はCPU実行を表す`None`だった。
- 1回実行承認は消費済みとし、同kernelへの自動再pushは行わない。

## Kaggle CPU v1結果

- Kaggle status: `COMPLETE`
- summary completed_at: `2026-07-24T08:28:14.452812+00:00`
- core audit: `436.170447442 sec`
- final log timestamp: `446.345942703 sec`
- 技術実行: PASS
- Stage 0 integrity: **FAIL**
- Stage 1 identifiability: 未実行
- truth join: 未実行
- decision: `close_before_truth_join_and_block_exp378_exp379_exp380_without_rescue_grid`

Stage 0の固定checkは`effective_donors_p05`だけが不合格だった。

| 項目 | 実測 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| rows | 3,783,989 | 3,783,989 | PASS |
| wells | 773 | 773 | PASS |
| segments | 12,368 | 12,368 | PASS |
| outer fold runs | 5 | 5 | PASS |
| primary coverage | 1.0 | >= 0.98 | PASS |
| surface fallback fraction | 0.0 | <= 0.05 | PASS |
| effective donors p05 | **2.59469484575288** | >= 10 | **FAIL** |
| valid truth reads | 0 | 0 | PASS |
| valid formation reads | 0 | 0 | PASS |
| source / valid overlap | 0 | 0 | PASS |

target-free bundle logical SHA:
`944af71f245e5e4615953c7d69fbbb3f22e48757cf63d8474e16d0398a683e5a`

exp226 OOFは事前固定したdecompressed SHA
`709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
と一致し、pre-freezeで読んだ列は`well_id,row_idx,suffix_offset,fold`だけだった。

必要な実ファイル確認のため、full output archiveではなくsummary、Stage 0 guard、
freeze manifest、SHA manifest、fold manifest、role ledger、input manifestだけを
`kaggle/output/train_v1/`へ取得した。SHA manifest対象の取得ファイルはすべて
file SHA一致を確認した。66.6 MBのprimary path、2.3 MBのsegment schedule、
1.2 MBのdonor fieldは取得していない。

## 結果の解釈

最近傍segmentを50件選んでも、500 ft bandwidthとlocal-linear weightを通した
effective supportは下位5%で2.59まで崩れた。行・fold・leakage・formation surface・
coverageは正常なので、実装エラーやtruth leakageではなく、事前固定した
exp226型局所donor supportの非退化条件不成立として扱う。

truthを開かなかったためformation-relative rate/pathのRMSE自体は未評価だった。

その後のコード監査で、direct controlと6 relative fieldは同じeligible donor XY
inventory、近傍50、bandwidth 500 ftを共有し、effective-donor数はtreatmentとcontrolを
区別しないことを確認した。したがってv1のhard gateはcontrolled comparisonそのものを
止めており、formation-relative固有の安全条件としては不適切だった。

## Kaggle CPU v2承認と修正

2026-07-24のユーザー指示「1を実行してください」を、同じcanonical kernelへの
CPU v2 1回実行承認として受領した。

- K16、近傍50、bandwidth 500 ft、ridge 1、方位projectionを変更しない。
- 6 formation-relative系列、median6 primary、Stage 1固定AND gateを変更しない。
- `effective_donors_p05 >= 10`の実測と判定は保存するが、report-only warningとする。
- それ以外のStage 0 integrityは引き続きhard gateとし、不合格ならtruth前に停止する。
- Stage 0 hard gate PASS時だけtruthをlate joinし、direct対relative Stage 1を1回評価する。
- model / HMM / PF / booster / parent-control再生成は各0。
- inference、submission、kernel parameter grid、追加variantは実行しない。

修正後の専用test + scaffold + Kaggle Notebook testは`20 passed`。

## Kaggle CPU v2 preflight

- execution start: `2026-07-24T09:14:22Z`
- kernel: v1と同じ
  `kentookumura/exp377-formation-relative-k16-slope-readout-train`
- target version: `2`
- metadata: private / CPU / internet off / `run_on_push=true`
- competition source: `rogii-wellbore-geology-prediction`
- notebook source: `kentookumura/exp226-k16-kappa-repro-train`
- 実行量: 1 diagnostic / 6 reporting surfaces / 5 folds /
  model・trained fold・booster・HMM・PF・parent-control再生成各0
- K16 / nearest 50 / bandwidth 500 / ridge 1はv1と同一。
- 正規train Notebookは20 cells、packageはbootstrap込み21 cellsで、本体20 cellsは一致。
- source / packaged config SHA:
  `08c4b536093d1614fb0fe07ba1dfcc4419166b5b2dcf3c73c1526c13e80a82cd`
- source / packaged train `.py` SHA:
  `cdf79f3e6c35a718ebe13cfca1bf70a5548a3274af17631d478bdca891fd0018`
- packaged Notebook SHA:
  `e022dbbe27b8361d4c361db1949b114247ed7ccdf921be08f7617ef267f25a99`
- bootstrap一時展開後のconfig/train source SHAもpackageと一致。
- py_compile / full Ruff / Jupytext round-trip / strict experiment validation /
  template validation: PASS
- 専用test + scaffold + Kaggle Notebook test: `20 passed`
- ローカルNotebook実行: 未実施

実行開始状態をconfigへ記録したため、push直前にpackageを再生成し、最終SHAを再確認する。

push直前の最終確認:

- final config / packaged config SHA:
  `676e2c1ebfffe98ab38f8b847eec806861343d7cd626ccde99dc76306f76ded9`
- final packaged Notebook SHA:
  `cd75b0571953d9bd901042b35b6664627c701b4ca0745099d0561be83df31b94`
- final bootstrap config/train source parity: PASS
- final targeted tests: `20 passed`
- 同一kernelをpush前に`kaggle kernels pull -m`し、id_no `128452991`、
  CPU、internet off、exp226 sourceの既存v1を確認した。

## Kaggle CPU v2 push

- push: 成功
- kernel version: `2`
- kernel id_no: `128452991`
- URL: `https://www.kaggle.com/code/kentookumura/exp377-formation-relative-k16-slope-readout-train`
- push直後に同じkernel idをpullし、CPU、internet off、competition source、
  exp226 kernel sourceを再確認した。
- v2の1回実行承認は消費済み。自動再pushは行わない。

## 検証コマンド

```bash
.venv/bin/python -m py_compile \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_train.py
.venv/bin/python -m py_compile \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_train.py \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_inference.py \
  --select F821
.venv/bin/pytest -q \
  tests/test_exp377_formation_relative_k16_slope_identifiability_readout.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp377_formation_relative_k16_slope_identifiability_readout/\
exp377_formation_relative_k16_slope_identifiability_readout_compact_selfcontained_train.py
make validate-exp EXP=exp377_formation_relative_k16_slope_identifiability_readout
make validate-template
```

結果:

- 専用test: `8 passed`
- `py_compile`: train / inferenceともPASS
- Ruff F821: PASS
- Jupytext conversion / round-trip: train / inferenceともPASS
- strict experiment validation: PASS
- template validation: PASS
- `__file__`: compact候補内に0件
- ローカルNotebook実行: 未実施
- `make validate-exp`初回はscaffold由来の`model.name`未定義でFAILした。仮説や実行量を変えず`diagnostic_only_no_fitted_model`を明記し、再実行でPASSした。
- 記録更新後の再validationではREADMEの必須`## 所見`節不足を検出した。実装済み点と未評価点を追記し、最終validationでPASSした。
- 専用testに共通scaffold/Notebook testを加えた対象回帰は`19 passed`。

## Notebook構成比較

直前の同系実装exp376 compact self-contained trainは4,007行・9章だった。exp377はcandidate bank、GR correction、U-projectionを持たない0-HMM rate/path readoutなので、役割を欠落させず2,484行・9章へ縮小した。Imports、runtime/SHA、K16/formation/kernel、role-safe input、fold-local生成、Stage 0、truth late join/Stage 1、生成物、execution orchestrationをNotebook上で追跡できる。同一exp helper importと`__file__`はない。

## 再現性メモ

- seed policy: no RNG、fold/well/segment/row順を固定。
- stable tie order: distance、donor well id、donor segment id。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU v1を実行、GPUなし。
- gzip: raw gzip SHAに加えdecompressed content SHAとlogical content SHAを記録する。
- model manifest / model SHA: fitted model 0のため非該当。
- prediction SHA: target-free segment/path readoutのlogical/decompressed SHAを記録する。
- submission SHA: inference/submission禁止のため非該当。
- deterministic anchor: 成功rerun一致未確認のためfalse。

## 次のアクション

1. exp377をscientific negativeとして終了する。
2. exp378 / exp379 / exp380 / exp382を未実装・未実行のまま閉じる。
3. surface / kernel / scopeのposthoc救済、inference、submissionを行わない。

## Kaggle CPU v2結果

- status: `COMPLETE`
- completed_at: `2026-07-24T09:24:09.503834+00:00`
- log last timestamp: `489.190654601 sec`
- Stage 0: PASS
  - rows `3,783,989` / wells `773` / K16 segments `12,368` / folds `5`
  - primary coverage `1.0`
  - surface fallback `0.0`
  - valid truth/formation pre-freeze read `0 / 0`
  - source-valid overlap `0`
  - effective donors p05 `2.59469484575288`: fixed numeric checkはFAILだが
    shared-kernel report-only warning
  - target-free bundle logical SHA:
    `944af71f245e5e4615953c7d69fbbb3f22e48757cf63d8474e16d0398a683e5a`
- truth late join: 実行
- Stage 1: FAIL、7 checks中PASS 0
  - segment rate RMSE: direct `0.012300807400595496` →
    median6 `0.03845360562025232`
  - rate relative gain: `-2.1261041952734527`
  - cumulative path RMSE: direct `16.100131165038366 ft` →
    median6 `38.776238158630434 ft`
  - path gain: `-22.676106993592068 ft`
  - rate/path positive folds: `0/5` / `0/5`
  - H512 delta: `+3.3760026050730767 ft`
  - 164/773 wells改善、609/773 wells悪化
  - median / p95 well delta: `+6.597140518306955 /
    +49.43456225293919 ft`
  - worst: `a247e7cf`, `+408.0446864027287 ft`
- 個別6 formation path RMSEも`39.022186--40.355628 ft`で、
  direct `16.100131 ft`よりすべて悪化した。
- decision:
  `close_and_block_exp378_exp379_exp380_without_surface_kernel_or_scope_rescue`

技術guardとlate-join契約は正常で、target-free bundle SHAもv1と一致した。
negative resultはv2修正やtarget-free生成変更ではなく、固定したformation-relative
分解そのものの科学的失敗と判断する。median6集約だけでなく個別6面も全悪化しているため、
posthoc best-surface選択へ進む根拠はない。

選択取得したv2出力は
`kaggle/output/train_v2/`へ保存し、SHA manifest対象の取得ファイルはすべて照合PASS。
大容量のdonor field / segment schedule / primary path scheduleは、評価に必要な
metrics・guard・manifest確認だけで足りるため取得していない。

## v2確定SHA

- final config/package/bootstrap:
  `676e2c1ebfffe98ab38f8b847eec806861343d7cd626ccde99dc76306f76ded9`
- train source/package/bootstrap:
  `cdf79f3e6c35a718ebe13cfca1bf70a5548a3274af17631d478bdca891fd0018`
- packaged Notebook:
  `cd75b0571953d9bd901042b35b6664627c701b4ca0745099d0561be83df31b94`
- truth manifest logical:
  `6cf3ad2ad64f075189e9f162258f9bd727a80be197df51addca8a5a66760c19e`
- segment actual logical:
  `ef644b7c9e404c2925950c727ca7e8d4f1721361c475344a8c4ab25b7c00012c`
