# exp267_well_segment_candidate_divergence_signature_cluster_on_exp265 セッションノート

## 目的

exp265の512-row block clusterがterminal partial blockとraw MD spanに支配された失敗を受け、
6 primitive candidateのwell内序盤・中盤・終盤の広がりだけを18次元target-free署名へ縮約する。
stableなwell clusterとexp264 selector add-onlyの前提を、学習前に0 boosterで監査する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage A Kaggle CPU version 2完了・guard FAIL・branch closed
- CV / LB / submission: Stage Aでは対象外
- inference: disabled

## 実行量契約

- Stage A: 0 variant、0 LightGBM config、0 fold training、0 booster。
- 親/control再学習: 0。
- runtime: Kaggle CPU、GPU/TPU/internet off。
- 入力: exp263 Stage 0 candidate cache、exp264 Stage B score/model manifest/metrics。
- conditional Stage B: disabled。1 variant × 2 objectives × 5 folds = 10 CPU boosters。
- downstream 15 GPU boosters、inference、submission: disabled。

## 設計判断

- `evaluation_progress = eval_position / eval_len`をexp264と揃え、fixed thirdsへ分割する。
- 15-pair gapは区間内の全raw row×15 pairを同じ重みでmean/p90集約する。
- empty/非finite segmentはNaNとcoverage/fallbackを保存し、cluster入力だけouter-train median補完する。
- soft membershipはexp265と同じ`softmax(-distance / temperature)`。temperatureはouter-trainの
  正の最近傍centroid距離中央値。
- cluster semanticは18 clipped robust-scaled centroid特徴の平均でlow/middle/highへ並べる。
- candidate winner guardはlow/middle/high winner patternが4/5 foldsで一致し、pattern内に
  2候補以上あること。calibration guardはhigh-low bias差の符号が4/5 foldsで一致すること。
- worst clusterは最高actual-error wellを1本除いてもpooled worstのままであることを要求する。

## 再現性

- `docs/06_reproducibility.md`確認済み。
- seed: primary `42 + outer_fold`、audit `10042 + outer_fold`。
- stochastic component: KMeans initializationのみ。
- global RNGは使わず、sklearn `random_state`を明示する。
- exp263 manifest/catalog、exp264 score/model manifest/metrics SHAをfail-closed照合する。
- 18-feature schema、well signature logical content、preprocessor、centroid/assignment SHAを保存する。
- Stage Aではmodel、prediction、submission SHAは対象外。
- Kaggle rerun前はdeterministic anchorと呼ばない。

## 実装

- `docs/legacy/steering/20260717-exp267-well-segment-candidate-divergence-signature-cluster-on-exp265/`
  に仮説、18特徴、fold-safe処理、guard、禁止事項を固定した。
- `src/well_segment_candidate_divergence.py`へsignature、outer-fold KMeans、semantic soft membership、
  stability、post-assignment Parquet streaming、生成物/SHA保存を実装した。
- exp264 scoreは6 primitiveが各3,783,989 rows、合計773 wellsを完全に覆わない場合もfail-closedにした。
- train Jupytext sourceは7章構成で、入力/SHA、18署名、cluster、score監査、生成物を展開した。
- inference Jupytext sourceはdisabled guardだけを持ち、submissionを生成しない。
- targeted unit testは4件。18特徴/common translation不変性、3区間coverage、fold-safe deterministic
  cluster/soft probability、post-assignment score guard、0-booster契約を検証する。
- 親exp265 trainは323行/7章、本実験trainは351行/7章で、同じrole slotを維持した。

## コマンドログ

### 2026-07-17 作成・実装

```bash
make new-steering EXP=exp267_well_segment_candidate_divergence_signature_cluster_on_exp265
make new-exp EXP=exp267_well_segment_candidate_divergence_signature_cluster_on_exp265
.venv/bin/ruff check src/well_segment_candidate_divergence.py tests/test_well_segment_candidate_divergence.py
.venv/bin/pytest -q tests/test_well_segment_candidate_divergence.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py/inference.py>
```

- targeted tests 4件: PASS。
- repository全84 tests: PASS。
- py_compile、Ruff F821、Jupytext train/inference round-trip、strict experiment validation、
  template validation: PASS。
- local notebook実行と実データStage Aは行っていない。最初の実行はKaggle CPUを正とする。

## 次のアクション

1. conditional Stage B、downstream GPU、inference、submissionへ進まない。
2. K=3 / clip / segment / feature gridによる事後救済を行わない。
3. 再訪する場合は保存済み18署名を連続量で監査する独立0-booster backlogから開始する。

## Kaggle package監査

- run-on-push offでtrain packageを生成した。Kaggleへのpush、外部実行はしていない。
- kernel id / title:
  - `kentookumura/exp267-well-segment-divergence-cluster-train`
  - `exp267 well segment divergence cluster train`
- metadataはprivate CPU、GPU/TPU/internet off、competition source 1、kernel source 2、
  `run_on_push=false`。
- package configは`run_approved=false`、Stage A 0 booster、conditional Stage B / inference disabled。
- canonical / loose package / bootstrap ZIPのSHA一致:
  - config: `bcac627b...67fd0`
  - candidate contract: `ca0350c9...b49e`
  - settings: `04d7e95c...ba22d`
  - `src/well_segment_candidate_divergence.py`: `48663f43...2e27d`
- bootstrap manifest / ZIPは14 entriesで一致。canonical 17 cellsとpackageのbootstrap後17 cellsも一致。
- canonical notebook SHAは`94623e4d...5d65f`、bootstrap付きpackage notebook SHAは
  `bd05a0ed...cbeed`。

## Stage A Kaggle v1実行承認

- 2026-07-17、ユーザーの「実行してください」によりStage A Kaggle CPU v1の実行承認を得た。
- 実行契約を再確認した: 0 variant / 0 LightGBM config / 0 trained fold / 0 booster、
  親/control再学習0、GPU/TPU/internet off。
- conditional Stage B 10 CPU boosters、downstream GPU、inference、submissionは未承認・disabledを維持する。
- credential checkerはOAuth credentialとlegacy credentialを利用可能と確認した。token実値は記録していない。
- canonical exp267 kernelの事前pullは403でresource未作成。比較として既存exp266 canonical kernelは
  id_no `127531300`、private CPU、GPU/TPU/internet offでpull成功し、認証とexp266完了状態を確認した。
- run-on-push onのstrict packageをcanonical id/titleで再生成した。metadataはprivate CPU、
  GPU/TPU/internet off、competition source 1、kernel source 2、`run_on_push=true`。
- package configは`run_approved=true`、conditional Stage B / inferenceはdisabledのまま。
- 再生成後のcanonical / loose package / bootstrap ZIP監査はPASS:
  - config SHA: `bfaf7ac8...11a53`
  - candidate contract SHA: `ca0350c9...b49e`
  - settings SHA: `04d7e95c...ba22d`
  - `src/well_segment_candidate_divergence.py` SHA: `48663f43...2e27d`
  - canonical notebook SHA: `94623e4d...5d65f`
  - bootstrap付きpackage notebook SHA: `4e2add53...5ced`
  - bootstrap manifest / ZIP / loose files 14 entries一致、canonical 17 cellsとpackage bootstrap後17 cells一致。
- `make push-kaggle-train EXP=exp267_well_segment_candidate_divergence_signature_cluster_on_exp265`
  を実行し、canonical kernel version 1のpushに成功した。
- kernel URL: `https://www.kaggle.com/code/kentookumura/exp267-well-segment-divergence-cluster-train`

## Stage A Kaggle version 1入力ERRORと修復

- Kaggle kernel version 1 / id_no `127573486`はbootstrap後の入力解決でERRORとなった。
  `candidate_score_oof.parquet`が見つからず、signature生成、KMeans、score集計、生成物保存には到達していない。
- 原因はexp264 canonical kernelがStage C version 3へ更新され、latest outputからStage B version 2の
  `candidate_score_oof.parquet`、`selector_model_manifest.json`、`selector_metrics.json`が外れたこと。
  exp267 v1はmutableなlatest kernel sourceを参照したため、設計時のStage B pathを解決できなかった。
- Kaggle APIのhistorical output `version_number=2`から必要3ファイルだけを取得した。
  scoreは45,407,868 rows / 190 row groups / 411,514,284 bytes。固定SHAはconfigと全て一致した:
  - candidate score: `e51bb674...45a5a`
  - selector model manifest: `12375038...4c9a`
  - selector metrics: `568140aa...c16`
- 3ファイルとprovenanceだけをprivate dataset
  `kentookumura/exp264-stage-b-selector-oof-v2`（dataset id `11230479`）へ固定した。
  dataset file size、private属性、source kernel version 2を確認した。
- configのexp264入力をこのimmutable private datasetへ変更し、mutableなexp264 latest kernel sourceを外した。
  exp263 Stage 0は従来どおりkernel sourceを使う。仮説、18特徴、K=3、fold、seed、guard、
  0 booster契約は変更していない。version 2は同じcanonical exp267 kernelへ再pushする。

## Stage A Kaggle version 2 package監査

- strict experiment validation、Ruff F821、targeted tests 4件はPASS。承認後の契約に合わせ、
  contract testは`run_approved=true`かつStage A 0 booster、conditional Stage B / inference disabledを確認する。
- canonical id/titleはversion 1と同じ。metadataはprivate CPU、GPU/TPU/internet off、
  competition source 1、exp263 kernel source 1、immutable exp264 Stage B v2 dataset source 1、
  `run_on_push=true`。
- bootstrap manifest / ZIP / loose filesは14 entriesで一致し、canonical 17 cellsとpackageの
  bootstrap後17 cellsも一致した。
- version 2 push前SHA:
  - config: `50fd9c25fc9f776d5d7aab2c2a28e63d1293d8be504123a79f20f6e0e99d2970`
  - candidate contract: `ca0350c943c33028349f9dbce92b218a1c03ff1082178971eb4806e10592b49e`
  - settings: `04d7e95ce0d35823f9e5bf56bccf58afaa96fb5a4d69028676e07c1b372ba22d`
  - `src/well_segment_candidate_divergence.py`: `48663f439e80b5f2ab6a4f4e67d8398c3762dac6edcf1e21c60460dcf532e27d`
  - canonical notebook: `94623e4d862af4161f00d3510de3d0045aa245b6dc9dfadc7bc6fc9f86e5d65f`
  - bootstrap付きpackage notebook: `0ee3b2e3c71bff40f803a17354719f2b36e0ac2786ef5c5a81447a917fbc5549`
- 同じcanonical kernelへversion 2をpushした。Kaggle id_noは`127573486`。
- push後pullでprivate CPU、GPU/TPU/internet off、competition source 1、exp263 kernel source 1、
  immutable exp264 dataset source 1を再確認した。

## Stage A Kaggle version 2結果

- status: `COMPLETE`。notebook最終出力時刻は約162秒、Stage Aは契約どおり0 variant / 0 config /
  0 trained fold / 0 booster、親/control再学習0。
- technical: 3,783,989 rows / 773 wells / 5 folds / 18 features / forbidden hit 0でPASS。
- coverage: 773/773 wellsが3区間を持ち、fallback well / segmentは0。
- stability: centroid-matched agreementはfold 0--4で
  `1.000000 / 0.993548 / 1.000000 / 0.954839 / 0.993506`、各fold基準0.95をPASS。
- occupancy: pooled low/middle/highは`538 / 41 / 194` wells。middleはpooled基準75をFAILし、
  fold 1/2/3も`7 / 3 / 9` wellsでouter-valid基準10をFAIL。
- divergence profile: bank range low<middle<highがearly/middle/lateすべて成立したのはfold 4だけ。
  fold passは`false / false / false / false / true`で1/5、要求5/5をFAIL。
- semantic label consistencyはPASSしたが、18 scaled centroid平均のsemantic順序は区間別bank rangeの
  単調順序を保証せず、middleはfoldごとにlow寄りまたはhigh外れ値へ変化した。
- candidate winnerはfold patternが全て異なり、modal fold count 1で要求4をFAIL。
- calibration方向はhigh-minus-low biasが5/5 foldsで負となりPASS。
- pooled cluster actual MAEはlow `5.889654`、middle `6.021557`、high `12.737138` ft。
  最大誤差well `1b1eba53`を除いてもhighがworstでleave-one-well-out guardをPASS。
- score separability guardはPASSしたがstructure guardがFAILし、総合`stage_a_guard_pass=false`。
  conditional Stage B 10 CPU boostersは未実行のまま不採用とする。

## 生成物確認

- Kaggle出力14件を`kaggle/output/train_v2/artifacts/`へ選択取得した。合計約488 KB。
- signatureは773 rows × 21 columns（key 3 + feature 18）、assignmentは773 rows × 12 columns、
  coverageは2,319 rows。manifest記載の10主要artifact byte SHAと取得ファイルが全て一致した。
- logical SHA:
  - signature: `bb193a6a540675a469c52c2cb8abb572a2e5827a2bec0ee6262b866ca593decf`
  - assignment: `325d1f8adb26e7806ca8743b9792f46bdc51c3dd72e07664072c57599aa962b9`
  - preprocessor: `a45d662e88ad22e0d8be86c270630c5025246811320c6e487a5f1e41f8432e39`
- local `execution.run_approved=false`へ戻し、同じpackageの誤再pushを防止した。
- 最終検証はrepository全87 tests、strict experiment validation、config/metrics/summary契約、
  artifact SHA/shape契約を全てPASSした。
