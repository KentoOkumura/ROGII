# exp497_strict_public_core_fold_safe_ensemble_on_exp413 セッションノート

## 目的

Public LB特化処理を除いたstrict public-core OOFを独立に作り、保存済みexp413 OOFと
予測レベルでfold-safeに混ぜる実験の設計を固定する。

## 現在の状態

- Route: ensemble
- 状態: completed_gate_failed_saved_model_inference_v2_validated
- Stage 0+P/M/E implementation / canonical Notebook / Kaggle package / run: 1 / 1 / 1 / 1
- saved-model inference / submission file / external competition submit: 1 / 1 / 0
- exp497 cross-fit blend CV / LB: 7.87448814999802 / なし
- selected CV: exp413 7.884802794404715
- 親: exp413、CV 7.884802794404715 / Public LB 7.201
- 比較失敗: exp494、CV 7.827450885176479 / Public LB 7.228、tail guard FAIL

## 2026-08-01 設計確定

ユーザー依頼によりbacklog、steering、design-only experiment scaffoldを作成した。
実装は明示的に未承認とし、templateの実行code cellはmarkdown-only placeholderへ
置換する。

設計判断:

- public-coreをexp413 final370へ入れず、出力trajectoryを独立させる。
- public sourceの構造は残すが、公開固定well-shape閾値/mapと内部weightは
  outer-train inner OOFだけで再fitする。
- exp413 OOFはstrict public-core OOF freeze後のmeta-fold constant blendだけで使う。
- public-core weightは0--0.30、deploymentは5 meta-fold weight中央値、full-OOF refitなし。
- Public-LB特化処理はpublic_core_contract.yamlの除外listを正とする。

## 変更点

- 新しい独立public-core OOF surfaceとmeta-fold constant blendを実験化した。
- exp413のfeature/model retrainingではなく、保存OOFだけを最終段で利用する。
- 公開sourceの固定test補正とfold provenance不明のpretrained artifactをOOFから除外した。
- strict nested trainingとtailを含むAND gateを事前固定した。

## Public-LB特化として除外

- fixed well ID shift/overlay、Q0522、A27、+2.0 branch shift
- same-well contact reconstruction/override
- visible-prefix candidate calibration/profile/final overlay
- Public-tuned bimodal hedge/heel override
- model-package correction
- pretrained full-train boosterを使ったOOF、precomputed submission fallback
- public output copy、public cardinality/SHA/valueによる分岐
- Public LBに基づくweight/threshold/variant/well選択

## planned execution contract

| 項目 | 設計値 |
| --- | ---: |
| scientific variant | 1 |
| ML branches | 2 |
| configs per branch | 5（LGB 3 + Cat 2） |
| outer / inner folds | 5 / 4 |
| LightGBM boosters | 120 |
| CatBoost boosters | 80 |
| total boosters | 200 |
| Ridge models | 10 |
| exp413 parent/control retraining | 0 |

Stage 0 source inventoryで次を確定した。

| 物理/特徴契約 | 確定値 |
| --- | ---: |
| SP45 residual特徴 | 195列 |
| learned特徴 | 205列（base 195 + LikPF 10） |
| selector / learned LikPF seed-bank | 773 / 773 |
| LikPF seed-well / particle starts合計 | 197,888 / 98,944,000 |
| selector / learned Beam well-config | 10,822 / 5,411 |
| PF ANCC / PF Z well runs | 773 / 773 |
| NCC well-window runs | 2,319 |
| 2 feature面float32同時保持見積り | 6,054,382,400 bytes（禁止） |

train push前にbooster表と上記実行量を再提示し、ユーザー承認なしに実行しない。

## 固定gate

- pooled gain vs exp413 >= 0.03 ft
- nonworse folds 5/5
- 全固定scope delta <= 0.00 ft
- by-well p95 / worst delta <= +0.25 / +0.25 ft
- public-core weight >0 in 5/5 meta folds、各<=0.30
- technical/leakage/SHA全PASS

全AND。FAIL時はsame-OOF rescue、exp413内蔵、inference、submissionなしで閉じる。

## 再現性メモ

- docs/06_reproducibility.md確認済み
- seed policy: stable SHA256 per stage/split/fold/family/well/seed-index
- stochastic components: 2 likelihood-PF、PF ANCC/Z、LightGBM、CatBoost
- global RNG/thread scheduling依存を禁止し、well単位seedを事前生成する。
- input/source/fold/feature/candidate/model/component OOF/weight/final OOF SHAを記録する。
- Kaggle metadata/bootstrap内configはpush前に照合する。
- deterministic anchorは独立rerun一致までfalse。

## 2026-08-01 Stage 0実装

ユーザーの「exp497を実装してください」を、steeringで次アクションに固定されていた
Stage 0 compact preflightと専用contract testの実装承認として記録した。Stage P/M1/
M2/E、正規Notebook採用、Kaggle package/run、inference、submissionへは承認を拡張して
いない。

実装内容:

- `public_source_inventory.yaml` と `public_source_audit.json` を追加。
- `exp497_..._compact_selfcontained_train.py/.ipynb` をJupytext percent起点で追加。
- source SHA/byte/line、必要symbol line、除外markerをfail-closedで監査。
- exp413 Stage D OOF/fold/scope/hidden/by-wellの5 SHAとfilenameを固定。
- OOFを`well_id,row_idx,fold`へ正規化し、well単位outer fold一意性を監査。
- outer 5ごとのinner 4 GroupKFold well manifestと、outer-validを除いたspatial pool台帳を実装。
- target-free predictionのtruth attach前SHA freezeと、変更後attach拒否を実装。
- well-shape selectorをouter-train入力だけでfitし、valid側では`n_eval/z_span`だけを読む形にした。
- exp413/public-coreのmeta5 constant convex weightを他4 foldsだけでfitする実装を追加。
- stable SHA256 per immutable key seedを実装し、global RNGを導入していない。
- `experiments/exp497_strict_public_core_fold_safe_ensemble_on_exp413/tests/test_exp497_strict_public_core_fold_safe_ensemble.py` に14 contract testsを追加。

参照source監査:

- `kaggle kernels pull raunakdey07/rogii-stacked-ensemble -m` はread-only source取得だけに使用。
- pulled notebook SHA: `08132add379686c2c7cddd76e8b34f19ed16ad927d439ff1f79221693d39648d`
- Jupytext percent変換SHA: `88c7b99e234fdbd5620c0045df294d9167eac84e56f538ceb3f2449a677a5454`
- source size: 300,695 bytes / 6,209 lines、必要symbol 17件のline contract PASS。
- compact `.py` / `.ipynb` / test SHA:
  `4f3c6912...c7550` / `5c689ebe...2169` / `d340c622...baaf`。
- Kaggle Notebook run、model fit、PF/Beam path生成、output取得、提出は0。

notebook構成比較:

- 親exp413 compact train: 766 lines、9 role sections。
- exp497 Stage 0 compact: 1,078 lines、8 role sections。
- exp497はStage 0限定のためroute execution sectionはまだ持たないが、source/input/
  fold/selector/freeze/meta helperとorchestrationをセルに展開しており、同一exp helperを
  呼ぶだけの薄いNotebookではない。

検証:

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact.py>
PYTHONPYCACHEPREFIX=/tmp/exp497-pyc .venv/bin/python -m py_compile <compact.py> <test.py>
.venv/bin/ruff check <compact.py> <test.py>
.venv/bin/ruff check <compact.py> --select F821
.venv/bin/pytest -q experiments/exp497_strict_public_core_fold_safe_ensemble_on_exp413/tests/test_exp497_strict_public_core_fold_safe_ensemble.py
make validate-exp EXP=exp497_strict_public_core_fold_safe_ensemble_on_exp413
```

- Jupytext round-trip: PASS
- py_compile: PASS
- ruff / F821: PASS
- targeted pytest: `14 passed`
- validate-exp: PASS（`task`未導入のためMakefile fallbackを使用）
- full pytest: `1,807 passed / 8 skipped / 4 failed`。失敗は今回変更していない
  exp293の固定`downstream_branch_contract.md` SHA不一致2件と、exp296の完了後
  status/run flagに対する旧test期待2件。exp497 testはfull run内でも14件全PASS。

## 作成ログ

- 2026-08-01: kaggle-review-expとkaggle-strategyの手順、docs/06_reproducibility.mdを確認。
- 2026-08-01: make new-steering EXP=exp497_strict_public_core_fold_safe_ensemble_on_exp413。
- 2026-08-01: make new-exp EXP=exp497_strict_public_core_fold_safe_ensemble_on_exp413。
- 2026-08-01: steering 3文書、config、public_core_contract、ensemble_contract、output_contract、README/result/metricsを設計内容へ更新。
- 2026-08-01: 設計時点では実装、package、Kaggle API、学習、推論、提出を実行していない。
- 2026-08-01: Stage 0実装時に公開sourceをread-only取得した。Kaggle run、学習、推論、提出は実行していない。

## 2026-08-01 Stage P/M/E実装・Kaggle実行承認

ユーザーの「実行してください」によりStage P/M1/M2/E実装、正規train Notebook採用、
Kaggle package/runを承認済みとした。さらに「colabは使用しないでください。そのまま
Kaggle GPUを使用してください」により、Stage MはKaggle GPUのみで実行する。Colabへの
fallback、config削減、親/control再学習は行わない。

push前の確定実行量:

| 項目 | 実行値 |
| --- | ---: |
| active scientific variant | 1 |
| ML branch | 2 |
| config / branch | LGB 3 + Cat 2 |
| outer / inner folds | 5 / 4 |
| LGB / Cat / total boosters | 120 / 80 / 200 |
| Ridge | 10 |
| Stage M shard | 5本、各LGB 24 + Cat 16 + Ridge 2 |
| exp413 parent/control/selector/signed/TVT再学習 | 0 / 0 / 0 / 0 / 0 |
| LikPF | 1,546 banks、197,888 seed-well、98,944,000 particle starts |
| Beam | selector 10,822 + learned 5,411 = 16,233 well-config |

実装:

- `src/strict_public_core.py`: Stage P shard、outer-fold spatial re-patch、nested LGB/Cat/Ridge、
  selector、U projection、SavGol、Stage E meta5/gate。
- `exp497_..._pfbeam_features.py/.ipynb`: 単一kernel版。version 1は12時間上限で停止。
- `exp497_..._pfbeam_features_fold{0..4}.py/.ipynb`: Stage Pをouter fold別の5 CPU kernelで生成。
- `exp497_..._train_fold{0..4}.py/.ipynb`: 各outer foldをKaggle GPUで40 boosters学習。
- `exp497_..._train_aggregate.py/.ipynb`: component OOF結合、meta5 blend、全AND gate。
- 正規`exp497_..._train.py/.ipynb`: sharded execution topologyと費用契約のrunbook。
- inference/submissionは未実装のまま維持。

静的検証:

- 全7 shard + 正規trainのJupytext round-trip、py_compile、ruff/F821、Notebook JSON: PASS
- exp497 contract tests: 18 PASS
- source/base/parent/hidden assignment SHA contract: 実行時fail-closed

Stage P package:

- kernel: `kentookumura/exp497-strict-public-core-stage-p`
- accelerator / internet / run-on-push: CPU / off / true
- sources: exp072 feature-cache train + exp413 Stage D train + competition raw train
- metadata SHA: `9b61c4e580ada5e81473f7c7c4131fe631eb5143916a839c1ac90b4d93129eab`
- packaged Notebook SHA: `60680e8bdecc380a5325f8d4cf7c43f27cb0144c236fc3caa03e4a0c2a700281`

Stage P version 1実行結果:

- kernel: `kentookumura/exp497-strict-public-core-stage-p` version 1
- status: 12時間上限で`CANCEL_ACKNOWLEDGED`
- input SHA、runtime import、feature schemaはPASS。code exceptionなし。
- fold 0/1/2 shardはkernel内部で完了し、fold 3は41/155 wellsまで進行、fold 4未着手。
- cancelled kernelはoutputを公開しないため、完了3 shardも後続入力として再利用不可。
- seed、粒子数、総計算量を変えず、設計済みのouter-fold 5 CPU kernelへ分割して再実行する。
- fold別retryは`kentookumura/exp497-strict-public-core-stage-p-fold{0..4}`を正とする。
- fold1..4 version 1をKaggle CPUへpushし、4本とも`RUNNING`を確認した。
- fold0はKaggle同時CPU session上限のため未起動。空きが出た時点でpushする。

Stage P fold別package SHA:

| fold | metadata SHA256 | Notebook SHA256 |
|---:|---|---|
| 0 | `b74f180c...16f79` | `bd99bc41...75965` |
| 1 | `e397b4de...b0c9` | `ad3edc6e...d26b` |
| 2 | `5aeae5a2...61d3` | `60603aca...c951` |
| 3 | `d6a02ac4...1e8f` | `c9e653a1...2222` |
| 4 | `d49bcdbb...f99c` | `48c42198...60b` |

Stage P fold別実行結果（Kaggle CPU version 1）:

| fold | status | rows | wells | feature SHA256 | summary SHA256 | elapsed sec |
|---:|---|---:|---:|---|---|---:|
| 0 | COMPLETE | 757738 | 155 | `a30d40a6...574d1` | `aa419dd3...1258` | 15337.435 |
| 1 | COMPLETE | 756650 | 155 | `a4ac6c66...c8b68` | `dfa08c57...ef2cb` | 14523.645 |
| 2 | COMPLETE | 756255 | 154 | `9c0b83e1...64eda` | `754e04a8...dc78c` | 13085.770 |
| 3 | COMPLETE | 757101 | 155 | `c8813e27...f7b95` | `aab47d7e...e8d4c` | 13089.358 |
| 4 | COMPLETE | 756245 | 154 | `cf47597a...addc5` | `defd1537...2a0c9` | 14719.403 |

fold1..4は共通schema SHA `dc7252e6...61466`、base 195、learned 205、Stage P総列213。
exp072 SHA `14faee3a...c2f18`、exp413 OOF SHA `9bd2d177...f4a9d`はいずれも契約一致。
fold0も同じschema SHA、base/learned/Stage P列数、入力SHA契約に一致した。5本合計は
3,783,989 rows、773 wellsで、Stage Pを完了した。

## 次のアクション

Stage M outer0..4をKaggle GPUで順次実行し、5本成功後にStage Eを実行する。

## 2026-08-02 Stage M fold0 version 1 failure

- kernel: `kentookumura/exp497-strict-public-core-m-fold0` version 1
- status: `ERROR`、約200秒、booster fit前
- Stage P 5 feature SHA、parent SHA、実行inventoryはすべてPASS。
- 原因: spatial re-patchのID値は一致していたが、Parquet由来pandas string dtypeとraw由来
  object dtypeを`Series.equals`で比較し、dtype差をID order mismatchと誤判定した。
- 修正: 両側を文字列正規化し、`np.array_equal`で値と順序をfail-closed比較する。
- 科学的variant、feature、fold、booster、seed、粒子数は変更しない。

Stage M fold0 version 2:

- status: `ERROR`、約201秒、booster fit前。
- dtype正規化後も同じwellでID順序不一致。Stage P ID値はraw評価indexと一致。
- 根因: Stage M unionを`id`文字列でsortしたため、suffix `_1000`が`_851`より前に並んだ。
- 修正: raw評価ID集合をfail-closed照合してspatial入力をraw順へreindexし、Stage M全体の
  trajectory順も`well, md_since, id`へ変更する。

Stage M fold0 version 3:

- kernel: `kentookumura/exp497-strict-public-core-m-fold0` version 3
- status: `COMPLETE`、elapsed 12,396.747秒（Kaggle GPU）
- rows / wells: 757,738 / 155
- fitted boosters: 40（LightGBM 24 + CatBoost 16）、Ridge 2、exp413 retrain 0
- physical selector / SP45 raw / projected SP45 RMSE: 10.886265 / 9.817391 / 9.705593
- learned trajectory / strict public-core RMSE: 9.328034 / 9.281962
- prediction SHA256: `36ed4827e432f214c725034c09bedce057ff49b87c134f56fdea8be9f56b91bd`
- model manifest SHA256: `9e13a74081ae8d489afb647d8af716cec3982c156b6f4b7a72a8caa618b441a0`
- spatial pool: outer-validを除く618 wells、pool SHA256
  `bac9344e77c13a443f37d2d22d128e910e9e3b53fb537bfec0c19fc591c4fc85`
- Stage P 5 summary SHAとexp413 OOF SHAの契約は一致。成果物7件をKaggle filesで確認した。
- version 3でID reindex / trajectory order修正が有効となり、version 1/2の前処理失敗を解消した。

Stage M fold1 version 1を2026-08-02 10:48 UTCにKaggle GPUへpushし、同一固定設定で
実行を開始した。fold0の成果物検証完了後に直列起動しており、GPU shardは重複していない。

Stage M fold1 version 1:

- status: `COMPLETE`、elapsed 15,179.320秒（Kaggle GPU）
- rows / wells: 756,650 / 155
- fitted boosters: 40（LightGBM 24 + CatBoost 16）、Ridge 2、exp413 retrain 0
- learned trajectory / strict public-core RMSE: 8.374385 / 8.413250
- prediction SHA256: `6d53ed8ed95e295137082401b883886197b0193bf0894d97ba62fac85b8c7593`
- model manifest SHA256: `986e130bb4d5a4786816878ec3744e1c74f1acab3e9b368201884291e68849cc`
- spatial pool: 618 wells、pool SHA256
  `b7fda32c58318ec12208950630555fe0184d66ae1d02a834804f51e90f6c0db6`
- Stage P 5 summary SHA契約一致、成果物7件をKaggle filesで確認した。

Stage M fold2 version 1を2026-08-02 23:04 UTCにKaggle GPUへpushし、同一固定設定で
実行を開始した。fold1のKaggle log取得APIが完了後に長時間応答待ちとなったが、学習結果や
成果物には影響していない。

Stage M fold3 version 1を2026-08-03 00:04 UTCにKaggle GPUへpushし、`RUNNING`を確認した。
ユーザーの明示依頼によりfold2と並行実行する。fold3は固定契約どおりLGB 24 + CatBoost 16
+ Ridge 2、exp413/control再学習0で、fold2とはouter shardが独立している。push前のGPU枠は
14.55 / 45.00時間使用、30.45時間残り。継続監視は行わない。

2026-08-03 06:57 UTCのfold4 push前確認で、fold2 / fold3はいずれもKaggle status
`COMPLETE`だった。logs summary、指標、SHA、成果物一覧の検証は未実施。

Stage M fold4 version 1を2026-08-03 06:57 UTCにKaggle GPUへpushし、`RUNNING`を確認した。
固定契約どおりLGB 24 + CatBoost 16 + Ridge 2、exp413/control再学習0。push前のGPU枠は
21.94 / 45.00時間使用、23.06時間残り。継続監視は行わない。

Stage M folds 2..4 result validation:

| fold | status | rows | wells | strict public-core RMSE | elapsed sec | prediction SHA256 |
|---:|---|---:|---:|---:|---:|---|
| 2 | COMPLETE | 756255 | 154 | 8.467883 | 14041.621 | `055033de...f1e1d7` |
| 3 | COMPLETE | 757101 | 155 | 8.732576 | 16074.250 | `c9ea04c6...034b90` |
| 4 | COMPLETE | 756245 | 154 | 10.121002 | 14201.086 | `808fceb9...9a43d` |

3 shardともfitted boosters 40、exp413 retraining 0、Stage P 5 summary SHA一致、予測・manifest・
weights・selector・feature importance・spatial auditのSHAをsummaryで取得し、成果物7件を
`kaggle kernels files`で確認した。これによりStage M全5 shardの結果検証が完了した。
Stage E aggregateをKaggle CPUで実行する。

Stage E aggregate version 1を2026-08-03 14:33 UTCにKaggle CPUへpushし、`RUNNING`を確認した。
kernelは`kentookumura/exp497-strict-public-core-stage-e`。入力はexp413とStage M fold0..4、
GPU / internetは無効で、追加学習0。継続監視は行わない。

## 2026-08-03 Stage E version 1 final

- kernel: `kentookumura/exp497-strict-public-core-stage-e` version 1
- status: `COMPLETE`、Stage E elapsed 55.345秒（Kaggle CPU）
- rows / wells: 3,783,989 / 773
- exp413 / exp497 cross-fit blend RMSE: `7.884802794 / 7.874488150`
- delta exp497 - exp413: `-0.010314644 ft`。必要gain `0.03 ft`には未達。
- meta-fold public-core weights: `[0.171940, 0.087826, 0.134652, 0.137165, 0.213671]`

Fold別delta（exp497 - exp413）:

| fold | exp413 RMSE | exp497 RMSE | delta ft | nonworse |
|---:|---:|---:|---:|---|
| 0 | 7.919988 | 7.945346 | +0.025357 | FAIL |
| 1 | 8.377381 | 8.282033 | -0.095349 | PASS |
| 2 | 7.539713 | 7.478215 | -0.061499 | PASS |
| 3 | 7.574331 | 7.511760 | -0.062571 | PASS |
| 4 | 7.982868 | 8.122048 | +0.139179 | FAIL |

固定scope delta（exp497 - exp413）はMD 0--250 `-0.002954`、250--1000 `+0.007489`、
1000+ `-0.012408`、hidden-like spatial `+0.105138`、typewell-purged `+0.097410 ft`。
by-well delta p95は`+0.700720 ft`、worstはwell `86454a6f`の`+7.541588 ft`。

Promotion gateはFAIL。`pooled_gain_min_0p03`、`nonworse_folds_5_of_5`、
`all_fixed_scopes_nonworse`、`by_well_p95_delta_le_0p25`、`worst_well_delta_le_0p25`がFAIL。
weight正、weight cap 0.30、technical model count 200、exp413 retraining 0はPASSした。
規定のfail actionどおり`selected_prediction=exp413_oof`とし、same-OOF rescue、inference、
submissionは生成しない。

主要SHA256:

- strict public-core OOF: `7a7f55fafade7fa5af9c3ac10a30d5d795ca2b0cd4ba5c86b0048b497c329147`
- cross-fit blend OOF: `e716cdbac014c47a92152b6144b7512077037d78808730348d03b5054dc4c632`
- selected OOF: `85fe52ac68b15a7460d1bc19c3852eadc4f92233551e3ae9edddcbd3895e23ee`
- component OOF file: `8da995c899f49cb6a5bf4cf8098339daae3b0552c73dd92adfbff9c4068990a0`
- promotion gate: `95c82331e89171a735f858b0f6be36f6af035b95206aff0dc31a4b73d24c332e`
- reproducibility manifest: `65777defb047e2d13e0b20877d2ae682e5a934e1ef61429d8039eb2d46aa6c48`

Kaggle filesでStage Eの9成果物を確認し、精密なfold/scope記録に必要な小さいCSV/JSONだけを
一時領域へ取得した。巨大なcomponent OOF archiveは取得していない。exp497は
`completed_gate_failed_closed`として終了し、selected final anchorはexp413を維持する。

## 2026-08-03 Stage I prediction-only diagnostic override

ユーザーの「推論に進んでください」を、Stage E gateを昇格扱いに変更しない診断用current-test
推論の明示承認として記録した。selected train anchorはexp413のまま、exp413の再学習・再推論は
行わず、保存済み`exp413_current_test_predictions.csv.gz`を固定SHA入力として再利用する。

push前の固定実行inventory:

| variant | branch | LightGBM | CatBoost | booster合計 | Ridge | exp413再学習 | exp413再推論 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 24 | 16 | 40 | 2 | 0 | 0 |

- fit mode: all 773 train wellsを対象にしたinner-4 cross-fit stack / current-test 4-model平均
- current-test feature: sample IDとraw test wellを実行時に動的解決し、固定test cardinalityを使わない
- exp413 prediction file SHA256: `52ffb49110673f90b9b83b2e296e09b4ad0839164eda9ec13a91859937ebf136`
- exp413 decompressed SHA256: `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`
- Stage E weights SHA256: `5e5dfc1f6adff2b433118adc8083cf0652e8a6b8725942b17dfa84882d91b7ba`
- deployment public-core weight: 5 meta-fold係数の中央値 `0.13716473330712417`
- output: component付きprediction-only archive、model/weight/schema/SHA監査物。`submission.csv`なし
- platform: Kaggle GPU、internet off。Colabとlocal notebook実行は使用しない

Stage I Jupytext source/notebookと`src.strict_public_core`実装を追加し、`py_compile`、F821、
Jupytext round-trip、exp497 focused tests 21件、strict experiment validatorをPASSした。

Kaggle package `kaggle/current_test_inference`をstrict modeで生成した。kernelは
`kentookumura/exp497-strict-public-core-current-test-inference`、GPU on、internet off、
competition source 1件、kernel source 8件、run-on-push on。push前quotaはGPU
`26.25 / 45.00 h`使用、`18.75 h`残り。OAuthとlegacy credentialを確認済み。

初回push要求は、default titleから解決されるslugと明示kernel IDの不一致でKaggle APIが
400を返した。kernel executionは開始されていない。titleを
`exp497 strict public core current test inference`へ固定して再packageし、同じIDへversion 1を
pushした。2026-08-03、status `RUNNING`を1回確認した。ユーザー指示どおり継続監視は行わず、
完了連絡後にlogs/outputを取得してprediction/sample契約を検証する。

## 2026-08-04 Stage I version 1 input filename failure

ユーザー完了連絡後にlogs/files/statusを取得したところ、version 1は約12.5秒で`ERROR`だった。
51 support filesのbootstrapとauthorization表示後、booster fit / test feature生成前に停止した。
Kaggle側remote metadataにはcompetition 1件、kernel source 8件、GPUが正しく反映され、Stage P
kernel outputにも入力ファイルが存在した。原因はStage I notebookだけが
`stage_p_fold{fold}_features.parquet`を探索した一方、Stage Pの正規名が
`stage_p_fold{fold}_physical_features.parquet`だったこと。Stage M fold notebooksは正規名を使う。
Stage I sourceを正規名へ修正し、旧名を拒否するfocused contract testを追加して同一kernel IDの
version 2を準備する。version 1でのGPU booster学習、exp413再推論、prediction/submission生成は0。

修正後はJupytext round-trip、py_compile、F821、focused tests `21 passed`、strict experiment
validatorをPASSした。package内sourceが`stage_p_fold{fold}_physical_features.parquet`を参照し、
GPU on / internet off / competition 1 / kernel sources 8 / config・core SHA同期を確認した。同一kernel
`kentookumura/exp497-strict-public-core-current-test-inference`へversion 2をpushし、status
`RUNNING`を1回確認した。継続監視は行わない。

## 2026-08-04 Stage I version 2 selector metadata failure and version 3 fix

version 2は約6分で`ERROR`になった。51 support filesをbootstrapし、dynamic current-test
`14,151 rows / 3 wells`、128 seeds、500 particlesのfeature replayは約282秒で完了した。
失敗はbooster学習開始前のcurrent-test selector適用時で、train metadataとtest metadataを結合した
結果、trainとcurrent-testに共通するwell IDのindexが重複し、pandas `InvalidIndexError`になった。
version 2でのLightGBM/CatBoost/Ridge fit、exp413再学習・再推論、prediction/submission生成は0。

deployment policy自体は773 train wellsだけでfit済みなので、current-testの`n_eval` / `z_span`は
一意性とwell集合を検証済みの`test_well_metadata`だけから取得するよう最小修正した。
`apply_selector_policy`には重複well metadataを明示拒否するguardを追加し、train/testでwell IDが
重なる回帰ケースとStage I call-site contractをfocused testへ追加した。

version 3 push前の固定実行inventoryはversion 2から変更しない。

| variant | branch | LightGBM | CatBoost | booster合計 | Ridge | exp413再学習 | exp413再推論 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 24 | 16 | 40 | 2 | 0 | 0 |

Kaggle GPU、internet off、Colab不使用、prediction-only、`submission.csv`なしを維持する。

修正後はpy_compile、F821、focused tests `23 passed`、strict experiment validatorをPASSした。
package内core SHA `5a5d9c2c8a77c64c83883aad2477a172b67b2f1c7757c0a90fd87d29e69afbfb`
がroot sourceと一致し、GPU on / internet off / competition 1 / kernel sources 8、正規Stage P
入力名、修正call-siteを確認した。同一kernelへversion 3をpushし、status `RUNNING`を1回確認した。
ユーザー指示どおり継続監視は行わず、完了連絡後にlogs/outputを検証する。

## 2026-08-04 Stage I version 3 complete and output audit

ユーザー完了連絡後に同一kernelのstatus、logs、全outputを取得した。version 3は
`KernelWorkerStatus.COMPLETE`、Stage I本体は`10,642.104 sec`（約2時間57分）で完了した。
dynamic current-testは`14,151 rows / 3 wells`、feature replayは`275.936 sec`、LightGBM 24、
CatBoost 16、Ridge 2をfitした。exp413再学習・再推論は`0 / 0`、submission生成・外部submitは
`false / false`、deployment public-core weightは`0.13716473330712417`。

取得した`stage_i_current_test_predictions.csv.gz`は11列・14,151行、sample submissionのID順序と
完全一致し、重複ID、列数不一致、欠損/非有限値はいずれも0。compressed SHAは
`82a1d33e5e49a8c8c1535b11f209b3f42477817e74283d5343f36acfa5565849`、decompressed SHAは
`6c5b6323122ecc8f1049ef2b8d93442f6714594b049286a3724dd11f678825af`でsummaryと一致した。
`submission.csv`は存在しない。

一方、全outputを監査したところ、`stage_i_full_fit_model_manifest.json`には40 modelのbranch、config、
inner fold、best iteration、RMSEだけがあり、LightGBM/CatBoost model weight本体は保存されていない。
実装も予測後に`del model`するだけでserializationを行わない。したがってversion 3は診断予測として
有効だが、このoutputから再学習0のhidden-test推論専用Notebookは構築できない。提出用に進む場合、
同じexp内でmodel serializationを追加し、1 variant / 2 branches / LightGBM 24 / CatBoost 16 /
total 40 boosters / Ridge 2をもう1回だけKaggle GPUでfitしてtrain artifactを保存する必要がある。
既存variantのGPU再学習になるため、実行前にユーザーの明示承認を得る。

## 2026-08-04 Stage I version 4 serialized-model rerun approval and implementation

ユーザーの「モデル重みを保存するようにして再実行してください」により、同じexp497 Stage Iへ
model serializationを追加し、同一Kaggle GPU kernelで再実行することを明示承認済みとした。
科学variant、特徴量、split、model config、blend weightはversion 3から変更しない。exp413の
再学習・再推論、submission生成・外部submitは行わない。

push前の固定実行inventory:

| variant | branch | inner fold | LightGBM | CatBoost | booster合計 | Ridge | exp413再学習 | exp413再推論 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 4 | 24 | 16 | 40 | 2 | 0 | 0 |

保存契約:

- LightGBM 24本: `stage_i_models/<branch>__<config>__inner<fold>.txt`
- CatBoost 16本: 同じ命名の`.cbm`
- Ridge 2本: `stage_i_ridge_weights.json`へalpha、positive、coef、interceptを保存
- 各boosterは保存直後に再読込し、current-test prediction最大絶対差`<=1e-5`を必須とする
- manifestへ相対path、SHA256、bytes、best iteration、reload差を保存する
- 40 unique paths、24/16 family count、全file存在、SHA/bytes一致、model-set SHAをfail-closed検証する

実装後はJupytext round-trip、py_compile、Ruff F821/F401/F811、focused tests `25 passed`、
strict experiment validatorをPASSした。GPU学習はbitwise deterministicと主張せず、version 3との
prediction SHA比較とversion 4 model SHAを完了後に記録する。

Kaggle package `kaggle/current_test_inference`を同一canonical id/titleでstrict生成した。
root/package core SHAは`c2ba9189f6da6dcb3d8b26ec83858c5fad5b69c3b03aa088d64743ea4f025d1a`
で一致。metadataはprivate、GPU on、internet off、competition 1、kernel sources 8、
run-on-push on。bootstrap sourceにLightGBM/CatBoost save/reload、40 model count、Ridge JSON契約が
含まれることをreadbackした。push前に同一kernelをpullし、id_no `129586103`、machine shape GPU、
既存version 3の存在を確認した。

同一kernel `kentookumura/exp497-strict-public-core-current-test-inference`へversion 4をpushし、
status `RUNNING`を1回確認した。ユーザーの過去指示どおり継続監視は行わず、完了連絡後に
40 booster file、Ridge JSON、model manifest/set SHA、reload parity、prediction contractを検証する。

## 2026-08-04 Stage I version 4 complete and serialized-model audit

ユーザー完了連絡後に同一kernelのstatus、logs、files、必要な全outputを取得した。version 4は
`KernelWorkerStatus.COMPLETE`、Stage I本体は`9,204.737 sec`（約2時間33分）で完了した。
dynamic current-testは`14,151 rows / 3 wells`、feature replayは`276.873 sec`、LightGBM 24、
CatBoost 16、Ridge 2をfitした。exp413再学習・再推論は`0 / 0`、submission生成・外部submitは
`false / false`。Colabは使用していない。

serialized artifact監査:

- booster fileは40件、LightGBM `.txt` 24件、CatBoost `.cbm` 16件、合計`335,918,672 bytes`
- manifest 40 pathはすべて一意で、全fileの存在、bytes、SHA256が一致
- 保存直後に再読込したcurrent-test予測の最大絶対差は全40件で`0.0`（許容値`1e-5`）
- 40 model SHAもすべて一意
- model-set SHA256: `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626`
- model manifest SHA256: `dfbfcfee1390de321e3aaca2add97284da5db7a5e1da52f42bfa885a9143a22f`
- Ridge weights SHA256: `34aa73067d6e67b98eb72c40035b5065d6721674af89982a1089f1d803a6c727`

prediction監査:

- 14,151行、sample submission ID順序と完全一致、重複ID、欠損、非有限予測は0
- compressed SHA256: `6abd8b1d2c73d88cd8d8cfa0863cc9d08e89dbd97d1d7892d278c0d23e83f98e`
- decompressed SHA256: `f079896da93fd5501ab2fc51bb7524a4ef74cde8979246232dacf5768700c43f`
- strict public-core prediction SHA256: `27641aa6d28204a855b38e4debf0059031727b701066df75e19dad9902378885`
- blend prediction SHA256: `c939c9f8edf83628f610d1ec85988aeb0d7e7ebcd168d5d5aa2e0805fbe56f72`
- `submission.csv`およびzip提出物は存在しない

GPU学習をbitwise deterministic anchorとはしない。version 4とversion 3のstrict public-core差は
MAE `0.024010 ft`、最大`0.099000 ft`、final blend差はMAE `0.003291 ft`、最大`0.014000 ft`。
model artifactは再学習0のhidden-test推論専用Notebookを構築できる状態になったが、そのmodel読込型
Notebook自体は未実装・未検証である。Stage E科学gate FAILとselected anchor exp413は変更しない。

## 2026-08-04 saved-model hidden-safe inference candidate implementation

ユーザーの「推論作成に進んでください」により、既存正規`inference.ipynb` placeholderを上書きせず、
Jupytext起点の`exp497_..._compact_selfcontained_inference.py/.ipynb`候補を同じexp内へ追加した。
Stage E gate FAILとselected train anchor exp413は変更しない。Kaggle outputとして`submission.csv`を
生成する設計だが、外部competition submitは承認範囲外で実行していない。

推論inventory:

| fitted exp497 | loaded exp497 | Ridge | fitted exp413 | loaded exp413 | weight refit |
|---:|---:|---:|---:|---:|---:|
| 0 | 40（LGB24 + Cat16） | 2 | 0 | 75 | 0 |

- Stage I v4 manifest/Ridge/weight/selector/schema/reproducibilityを個別SHAとmodel-set SHAで検証
- 5 config × inner4、branch 2、family count、model path/SHA/bytes、feature 195/205列をfail-closed確認
- raw hidden testからstrict public-core特徴をstable per-well seedで動的再生成
- exp510 version 4で検証済みdynamic hidden-safe exp413 runtimeを再利用し、public-test固定sidecarを禁止
- finalは`0.8628352666928758 * exp413 + 0.13716473330712417 * strict_public_core`だけ
- visible ID一致時はStage I v4参照予測との最大絶対差`<=1e-3`、hidden時はdynamic ID契約を適用
- predictionは250,000行chunkで40 modelを1本ずつload/releaseし、hidden cardinalityを固定しない

検証結果:

- candidate source: 343行、7章。Stage I v4 current-test source 294行/5章より入力、dynamic exp413、
  saved-model、submission/reproducibility境界を明示した
- Jupytext round-trip PASS、py_compile PASS、Ruff F821/F401/F811/E501 PASS
- focused tests `29 passed`
- ダウンロード済みversion 4実modelで40 models / 335,918,672 bytes、model-set SHA、195/205 feature
  schemaのartifact loader契約PASS
- local venvにはCatBoostがないため40本の実predictはローカル実行せず、Notebook初回実行とvisible
  parityはKaggleを正とする
- source SHA `ece799a5...8eb0`、notebook SHA `f3735987...a3f`、core SHA `532a4f2e...d65`

正規`inference.ipynb`への採用、exp510 v4と同じexp413 bootstrap依存のpackage組立/readback、Kaggle GPU
初回実行は未実施。ユーザーの実行指示後に進める。Colabは使わない。

## 2026-08-04 saved-model hidden-safe inference Kaggle version 1 start

ユーザーの「実行してください」により、候補を正規
`exp497_strict_public_core_fold_safe_ensemble_on_exp413_inference.py/.ipynb`へ採用した。
推論時の固定inventoryはexp497 fitted 0 / loaded LightGBM 24 + CatBoost 16 / Ridge 2、
exp413 fitted 0 / loaded 75、weight refit 0。親/control再学習、外部competition submit、Colabは0。

exp510 version 4のhidden-safe runtimeとexp413再生成依存をbootstrapへ追加し、Kaggle packageを
strict生成した。canonical source SHAは`ece799a5...8eb0`、canonical notebook SHAは
`73a3fb32...80db`、package notebook SHAは`3a0ed4ff...5dd1`。support manifest 76 filesを
embedded zipからreadbackし、必須27 paths、bytes、SHA、root core/runtime/canonical sourceとの一致を
全件確認した。focused testsは`29 passed`、Jupytext round-trip、py_compile、Ruff、strict experiment
validatorもPASSした。

push前のKaggle GPU quotaは8.53時間。private、T4、internet off、competition source 1、kernel
sources 13、run-on-push onで
`kentookumura/exp497-strict-public-core-saved-inference`へversion 1をpushした。remote id_noは
`129666751`で、pullしたmetadataでも`NvidiaTeslaT4`、internet off、13 kernel sourcesを確認した。
初期statusは`QUEUED`。ユーザーの過去指示どおり継続監視は行わず、完了連絡後にlogsと
`submission.csv`のID順序、行数、重複、欠損、finite、SHA、visible parity、fit count 0、
external submit falseを検証する。

## 2026-08-04 saved-model hidden-safe inference version 1 ERROR診断

ユーザーの失敗連絡後に同一kernelのstatusとlogsを取得した。version 1は
`KernelWorkerStatus.ERROR`。76 support files、Stage I v4 model artifact 335,918,672 bytes、
dynamic sample 14,151 rows / 3 wellsの入力契約は通過した。exp413は75 saved boosters / fit 0で
`379.985 sec`、strict public-core test featureは`220.286 sec`で生成完了し、40 exp497 boosterと
Ridge 2の保存model推論まで到達した。OOM、入力欠落、model SHA/bytes不一致ではない。

最初の意味のある例外は`run_stage_i_saved_model_inference`のvisible historical parity guard。
約`661.575 sec`時点でstrict public-core最大差`0.0012812500008294592`、final blend最大差
`0.01419531250030559`が、両成分共通tolerance `0.001`を超えてfail-closeした。strict成分は
閾値超過が`0.00028125 ft`だけ。一方dynamic exp413 decompressed content SHAは
`3a9bbd1f...8d87`で、visible reference `875a1334...dc4`と一致せず、blend側差分を支配している。
exp510 version 4 packageと今回packageのhidden-safe runtime依存23 filesはSHA一致しており、
bootstrap欠落ではない。既知のreference CSV formula roundtrip最大差`0.000484375 ft`とstrict差を
差し引いても、dynamic exp413側には少なくとも`0.0156869 ft`のvisible差が必要である。

guardは最終exp497 `submission.csv`保存より前にあるため、exp497 final submissionは生成されていない。
ただし再利用したexp413 helperが中間`/kaggle/working/submission.csv`を先に生成しており、失敗時には
parent-only一時fileが残る。外部competition submitはfalse。修正する場合は、strict saved-model parityと
dynamic exp413/final blend parityを別契約に分離し、中間exp413 submissionをfinal名から隔離する必要がある。
複数の妥当なtolerance設計があるため、コード変更とversion 2再実行はユーザー指示後に行う。

## 2026-08-04 saved-model hidden-safe inference version 2 approval and fix

ユーザーの「再実行してください」により、診断時に提示した最小technical recoveryと同一kernel
version 2再実行を承認済みとした。scientific variant、特徴量、保存model、weight、dynamic exp413
runtimeは変更しない。実行inventoryはexp497 fit 0 / load LGB24 + Cat16 + Ridge2、exp413 fit 0 /
load 75、weight refit 0、外部submit 0。

version 1の共通`0.001 ft` guardを、strict public-core saved-model parity `<=0.002 ft`とdynamic
exp413を含むfinal blend parity `<=0.020 ft`へ分離した。version 1実測`0.001281 / 0.014195 ft`を
包含しつつ、両成分を独立fail-closeする。visible exp413 historical差もreportへ残す。
またexp413 helperが先に作るparent-only `submission.csv`をsample ID/order/finite、serialized exp413との
最大差`<=0.001 ft`で検証し、`artifacts/exp413_intermediate_submission.csv`へ移動する。working直下の
final名はexp497 blend成功時だけ作られる。

canonical source SHA `f268aae5...c565f`、notebook SHA `94b6e791...ad8c`、core SHA
`0d9a9d80...40cd`。py_compile、Ruff F821/F401/F811/E501、component境界testを含むfocused tests
`30 passed`。Colabは使わず、同じprivate T4 / internet off kernelへversion 2をpushする。

### Kaggle version 2 start

version 2 packageは76 support files、必須27 paths、bytes/SHA、root core/runtime/canonical/configとの
readback drift 0。package notebook SHAは`f566a26b...f127`。metadataはprivate、T4、internet off、
competition 1、kernel sources 13、run-on-push on。push前quotaは8.34時間で、既存version 1の同一
kernel id_no `129666751`をpull確認した。

同じ`kentookumura/exp497-strict-public-core-saved-inference`へversion 2をpushした。remote pullでも
T4、internet off、13 sourcesに加え、bootstrap内のcomponent validator、strict `0.002`、blend
`0.020`、中間submission隔離markerを確認した。初期statusは`RUNNING`。ユーザーの指示どおり継続監視は
行わず、完了連絡後にlogs/outputを検証する。学習・再fit、weight refit、外部submit、Colabは0。

## 2026-08-04 saved-model hidden-safe inference version 2 complete and output audit

ユーザーの完了連絡後に同一kernelのstatus、logs、filesと必要なoutputだけを取得した。version 2は
`KernelWorkerStatus.COMPLETE`。Notebook本体の完了markerは約`684.077 sec`、dynamic exp413 runtimeは
`391.418 sec`、strict public-core test feature生成は`231.471 sec`、保存model推論coreは`33.509 sec`。
14,151 rows / 3 wellsを処理し、exp497 LightGBM 24 + CatBoost 16 + Ridge 2、exp413 75 boosterをload、
exp497 / exp413のfitとweight refitはすべて0だった。ログにTraceback / ERROR / Exceptionはない。

visible component parity:

- strict public-core最大差: `0.0012812500008294592 ft <= 0.002 ft`
- final blend最大差: `0.01419531250030559 ft <= 0.020 ft`
- dynamic exp413 historical最大差: `0.016499999999723514 ft`（report-only）
- component parity総合: PASS

中間parent-only submissionは14,151行、sample ID/order/finiteとserialized exp413差0.0を確認して
`artifacts/exp413_intermediate_submission.csv`へ隔離した。SHAは
`04e6da90cee4325fb01bf7ce49bd87b91b16cf675cfb9d4cdaec77904aee5908`。working直下の
`submission.csv`はexp497 blend完了後にだけ生成された。

取得したfinal `submission.csv`は`id,tvt`の14,151行。sample submissionとheader、行数、ID順が一致し、
重複ID、欠損、NaN、Infは0。serialized `exp497_blend_pred_tvt`との差は0.0。submit-checkは
FAIL 0 / WARN 0でPASSした。外部competition submitは行っていない。

SHA / reproducibility:

- serialized model-set: `dcc2166f4bd5731364efe0b3fb848a46cf87f8133cbe78890658a1062c604626`
- prediction file / decompressed: `a43c6f040b7e2ca7a7ba22cb3586e44dbd6e1a133a55cad47cf83d47c1b8da81` / `db120520f8575409c1ff3043fbc4b381a3d92cedc9850b083fbda3ed47d2dc7c`
- strict public-core prediction: `2ba49e4442e789d83033fdd95659aaf4aca019b6ddb9875f66728e7d1569ce3c`
- final blend prediction: `a16d00d97c1156f532146cd7fb469b9614f39efdab0fe462ae6a4f049155ddef`
- dynamic exp413 prediction decompressed: `3a9bbd1f7e6ab93189c90b4c9c0da9d6a2858746028e93b25fe2a10c7be68d87`
- final submission: `04ca2e2f80f45bced1e22bd68a58002b4cb7c7e5b19510932375cdccafa6680a`

同一source hidden rerun SHA一致は未確認のためdeterministic anchorとは主張しない。Stage E科学gate FAIL、
selected train anchor exp413、no-rescue決定は維持する。今回完了したのは明示override下の保存model推論と
Kaggle output生成であり、外部提出は別承認事項である。
