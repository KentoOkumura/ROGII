# exp370_triggered_reset_rejuvenation_pf セッションノート

## 目的

既存のchange-point atlas particle rejuvenation backlogを、採番した条件付きPF設計として確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_failed_close_without_rejuvenation_pf`
- 優先度: branch閉鎖
- CV / LB: なし
- compact self-contained Stage 0 train / fail-closed inference: 実装済み。
- private Kaggle CPU Stage 0 version 2: `COMPLETE`。
- 正規inference Notebook: placeholderのまま。
- Stage 1: scientific gate FAILにより不適格。
- inference / submission: 無効、未実行。

## コマンドログ

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp370_triggered_reset_rejuvenation_pf
make new-exp EXP=exp370_triggered_reset_rejuvenation_pf
```

### 2026-07-25 Stage 0実装

- ユーザーの`exp370を実装してください`を、設計済みStage 0だけの実装承認として記録した。
- Jupytext percent形式のcompact self-contained train候補とfail-closed inference候補を
  追加した。既存の正規`*_train.ipynb` / `*_inference.ipynb` placeholderは上書きしていない。
- 実行対象契約は500 particles × 1 seed × 773 wells =
  `773 diagnostic PF seed-well runs`。scientific variant 0、reporting folds 5、
  full parent PF control replay 0、LightGBM config 0、trained fold 0、booster 0。
- Stage 0 diagnostic PFはexp072と同じmomentum / process noise / Gaussian GR likelihood /
  systematic resampling順を使い、likelihood update後・resampling前のESS/Nを保存する。
- RNGは`SHA256(experiment|fold|well|family|seed_index)`からwell-localに生成する。
- GR changeはraw隣接差をvisible prefix median / 1.4826 MADでrobust z化し、
  prefix q99.5以上 AND ESS/N `<=0.20`をtrigger候補、refractory 512行を採用した。
- atlasは各foldのouter-train wellsだけから構築する。256行patchを32点へ圧縮、
  source stride 32、2 ft TVT bin、well/bin最大6、bin当たり2 source wells以上を固定した。
  trigger中心の前後128行queryとprototypeのZNCCを計算し、ZNCC降順・TVT昇順で
  10 ft未満の重複候補を除いてtop3を選ぶ。
- 保存済みexp072 cache
  `0503de0512302b06309d26e09fc06ba5095db0ef4d610b1508afe8c8d07ca536`
  をread-only baseとし、`last_known_tvt + likpf_mean_d`をbase TVTへ戻す。
  bad-eventはsaved likPFのtrigger起点128行RMSE `>=10 ft`、coverage比較もsaved likPF。
  Stage 0の1-seed PF predictionはESS診断専用でbase controlを置換しない。
- fold assignment、atlas prototype/manifest、trigger ledger、proposal ledger、score、
  saved control manifestをtarget truth前にgzip decompressed SHAでfreeze / readbackする。
  target well TVTとhidden-like roleはその後だけlate joinする。
- hidden-likeはspatial / typewell-purgedの両面でAUC-circular差とcoverage gainが
  ともに正方向の場合だけPASSとした。
- `execution.run_stage_0=false`、Kaggle package / push / run、Stage 1、inference、
  submissionはすべて無効のまま。

### 2026-07-25 静的検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py \
  tests/test_exp370_triggered_reset_rejuvenation_pf.py
.venv/bin/ruff check \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py \
  tests/test_exp370_triggered_reset_rejuvenation_pf.py
.venv/bin/pytest -q tests/test_exp370_triggered_reset_rejuvenation_pf.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp370_triggered_reset_rejuvenation_pf/exp370_triggered_reset_rejuvenation_pf_compact_selfcontained_inference.py
make validate-exp EXP=exp370_triggered_reset_rejuvenation_pf
make validate-template
make test
```

- `py_compile`: PASS
- `ruff`: PASS
- 専用test: `12 passed`
- Jupytext conversion / `--test`: train / inferenceともPASS
- `validate-exp` strict: PASS
- `validate-template`: PASS
- repository full test: exp370を含む`1098 passed / 7 skipped`。
  既存状態とtest期待値がずれているexp296の2件だけFAILした。後続のsaved-base
  integration test追加後も、exp370専用12件は全PASS。
  exp296のfailは`completed_train_side_guard_failed_closed`と
  `execution.run_variant=false`に対してtestが旧approved-run状態を期待しているもので、
  exp370実装による失敗ではない。
- train sourceは2,390行 / 12章、inference sourceは140行 / 3章。
- 親exp072にはcompact self-contained train sourceがないため直接比較は不可。
  最も近いPF Stage 0のexp367（1,630行 / 8章）と比較し、exp370ではさらに
  fold assignment、saved control、diagnostic PF、atlas、trigger/proposal freeze、
  late truth、metrics/gate、orchestrationをNotebook上へ展開した。
- 同一exp helper import、notebook上の`__file__`参照は0件。

### 2026-07-25 Stage 0実行承認

- ユーザーの`実行してください`を、compact train候補の正規Notebook採用、
  private Kaggle CPU package / push / runの承認として記録した。
- 実行対象はdiagnostic PF replay 1、500 particles、1 seed、773 wells =
  773 seed-well runs、5 reporting folds。
- scientific variant 0、full parent PF control replay 0、LightGBM config 0、
  trained fold 0、booster 0をpush前に再確認する。
- Stage 1の98,944 seed-well runs、inference、submissionは対象外。
- Kaggle logsからgate、fold metric、artifact SHAを回収できるよう、Stage 0終端に
  `EXP370_STAGE0_SUMMARY_BEGIN/END`で囲んだ機械可読summaryを追加した。

### 2026-07-25 Stage 0 push前検証

- compact trainをJupytextで再生成し、27 cells / 13 code cellsの正規
  `exp370_triggered_reset_rejuvenation_pf_train.ipynb`へ採用した。
- compact / canonical Notebook SHA256:
  `c4e0fa631d52b4291a10c3c0e8b530db8ab976a585a85e806bccbdf1dbee573f`。
- 専用test `12 passed`、ruff、py_compile、Jupytext `--test`、strict
  `validate-exp`を再度通過した。
- Kaggle packageはprivate、CPU、internet off、run-on-push、kernel source 2件。
  packaged Notebook SHA256:
  `c49c0175a46d1ee3f471c1b8fc159ddf12ee7cd5b1c170520fc2f473cba9bb7f`。
- bootstrap zip SHA256:
  `ac2f32a80132528a891f07c2a5b7aa3f43640e2b315fab76fb6ea1f61ca26b97`。
  24 support filesのsize / SHA manifestを検証し、embedded / loose config byte一致
  (`12481af06055891b35967ff15d879a1e26bbe489848ee676f3d2f88916f713e0`)を確認した。
- packaged Notebookのbootstrap後27 cellsはcanonical bodyと完全一致した。
- remoteには同slugの既存kernelがないことを`kaggle kernels list -m -s exp370`
  で確認した。push開始予定時刻は`2026-07-25 14:57:08 UTC`。

### 2026-07-25 Stage 0 Kaggle実行

- private kernel
  `kentookumura/exp370-triggered-reset-rejuvenation-pf-train` version 1をpushした。
- URL: https://www.kaggle.com/code/kentookumura/exp370-triggered-reset-rejuvenation-pf-train
- push結果: `Kernel version 1 successfully pushed.`
- version 1は科学計算前のinput identity guardで
  `Raw train and saved exp072 likPF well identity mismatch`によりfail-closedした。
  diagnostic PF seed-well runsは0。
- 保存済みcacheの`well`列をローカル2.1GB artifactからchunk scanし、
  3,783,989 rows / 773 wells、raw trainとの集合差0を確認した。
- 原因はKaggleのcompetition inputが
  `/kaggle/input/competitions/<slug>/train`にmountされる場合をresolverが直接扱わず、
  fallbackの辞書順で`test` directoryを選び得たこと。
- resolverへcompetition slug、`competitions/<slug>/train`、paired 773-well guard、
  fallbackの`train`限定を追加する。科学契約は変更しない。
- resolver回帰testを追加し、専用test `13 passed`、ruff、py_compile、
  Jupytext `--test`、strict `validate-exp`を通過した。
- version 2 canonical Notebook SHA256:
  `d545ea7cd22621a9a959f6e8486e68a21bcecacc37aa919a24f15143889f063e`。
- version 2 packaged Notebook SHA256:
  `e60c9e9ca83cee7164701f0972b99448d387f45edfc582d74a0d3ef272f49cfd`。
- version 2 bootstrap zip / config SHA256:
  `6b5c0e4e792f93f0ed4a322f15668833a81740e91e23edbbf4da5f279a8e82b4` /
  `c4c2c749c55fdd9939c3fd607e0bf9cf515d124dc0fd5db5dd86803bee49ce8d`。
- version 2を同じprivate CPU kernelへpushし、
  `Kernel version 2 successfully pushed.`を確認した。

### 2026-07-25 Stage 0 version 2完了

- Kaggle status: `COMPLETE`、kernel id_no `128591535`。
- remote 28-cell sourceはlocal packageと完全一致。source SHA256:
  `a43a0a4c1157f4f11fef8ff83b1148dc9d7822ee80e8da40b1afdb7c9998ba82`。
- runtime 671.342秒、3,783,989 rows、773 wells、773 diagnostic seed-well runs。
- technical gate: PASS。target truth / hidden role before freeze 0 / 0、
  donor-fold leakage 0、Stage 1 / inference 0、LightGBM / booster 0。
- accepted triggersは13 / 3,685,818 eligible rows、率`3.5270325e-6`。
- overall trigger bad-event AUC `0.4999984852`、circular差`-3.761e-12`。
- atlas top3 within10 coverage `0.076923`、saved likPF coverage `0.846154`、
  gain `-0.769231`。mean best-atlas absolute error 263.112 ftに対し、
  saved baseは5.726 ft。
- 5 foldsのaccepted triggerは`2 / 3 / 2 / 1 / 5`、passing folds `0/5`。
- hidden-like spatial / typewell-purgedはいずれもevent 1、coverage 0、
  saved base比`-1.0`で正方向gateをFAIL。
- decision:
  `stage0_failed_close_without_rejuvenation_pf`。Stage 1 eligible=false。
- Kaggle output archive全体はdownloadせず、logsの機械可読summaryと
  `kaggle kernels files`によるartifact存在確認を根拠に記録した。
- scientific contract / summary / gate report SHA:
  `4546b84c6ca6c3fa71fee3378d46b38101ece3bac1da94f817ea87712abcf875` /
  `3ce97d6b11d8bec962b67df600b0e67f17f8cb82d2a465b48ee343517d652075` /
  `941564dfb9b596111982a47f7b33d1d1c9308c93a3995ae2c7bd61b02fe00821`。
- trigger / proposal / atlas prototype decompressed SHA:
  `7abd280aca86ae4727a893bcf848d7e236afaaf1ca8234ea09930cb06227a666` /
  `58511f5376fe893912b3bf5f70b6a1fcfc977cff0b2e9205e4681687a3216a8f` /
  `ad2635c9f74412fe4b82b20bad88c81a53d2ec4b293796775cc4bdfdd2c1633d`。
- 結果記録後、loose configとlocal Kaggle packageをfail-closedへ戻した。
  metadata `run_on_push=false`、embedded Stage 0 / Stage 1 / inference /
  submission flagsは全false。
- fail-closed package / bootstrap zip / config SHA:
  `57cbdba82e380d33d1ae7aa179731151bab69a2e2748325ea70b54777d4b923d` /
  `9569811b93a53f0a44c72a2e478ae69e1c1f0502ab77cb74009f2ad517284674` /
  `d0356d3ec4c7afd33652b2b6d58e1c760095014f61a7494673ea6a5dc6b816a8`。
- 最終状態で専用test `13 passed`、ruff、py_compile、Jupytext `--test`、
  strict `validate-exp`、`validate-template`をPASSした。
- repository full testは`1101 passed / 7 skipped / 3 failed`。failureは既存の
  exp296 status / run flag期待2件とexp396 Stage B transient flag期待1件で、
  exp370専用13件は全PASS。exp370変更由来の新規failureはない。

## 変更点

- trigger、atlas query、topK、10%注入、17/17/16配分、jitter、source ageを固定した。
- Stage 0は1 seed diagnostic PF、773 seed-well runs、scientific variant 0。
- Stage 1は1 variant、500 particles、128 seeds、773 wells、98,944 seed-well runs。
- LightGBM / booster / full parent PF replayは0。

## 再現性メモ

- seed: `SHA256(experiment|fold|well|family|seed_index)`。
- stochastic: PF propagation/resamplingとatlas proposal jitter。
- outer-fold atlasを別生成し、test atlasは全train wellsから別生成する。
- global RNGは禁止。atlas manifest、trigger/proposal ledger、predictionはcontent SHAを記録する。
- gzipはdecompressed SHA。Stage 0 ledger / atlas / summary SHAを記録済み。
  inference prediction / submission SHAは未生成。

## 次のアクション

1. Stage 1、inference、submissionへ進まず、exp370 branchを閉じる。
2. 同じq99.5 / ESS 0.20 / atlas top3を緩和・再実行しない。
3. reset系PFを将来再検討する場合は、0-PF / 0-atlasの独立readoutで
   GR-changeとESSのjoint support非退化性だけを先に監査する。
