# exp275 XGBoost final regressor swap on exp238 セッションノート

## 目的

`exp238`の最終TVT回帰LightGBMだけを、ユーザー指定の公開 notebook `cdeotte/xgb-starter-cv-15` version 3のXGBoost設定へ差し替える。特徴、target、outer fold、sample weightなし、nested selector rank-slot変換は固定する。

## 現在の状態

- Route: `ml_model`
- 状態: reference inference / scoring完了、raw guard FAIL・train-side不採用・anchor不更新
- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- CV: XGBoost 8.302528478、parent 7.936690031、delta +0.365838447
- Public LB: 7.760（正規ref `54798185`、duplicate ref `54798337`も7.760）
- push承認: 2026-07-17T22:55:37+09:00にユーザーから取得。承認範囲は1 variant / 1 XGBoost config / 5 folds / 5 boosters / control再学習0。

## 実行計画とGPUコストガード

- active variant: 1 (`xgboost_cdeotte_public_v3`)
- XGBoost config: 1
- outer folds: 5
- 合計新規booster: 5
- 親/control LightGBM再学習: 0
- selector再学習: 0
- PF/Beam再生成: 0
- reference inference: 1 run（version 2 COMPLETE）
- submission: 正規1件（ref `54798185`）+ duplicate ref `54798337`

## 公開設定の出典

- Kaggle kernel: `cdeotte/xgb-starter-cv-15`, id_no `117996285`
- Local archive: `docs/notebooks/rogii-wellbore-geology-prediction/vote_top/cdeotte__xgb-starter-cv-15/xgb-starter-cv-15.ipynb`
- SHA256: `348323bd9f449b566301051ca1842692f4ba54bdf05e7cfcc8faa7fc72617f70`
- 選択: version 3 / `FAST_DEBUG=False`の`XGB_PARAMS`
- trees 450、learning rate 0.035、depth 5、early stoppingなし。

## コマンドログ

```bash
make new-steering EXP=exp275_xgboost_final_regressor_swap_on_exp238
make new-exp EXP=exp275_xgboost_final_regressor_swap_on_exp238 SOURCE=templates/experiment
```

`task`は環境に存在しなかったため、リポジトリ指定のMakefileフォールバックを使用した。

### 実行済みの静的確認

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py -o experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_train.ipynb
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_inference.py -o experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_inference.ipynb
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_inference.py
.venv/bin/python -m py_compile experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp275_xgboost_final_regressor_swap_on_exp238/exp275_xgboost_final_regressor_swap_on_exp238_compact_selfcontained_train.py --select F821
.venv/bin/pytest -q tests/test_exp275_xgboost_final_regressor_swap_contract.py
make validate-exp EXP=exp275_xgboost_final_regressor_swap_on_exp238
```

- Jupytext round-trip: train / inferenceともPASS。
- `py_compile`: train / inferenceともPASS。
- Ruff全体 / F821: PASS。
- exp275固有contract tests: 5 passed。
- repository全体: 124 passed。
- experiment validator: `experiment validation passed (strict)`。
- validatorへ未対応の`--strict`引数を渡した試行はCLIエラーになったため、引数なしの正規コマンドで再実行した。科学計算やKaggle runは開始していない。

### Notebook章構成の確認

- 親exp238の通常train script: 217行。
- exp275 compact self-contained train: 1,074行、markdown見出しで8章。
- exp275は目的・設定・入力契約・特徴再構成・nested rank-slot・fold学習・readout・artifact保存をnotebook上で追跡できる。薄い`main()`呼び出しにはしていない。

### Kaggle package確認

```bash
make prepare-kaggle-notebooks EXP=exp275_xgboost_final_regressor_swap_on_exp238 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp275-xgboost-final-regressor-swap-on-exp238-train --title 'exp275 xgboost final regressor swap on exp238 train' --strict"
```

- canonical train notebookをNvidia T4 / internet disabled / `run_on_push=false`でpackage化した。
- bootstrap ZIPは30 entries。公開notebook、exp238 / exp237 / exp218 helper、`config.yaml`の存在を確認した。
- 初回の非実行packageでは、bootstrap内configが`run_approved=false`、5 boosters、親再学習false、公開source SHAとXGBoost parameter完全一致であることを確認した。
- package作成のみで、Kaggle push / GPU trainは行っていない。

### GPU train承認後のみ

```bash
make prepare-kaggle-notebooks EXP=exp275_xgboost_final_regressor_swap_on_exp238 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp275-xgboost-final-regressor-swap-on-exp238-train --title 'exp275 xgboost final regressor swap on exp238 train' --run-on-push --strict"
make push-kaggle-train EXP=exp275_xgboost_final_regressor_swap_on_exp238
```

### GPU train push v1

- 2026-07-17T22:55:37+09:00に、1 variant / 1 config / 5 folds / 5 boosters / control再学習0でユーザー承認を取得した。
- `run_approved=true`へ変更後、固有5 testsとstrict experiment validationを再実行してPASSした。
- `--run-on-push --strict`で再packageし、T4、internet disabled、bootstrap 30 entries、config byte parity、公開source SHA / parameter parity、5 boosters、control / selector再学習falseを確認した。
- 最初のcanonical候補`kentookumura/exp275-xgboost-final-regressor-swap-on-exp238-train`は`SaveKernel` 400となり、学習は開始されなかった。同じIDのpullも403で、既存kernelは確認できなかった。
- id/titleはともに51文字だった。既存sibling exp274の命名規約に合わせ、意味を維持した短縮canonical名`kentookumura/exp275-xgboost-final-regressor-exp238-train` / `exp275 xgboost final regressor exp238 train`へ揃えて再packageする。別expは作らない。
- 短縮canonical packageの再検証はPASSしたが、pushは`Maximum batch GPU session count of 2 reached`で拒否され、exp275は未開始。既存2枠は`exp274-catboost-final-regressor-exp238-train`と`exp264-exp263-confidence-dual-selector-tvt-train`で、Kaggle statusはいずれもRUNNING。既存runは停止せず、空き枠後に同一packageを再pushする。
- 2026-07-18T06:36:49+09:00にユーザーから再実行依頼を受けた。実行範囲は既承認の1 variant / 1 config / 5 folds / 5 boosters / control・selector再学習0から変更しない。
- 2026-07-18の再packageは固有5 tests、strict validation、config byte parity、公開source SHA / parameter parityを再度PASSした。
- `kentookumura/exp275-xgboost-final-regressor-exp238-train`へのpushは2回とも`Notebook not found`で停止し、listにexp275は現れずstatusは404のため学習未開始。13 kernel sourcesは全件pull成功し、成功済みexp274とsource集合・competition source・T4設定が一致することを確認した。
- このslugは前日のGPU上限エラー時にserver側で不完全状態になった可能性がある。実体不在と重複リスクを確認したうえで、同じexp275内の短縮canonical名を`kentookumura/exp275-xgb-final-regressor-exp238-train` / `exp275 xgb final regressor exp238 train`へ変更する。仮説、config、fold、booster数、入力sourceは変更しない。
- `kentookumura/exp275-xgb-final-regressor-exp238-train`へのpushに成功。Kaggle version 1、id_no `127706029`、Nvidia T4、internet off、status RUNNINGをserver metadata / statusで確認した。push時刻は2026-07-18T06:43:04+09:00。
- 実行packageのconfig SHA256は`d933ebdd9e375accaa394f73d74c9788f3404451ff0e6d02fd1d52f64d63d1a6`、notebook SHA256は`29592f42f74bda1b194eab397ec304a471c7cdab01aec168519828f8240c97ca`。実行内容は1 variant / 1 config / 5 folds / 5 boosters / 合計2,250 trees、control・selector再学習0。
- version 1は約21.19秒、notebook In[3]のapproval guardでERROR。`config.yaml`の`model.approval.status=user_approved`に対し、notebookは文字列`approved`を要求していた。公開parameter auditの表示後、データ読込・特徴生成・XGBoost fit前に停止しており、学習boosterは0本。
- 失敗分類はconfig contract mismatch。承認状態を`approved`へ統一し、固有contract testも同じ値へ変更した。科学条件、公開パラメータ、fold、入力、5-booster承認範囲は変更しない。version 2として同じkernel IDへ再package / pushする。
- 修正後は固有5 tests、strict experiment validation、bootstrap config byte parity、`run_approved=true`、approval値とnotebook guard一致、公開source parameter、5 boosters、control / selector再学習0をPASS。version 2 package config SHAは`c54eb61f1760307ba628bb59e69831b8bdc03554408cdb5fcd971285ef5df61b`、notebook SHAは`501cca22a46e7f475a0142a8b870a2644a6c0fd98021311f5fa217f0e50d43d5`。
- 同じkernel IDへversion 2を2026-07-18T06:49:34.993+09:00にpushした。30秒超経過後もKaggle status RUNNINGで、version 1の約21秒approval guard停止点を通過した。完了監視は別途行う。

### GPU train v2完了とoutput監査

- `kentookumura/exp275-xgb-final-regressor-exp238-train` version 2、id_no `127706029`は`COMPLETE`。Kaggle T4、internet disabled、elapsed 2,984.807秒、5/5 boosters、合計2,250 treesを完走した。
- public parameter auditは保存済み`cdeotte/xgb-starter-cv-15` version 3と完全一致。1 variant / 1 config / 5 folds、parent-control / selector再学習0、sample weightなし、early stoppingなしを維持した。
- parent `lgb_mean` RMSE 7.936690031に対し、raw XGBoostは8.302528478、delta `+0.365838447`。fold 0-4のdeltaは`+0.245483 / +0.312755 / +0.085388 / +0.062621 / +1.072875`で、改善foldは0/5。
- 1000+は`+0.400383`、hidden-like spatial / typewell-purgedは`+0.668466 / +0.661976`、worst well `86454a6f`は`+13.880009`。全raw guardがFAILし、`adoption_supported=false`。
- 固定0.25 blendはRMSE 7.990746590、parent比`+0.054056559`。XGBoostとparentの予測相関は0.999995765で、実用的な多様性を支持しない。
- model / OOF / SHAの実ファイル確認が必要だったため、Kaggle outputを`/tmp/exp275-output-v2`へ一時取得した。3,783,989行 / 8列のOOF、5モデル各450 trees、415特徴を確認した。
- summary記載の主要artifact SHAは全件実ファイルと一致。5 model SHAはmanifestと全件一致。OOF decompressed SHA `285614e2c510e3250012b832f8b84d91e6d83a23ac153a7fee4ecdfa31554744`も一致した。大きな成果物はリポジトリへ保存していない。
- summary SHAは`12fbac5b3418d80e2e911623f1b30c92aa75f72dc7d92016d87f23d3e09bb143`、model manifest SHAは`0ecffa597108cfa86471bf7019c92d672a191bef56d1f844e0441018a5690d5a`、feature schema SHAは`85d57f2fce115f54d861c61bf47ba37eba3723d55ab71a4b074161460856805c`。
- prediction時のCPU `DMatrix` / GPU booster device mismatch警告と、特徴列組立時のDataFrame fragmentation警告が出た。いずれも性能警告で、学習完走・finite coverage・artifact監査には影響しない。

### 参考推論・スコアリングoverride

- 2026-07-18にユーザーから「参考のために推論とスコアリングに進む」明示承認を取得した。train-side不採用、raw guard FAIL、anchor不更新は維持する。
- primary output / submit対象は保存済みXGBoost 5本平均のraw `submission.csv` 1件だけ。parent LightGBM 15本平均と固定0.25 blendは比較生成物であり、submitしない。
- current-testではexp238と同じ380 base + fold-matched 35 nested rank-slot = 415列を再生成する。保存済みselector 20本、XGBoost 5本、parent 15本だけをloadし、学習・selector再学習は0。
- train summary / XGBoost manifest / feature schemaの期待SHAをconfigへ固定。全model SHA、feature order、450 trees、sample ID順、finite、fallback 0をfail-closedで検証する。
- exp274 reference inferenceの8章構成をXGBoostへ移植し、exp275 inferenceは917行 / 8章。薄い`main()`呼び出しにはしていない。
- Jupytext round-trip、py_compile、full Ruff、exp275固有5 tests、strict experiment validationをPASS。repository全体は128 tests PASS。
- Kaggle output取得後にsubmit-checkを行い、PASSしたraw XGBoostだけを1件submitする。LBは参考値として記録し、CV guardを事後変更しない。
- canonical inference package `kentookumura/exp275-xgb-final-regressor-exp238-inference`をprivate / T4 / internet off / run-on-pushで作成。kernel source 10件とcompetition sourceを確認した。
- bootstrap ZIPは30 entries。主要7依存sourceを含み、embedded / loose config SHAはともに`f939f812e245fe0b299e7fb7b9f4d76adb7acb5597c1a12b983ebfe2194a5afa`でbyte parity PASS。生成notebook SHAは`db6f7550410a7193ffc56426d64f38ca283a63443f24e016dc5023ff8381015f`。
- package上でもraw XGBoost submitだけ、5 XGBoost / 15 parent / 20 selector load、新規model 0、`.fit()`なしを確認した。
- `kaggle kernels push ... --accelerator NvidiaTeslaT4`でcanonical inference version 1をpush。id_no `127732356`、Kaggle側metadataでprivate / T4 / internet off / 10 kernel sourcesを確認し、statusは`RUNNING`。
- inference version 1は約25.55秒、train summary / manifest / schema SHA検証後に`KeyError: position`でERROR。exp275 schemaは`feature_index,feature,family`、移植元exp274は`position`だった。モデルload・current-test再生成前でprediction model 0本。
- schema sortを`feature_index`へ修正し、0始まり連番を追加guardした。推論条件、保存model、特徴、submission対象は変更せず、固有5 tests / Jupytext / py_compile / Ruff / strict validationを再度PASS。同じkernel IDのversion 2へ再packageする。
- version 2 packageはconfig SHA `aac075e...03e`、notebook SHA `217a89b...0bc`、10 sources、T4、internet off、`.fit()`なしをPASS。2026-07-18T12:32:46+09:00に同じkernel IDへpushした。
- inference version 2は`COMPLETE`。Kaggle T4で415.815秒、14,151行 / 3 wells、380 base + 35 nested rank-slot = 415特徴、fallback 0を確認した。保存済みXGBoost 5本、parent LightGBM 15本、selector 20本をloadし、推論中の学習は0。
- raw XGBoostの予測rangeは`11590.978516 - 12242.695312`、mean `11904.817982`。parentとの差分はRMSE `0.917322348` / mean `-0.106029457` / max abs `2.934570312`。固定0.25 blendとparentとの差分はRMSE `0.229326880`。
- fold別415列matrix SHA、train summary / model manifests / feature schema SHA、current-test selector surface SHAを検証した。inference summary SHAは`7da364539030e039b31313bb5f0c108aa82cc395060dc5a12ca9721b6e39658e`。
- raw / parent / fixed0.25 blend submission SHAはそれぞれ`79452e652e75c3e7f60cb3b77c39dd4f4e175f853f4b1d49accc28b67c70a01c` / `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e` / `ea0e11bc17faf5e3f7c04c023e6aabb20d9b643bd7b5cfadbcc8f3eab73a0c6c`。
- root `submission.csv`だけを一時取得し、14,151行、列`id,tvt`、sample ID順完全一致、重複0、finite 100%、summary記載SHA一致を確認した。公式submit-checkと`make submit-check`はFAIL / WARNともに0。
- raw XGBoostだけをKaggle code submissionとして1件提出した。submission refは`54798185`、kernel version 2、submitted at `2026-07-18T03:43:53Z`。
- 2026-07-18T18:36:40+09:00にKaggle CLIでref `54798185`が`COMPLETE`、Public LB `7.760`、Private LB空欄と確認した。CV 8.302528に対してLBはexp238 hidden-safe 7.775より`-0.015`良い一方、現ML submitted anchor exp274 7.715より`+0.045`、ensemble anchor exp082 7.601より`+0.159`悪い。
- monitorは10分後に追加されたref `54798337`へ追従し、340分で`COMPLETE` / Public LB `7.760`を記録した。API上で両refの同一scoreを確認したが、SHAと提出メッセージを追跡できる正規記録はref `54798185`とする。追加refの同一submission SHAはAPI一覧だけでは検証できないため、重複監視記録としてのみ残す。
- LBはCVの序列と逆転したが、全raw CV guard FAIL、exp274より劣ること、ensemble anchorにも届かないことからtrain rejection、parameter rescueなし、anchor不更新を維持する。

## 変更点

- `exp238`と同じ380 base特徴を親実装から再構成する。
- 保存済みouter別selector scoreから同じ35 rank-slot特徴を生成する。
- 公開 notebookの`XGB_PARAMS`を`.ipynb`からAST抽出し、`config.yaml`と完全一致しない場合は学習前に停止する。
- foldごとに415列float32 matrixを作り、XGBoostを1本だけ学習して直ちに解放する。
- 保存済み`exp238/lgb_mean`とのoverall/fold/bucket/hidden-like/by-well比較、予測相関、固定0.25 blendを保存する。
- fold matrix content SHA、5 model SHA、OOF decompressed SHA、manifest SHAを保存する。
- inference notebookは通常のadoption inferenceを停止したまま、明示承認されたreference-only推論 / scoringだけを許可する。

## 再現性メモ

- seed policy: 公開設定`random_state=42` + 保存済み`exp238` outer role。
- stochastic components: XGBoost GPU histogram training。
- CPU/GPU runtime: Kaggle Nvidia T4、internet disabled。version 1はpreflightで約21.19秒、booster 0本。version 2は2,984.807秒で5 boostersを完走。
- deterministic anchor: false。rerun監査前はbitwise再現を主張しない。
- input / feature schema SHA: 実行時に保存し、output実体との一致を確認。
- feature content SHA: base 380列とfold別415列matrixを保存し、manifestから確認。
- model manifest / model SHA: 5本を保存し、output実体と全件一致。
- prediction SHA: OOF gzipのdecompressed content SHAを保存し、output実体と一致。
- submission SHA: raw `79452e652e75c3e7f60cb3b77c39dd4f4e175f853f4b1d49accc28b67c70a01c`、parent `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e`、fixed blend `ea0e11bc17faf5e3f7c04c023e6aabb20d9b643bd7b5cfadbcc8f3eab73a0c6c`。competition submitはraw XGBoost 1件だけ。

## 次のアクション

1. scoring記録は完了。追加監視は不要。
2. parameter / blend rescueやanchor更新は行わず、train rejectionを維持する。
