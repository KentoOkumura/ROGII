# exp334_equal_well_loss_weighting_on_exp287 セッションノート

## 目的

exp287の421特徴・fold・3 LightGBM configを維持し、outer-trainで各wellの総学習重みを均等にする1点だけを変更して、global gainを保ちながらwell-level tail guardを回復できるか検証する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train完了、固定promotion guard不通過、非昇格でclose
- CV / LB: `8.09349752413077` / 未実行
- 実装 / preflight / train / inference / submission: 完了 / 完了 / 15 boosters完了 / 0 / 0
- 親control再学習: 0

## 2026-07-21 設計確定

### 根拠

- exp287 OOF RMSE: `8.136708220359452`
- exp264比: `-0.3241030172530248 ft`、5/5 folds改善
- Public LB: `7.530`
- exp264比worst-well: `+8.228409822385604 ft`
- `+1/+3/+5 ft`悪化well数: `135→140 / 39→40 / 14→19`

### 単一変更

outer foldごとに、outer-train総行数 `N`、well数 `W`、well別行数 `n_w` から、各行の重みを `N / (W * n_w)` とする。各wellの総重みは `N/W`、全体平均は1になる。

valid Datasetにはweightを付けず、early stopping、fold/pooled OOF、scope、by-well metricsは非加重とする。target、prediction、error、outer-valid rowsはweight作成に使わない。

### 固定するもの

- exp287のclean 273 + nested compact 74 + fold-safe formation 74 = 421特徴
- outer 5 folds、group `well`、score rows `TVT_input_isna`
- residual target、selector、formation生成、3 LightGBM configs `[0,1,2]`
- seed 42、T4、internet off、GPU DP/threads 8再現性設定
- 保存済みexp287/exp264 OOF control。control boosterは再学習しない。

### 実行量

- 1 active variant × 3 LightGBM configs × 5 folds = 15 GPU boosters
- control再学習 = 0 boosters
- 2026-07-21 19:42:57 JST、ユーザーの「実行してください」により、compact train候補の正規notebook採用、0-booster Kaggle preflight、その成功後の1回の15 GPU booster trainを承認済み。
- inference、submission、control再学習、追加rerunは未承認のまま維持する。

### Promotion gate

- pooled OOFがexp287比 `+0.02 ft`以内
- 4/5 folds以上でexp287以下
- 全scopeがexp287比 `+0.02 ft`以内
- by-well delta p95がexp287以下
- exp264比worst-well deltaが `+0.25 ft`以内
- exp264比 `+1/+3/+5 ft`悪化well数が `135/39/14`以下

すべてAND条件とし、不通過時のguard緩和や同一実験内weight gridは禁止する。

## コマンドログ

設計作業で実行したもの:

```bash
make new-steering EXP=exp334_equal_well_loss_weighting_on_exp287
make new-exp EXP=exp334_equal_well_loss_weighting_on_exp287
```

この節は設計時点の履歴であり、その時点ではKaggle関連コマンド、学習、推論、提出を実行していない。

## 2026-07-21 実装

### 承認範囲

- ユーザー依頼「exp334を実装してください」を、compact self-contained train候補、fail-closed inference候補、専用test、config/記録更新の実装承認として扱った。
- Kaggle package/push、preflight実行、15-booster train、inference、submissionは承認範囲外のまま維持した。
- `execution.stage=preflight_only`、`kaggle_push_approved=false`、`run_train=false`をfail-closedにした。

### 実装内容

- 正規notebook scaffoldは上書きせず、Jupytext起点の次の候補を追加した。
  - `exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_train.py` / `.ipynb`
  - `exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_inference.py` / `.ipynb`
- 親exp287のtrain outputからOOF、model manifest、metrics、fold metrics、by-well metrics、formation fold manifest、raw schema auditと10個のformation fold cacheをSHA固定する。
- formation 74列は再生成せず保存済みfold-role cacheを再利用する。clean 273列とnested compact 74列はexp287と同じStage C/source/configから再構成し、保存model manifestの421列順/SHAと照合する。
- 全5 foldsのouter-train weightをbooster fit前に計算し、finite、正値、平均1、well別総重み`N/W`、同一well同一weight、row identityを検証する。fold/well summaryとlogical SHAを保存する。
- train時にweightを再計算しpreflight SHAと一致させ、LightGBMへ`sample_weight=train_weights`だけを追加する。`eval_set`にはweightを渡さず、early stoppingと全OOF評価を非加重RMSEのままにした。
- promotion gateはexp287比pooled/fold/scope/by-well p95、exp264比worst、clean control比悪化well数`135/39/14`をAND判定する。
- inference候補はtrain guard PASS、15 model/421 feature SHA固定、別承認まで明示的に停止し、sample submissionをコピーしない。

### Notebook比較

- 親exp287にはcompact self-contained trainがないため、正規Jupytext trainを比較元にした。
- 親train: 362行、7章。exp334 compact train: 1,422行、10章。
- exp334では親surface再構成に加え、保存cache/SHA preflight、5-fold weight contract、weighted fit、exp287/exp264二重control gate、再現性manifestをNotebookセル上で追える。

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py>
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp334_equal_well_loss_weighting_on_exp287/tests/test_exp334_equal_well_loss_weighting_on_exp287.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp334_equal_well_loss_weighting_on_exp287/tests/test_exp334_equal_well_loss_weighting_on_exp287.py
make validate-exp EXP=exp334_equal_well_loss_weighting_on_exp287
```

- Jupytext round-trip、py_compile、ruff: PASS。
- 専用pytest: `10 passed`。
- repository全pytest: `502 passed, 2 skipped, 2 failed`。2件は未変更のexp296で、実行完了後の`completed_train_side_guard_failed_closed` / `run_variant=false`と旧testの`kaggle_cpu_*` / 実行承認期待が不一致な既存状態。exp334専用testとは独立のためexp296は変更していない。
- strict experiment validation、template validation、bootstrap dependency 19件の存在確認: PASS。
- `__file__`参照: 0。正規notebook上書き: 0。Kaggle package/push/run: 0。

## 再現性メモ

- seed policy: exp287から固定global seed 42を継承
- stochastic components: 将来承認された場合のLightGBM GPU学習のみ
- CPU/GPU runtime: Nvidia Tesla T4、internet off、`gpu_use_dp=true`、threads 8を予定
- deterministic anchor: 現時点ではfalse。GPU rerun parityを主張しない。
- exp287 OOF SHA: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 metrics SHA: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- exp287 fold metrics SHA: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- exp287 by-well SHA: `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- exp264 corrected OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- exp334 OOF SHA: `7c0bab3e24d72116bf955220b7b53c66b29afed7a0c8a3f093cb97d63d033afa`
- exp334 model manifest SHA: `8d2212b64bef1147967f68255a469965c0d60dd502726973746efaebeb816174`
- exp334 reproducibility manifest SHA: `782d10868ac10bb54a38442bfc561be7d95e19a59d2beb8db6fd8adac7b0aacd`
- submission SHA: 本実験の現承認範囲外

## 2026-07-21 Kaggle実行承認

- 承認時刻: 2026-07-21 19:42:57 JST
- 承認範囲: compact self-contained train候補を正規train notebookへ採用し、Kaggle T4 / internet offで0-booster preflightを1回実行する。preflightがPASSした場合だけ、同じ正規kernelへtrain stageを再packageし、1 variant × 3 LightGBM configs × 5 folds = 15 GPU boostersを1回実行する。
- control再学習: 0 boosters。保存済みexp287/exp264成果物だけを比較controlに使う。
- 承認範囲外: inference、submission、control再学習、guard緩和、weight/config/fold/feature変更、train rerun。
- Kaggle kernel id: `kentookumura/exp334-equal-well-loss-weighting-on-exp287-train`
- canonical slugは48文字でKaggle制約内のため、意味のある実験名を短縮せず使用する。
- preflight stageでは`run_train=false`を維持し、`boosters_trained=0`を確認する。PASS後だけ`execution.stage=equal_well_weight_train`と`run_train=true`へ切り替える。
- compact train sourceから正規train notebookをJupytext変換して採用した。正規notebook SHA256: `4b5c1a48503422742b9c49cdd315dc85244b51fb5aebc087224f0c77afdc90a5`、22 cells（code 10 / markdown 12）、保存output 0。
- 採用後のJupytext round-trip、py_compile、ruff、専用pytest `10 passed`、strict experiment validation: PASS。
- preflight package: `kentookumura/exp334-equal-well-loss-weighting-on-exp287-train`、title `exp334 equal well loss weighting on exp287 train`、private、run-on-push、T4、internet off、5 kernel sources。
- preflight packaged notebook SHA256: `bb40092da12331b2260d41ce1ce35e152696d71557dfc6f413193d127c39d1f4`。
- 埋め込みsupport ZIPは36 files / 268,606 bytesをmanifest SHAで全件照合した。埋め込みconfigは`stage=preflight_only`、`kaggle_push_approved=true`、`run_train=false`、`run_inference=false`、`submit_to_kaggle=false`、planned 15 / control retraining 0を確認した。
- Kaggle preflight version 1をpush。kernel id_no `128110184`、URL `https://www.kaggle.com/code/kentookumura/exp334-equal-well-loss-weighting-on-exp287-train`、初回確認は`RUNNING`。
- push後に同じkernelをmetadata付きでpullし、private / T4 / internet off / competition source / 5 kernel sourcesを再確認した。Kaggle正規化後notebookは23 cells / 保存output 0で、埋め込み36-file manifest、`stage=preflight_only`、`run_train=false`を再照合した。
- 2026-07-21、複数回の状態確認でversion 1は`RUNNING`、実行中logsは空。ユーザー指示「監視は止めていいです。完了したら連絡します。」に従い監視を停止した。Kaggle run自体は停止していない。
- ユーザーから完了連絡を受けるまで、結果確認、`equal_well_weight_train`へのstage切替、15-booster version 2 pushは行わない。
- 2026-07-21、ユーザーから完了連絡を受けてversion 1を確認。Kaggle statusは`COMPLETE`、logsのpreflight PASS messageは`647.994780625 sec`。
- outputの`preflight_manifest.json`を取得して実ファイル照合した。status=`preflight_passed_zero_boosters`、3,783,989 rows / 773 wells / 421 features、5 weight contracts、boosters 0、prediction/submissionなし。
- preflight manifest SHA256: `132169c0315febbd8fda69e164bc40e3c55eeb0cf12dd80524b495912c65bacb`。
- weight summary SHA256: `2826a8408093d18c911030395a0ce5a76564824b9bc6ef0d157ed88ac41c2ca5`。weight by-well SHA256: `2cdc3c29605c58e9458e8e86bccd7777d4a354c4ebcc951fc7b683824d836555`。両方ともmanifestと実ファイルが一致。
- 全foldでmean row weightは浮動小数誤差内で1、maximum total-weight absolute errorは`9.094947017729282e-13`、within-well weight rangeは0、target/error inputなし、validation weightなし。
- 保存済みexp287/exp264 control SHA、10 formation partitions、Stage C、421-feature schema SHA `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`をPASSした。
- preflight PASSを受け、承認済みの`equal_well_weight_train`へ切り替えた。実行量は1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0のまま。inference/submissionはfalse。
- train version 2 packageを同じcanonical id/titleで生成した。packaged notebook SHA256: `362c0202393ed0a532b75be5526653b311456642d187c0fe442345b0c3414668`。
- push前に埋め込みsupport ZIP 36 files / 268,706 bytesをmanifest SHAで全件照合した。埋め込みconfigは`stage=equal_well_weight_train`、preflight v1/id_no/manifest SHA、`run_train=true`、planned 15、control retraining 0、inference/submission falseを確認した。
- metadataはprivate、run-on-push、T4、internet off、competition source、5 kernel sources。push時も`--accelerator NvidiaTeslaT4`を明示する。
- train version 2を同じcanonical kernelへ`--accelerator NvidiaTeslaT4`付きでpush。kernel version 2、id_no `128110184`、初回statusは`RUNNING`。
- push後pullのmetadataはprivate / T4 / internet off / 5 kernel sourcesを維持。Kaggle正規化後notebookの埋め込み36-file manifestを再照合し、`stage=equal_well_weight_train`、`run_train=true`、planned 15、control retraining 0、inference/submission falseを再確認した。
- 2026-07-21 20:41:27 JST、前回のユーザー希望に従い継続監視を停止した。Kaggle train自体は停止せず、ユーザーの完了連絡待ちとする。

## 2026-07-22 Kaggle train完了・成果物監査

- ユーザーの完了連絡後、同一kernel version 2（id_no `128110184`）が`COMPLETE`であることを確認した。fatal error、Traceback、OOMはなく、15/15 boostersを完了した。
- 学習runtimeは`21882.805369142 sec`（約6時間4分43秒）。3,783,989 OOF rows、773 wells、421 features、15 modelsを生成した。control再学習は0。
- OOF RMSEは`8.09349752413077`、exp287 `8.136708220359452`比`-0.04321069622868201 ft`。fold deltaは`[-0.073653463, -0.080749855, -0.001351477, -0.014739468, -0.044164103]`で5/5 folds非悪化。
- scope deltaはnear `+0.003786852`、mid `+0.011048519`、1000+ `-0.050434507`、hidden-like spatial `-0.072137859`、typewell-purged `-0.068352712`で全件PASS。
- tailはby-well p95 delta `+0.429584617`でFAIL、exp264比worst-well `+7.156485377`でFAIL。`+1/+3/+5 ft`悪化well数は`133/40/19`で、許容`135/39/14`に対してPASS/FAIL/FAIL。
- fixed AND gateは、pooled/fold/scopeがPASS、p95/worst/countがFAIL。Kaggle metrics statusは`train_complete_guard_failed`。exp334を非昇格として閉じた。
- 非model成果物11件とmodel 15件をmanifest SHAと照合し、全件一致した。OOFのrow/fold、pooled/fold RMSE、by-well件数も実ファイルから再計算した。
- OOF SHA256: `7c0bab3e24d72116bf955220b7b53c66b29afed7a0c8a3f093cb97d63d033afa`。
- model manifest SHA256: `8d2212b64bef1147967f68255a469965c0d60dd502726973746efaebeb816174`。
- reproducibility manifest SHA256: `782d10868ac10bb54a38442bfc561be7d95e19a59d2beb8db6fd8adac7b0aacd`。
- feature schema SHA256: `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`。
- logs SHA256: `68dc71b80709c352e24db78b881a1810311825cb31087c68cc76aed65ae3e15b`。
- GPU bitwise rerun parityは主張せず、`deterministic_anchor=false`を維持する。
- inference、submission、追加train、weight grid、guard緩和は実行していない。

## 次のアクション

1. exp334は追加実行せずcloseを維持する。
2. 既存の0-booster `exp287_fold_safe_formation_tail_attribution_readout` は、exp334のtail不十分時だけ再開する条件を満たした。着手する場合は別途ユーザー確認を得る。
3. exp334のinference/submissionは行わない。
