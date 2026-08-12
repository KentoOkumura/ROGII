# exp502_exp501_fixed13_selector_replacement_on_exp413 セッションノート

## 目的

exp413 Stage DをTVT controlとし、既存nested selector compact74をexp501 fixed13
selector compact77へ差し替えたときのdownstream TVT OOF価値を測る。add-onlyにはしない。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train version 1完了、primary gate FAILで終端閉鎖
- CV: `7.882143903310376`（saved exp413比`+0.002658891 ft`改善）
- LB: 未実行
- canonical train notebook: compact self-contained候補を採用
- canonical inference notebook: template placeholder（未承認）

## 2026-08-02 Kaggle train実行承認

- ユーザーの「実行してください」を、直前に提示した正規train Notebook採用、Kaggle
  package作成、frozen trainの実行承認として記録した。
- 実行対象は1 treatment × 3 LightGBM configs × outer 5 folds = 15 GPU boosters。
- exp413 control、exp501 selector、exp413 signed selectorの再学習はすべて0。
- HMM / PF / Beam再実行は0。inference実装/runとsubmissionは未承認のまま。
- Kaggle T4、internet disabled、canonical kernelを使用する。

## 2026-08-02 push前検証

- Jupytext sourceから正規train notebookを生成した。inference notebookは変更していない。
- Jupytext round-trip、`py_compile`、F821、8契約test、strict experiment validation、
  template validationはすべてPASS。
- credential checkerはKaggle CLI用OAuthとlegacy credentialを確認した。API tokenは未設定だが、
  CLI実行に必要な認証経路は存在する。
- strict packageをcanonical id/title、`run_on_push=true`で生成した。
- metadataはprivate、T4 (`NvidiaTeslaT4`)、GPU true、internet false、competition source 1、
  dataset source 1、kernel source 7を確認した。
- bootstrapは46 files。正のconfigとbyte一致し、13 dependencyもsourceとbyte一致した。
- bootstrap内configでtrain approval/run flag、T4/GPU設定を確認した。
- bootstrap検証用one-linerの初回はf-string quotingのSyntaxErrorとなった。ファイルやpackageは
  変更せず、式を変数へ分離して再実行しPASSした。

### 初回push 400とcanonical slug短縮

- 初回planned slug
  `exp502-exp501-fixed13-selector-replacement-on-exp413-train`は58文字で、idとtitle由来slugは
  一致していたが、Kaggle `SaveKernel 400 Bad Request`で実行開始前に拒否された。
- 直後の同slug `kaggle kernels pull -m`は403で、Kaggle側に利用可能なNotebookは
  作成されていないことを確認した。
- repo内で反復確認済みの50文字slug上限パターンと一致するため、同じexp502のまま
  `exp502-exp501-fixed13-replace-exp413-train`（42文字）/
  `exp502 exp501 fixed13 replace exp413 train`へcanonical id/titleを同時に短縮する。
- 科学契約、入力、feature面、T4設定、実行量15 boosterは変更しない。拒否された長slugへは
  再pushしない。

### canonical version 1 push

- `2026-08-02 08:30:13 UTC`に
  `kentookumura/exp502-exp501-fixed13-replace-exp413-train`へpush成功。
- Kaggle kernel version: 1、id_no: `129459588`。
- URL: https://www.kaggle.com/code/kentookumura/exp502-exp501-fixed13-replace-exp413-train
- push時package config SHA256:
  `25d05d62b8656e983e45e9f439b8cc41fb42035d8d0d8705a983fe25a1767f58`。
- push直後の`kaggle kernels pull -m`でid/title、private、GPU true、internet false、
  `machine_shape=NvidiaTeslaT4`、dataset source 1、kernel source 7を確認した。
- 同じversion 1を監視し、logs空やstatus 500だけを根拠に再pushしない。

### canonical version 1完了・固定gate FAIL

- `2026-08-02 14:16:36 UTC`にstatus `COMPLETE`を確認した。notebook metrics出力は
  `20702.430524 sec`、最大log timestampは`20715.350925 sec`（約5時間45分）。
- frozen実行量どおりreplacement 1 variant × configs 3 × outer folds 5 = 15 / 15
  T4 modelsを学習した。saved exp413 control、exp501 selector、exp413 signed selectorの
  再学習は0、HMM / PF / Beam再実行は0。
- exp502 RMSEは`7.882143903310376`、saved exp413 `7.884802794404715`から
  `0.002658891094339033 ft`改善したが、必要な`0.03 ft`を未達した。
- fold delta（exp502 - exp413）は
  `[-0.056809309, -0.062355636, -0.259448397, +0.116026853, +0.234685837] ft`。
  nonworseは固定下限どおり3 / 5だが、fold 3 / 4で悪化した。
- scope deltaはMD 0--250 `-0.033026299`、250--1000 `-0.046726384`、1000+
  `+0.001321181`、hidden-like spatial `+0.139586563`、typewell-purged
  `+0.140943998 ft`。最大scope悪化は上限`+0.02 ft`を超えた。
- technical checksは全PASS。pooled gainとmaximum scopeの2 gateがFAILしたため、
  decisionは`FAIL_CLOSE_EXP501_FIXED13_SELECTOR_REPLACEMENT_ON_EXP413`。
- report-only tailはp95 `+1.293097772 ft`、worst well `a8ed028a +8.159899027 ft`、
  `+1 / +3 / +5 ft`悪化well数`60 / 14 / 8`。
- final373 schema / feature manifest / model manifest / OOF SHAは
  `5599190a...a865` / `024754ed...9457` / `ae221d2f...575` /
  `97230e2e...e99`。metrics / input contract SHAは`7ee56bff...07a` /
  `c59b862e...dfc`。
- 15 model SHAはすべて一意で3×5 gridを網羅。best iteration min / median / maxは
  `522 / 1950 / 9832`。10 train/valid final373 matrix content SHAもすべて一意。
- full output archiveは取得せず、logsとmetrics/fold/scope/hidden/by-well/model/feature/
  reproducibility等の小さい監査ファイルだけを`kaggle/output/train_v1_audit/`へ取得した。
  取得ファイルSHAはKaggle metrics内SHAと一致した。
- manifest監査one-linerの初回はfold surface数を5と誤認してassertした。実体は各foldの
  train/validで10件であり、role×fold gridを明示照合する形へ直してPASSした。生成物は変更していない。
- fixed gate直後で停止した。same-OOF subset / blend / weight / threshold / gate救済、
  inference、submissionは行わない。
- 完了結果を反映し、local train run flagと`run_on_push`をfalseへ戻したsealed packageを
  再生成した。sealed config / notebook / metadata SHAは
  `2377570d...fd4c` / `d7ff5c25...a94b` / `cd5e8957...fbe7`。bootstrap 46 files、
  completed FAIL status、train/inference/submission flag false、T4、internet disabledを確認した。

## 2026-08-02 train-side実装

- ユーザーの「exp502を実装してください」をtrain-side実装承認として記録した。
- `exp502_exp501_fixed13_selector_replacement_on_exp413_compact_selfcontained_train.py`
  と同名候補notebookを追加した。この実装時点ではcanonical train / inference notebookを
  上書きせず、後続の実行承認後にtrainだけを採用した。
- 親exp413 compact self-contained trainの9章 / 766行に対し、exp502候補も9章 / 1,447行。
  setup、入力SHA、保存compact検証、replacement-only final373 assembly、15-booster学習、
  fold/scope/hidden-like/by-well、feature importance、model/OOF/reproducibility保存を展開した。
- exp501 Stage C、removed exp413 Stage C、retained exp413 Stage S、saved exp413 Stage Dを
  marker SHA付きで別root解決する。exp413/exp501 fold manifestのbyte SHA一致を必須にした。
- old74/new77は同名slotを多く共有するため、old block 0 / new block 1は列名差集合ではなく
  block provenanceで検証する。final列名自体は373 unique、順序はclean273/new77/signed23。
- 各foldでexp501 compactとexp413 signedの全`KEY_COLUMNS`、role、model count、id重複、
  base join、well、last-known anchor、train/valid overlap、全row coverage、saved exp413 foldを照合する。
- final373 train/valid matrix content SHA、clean273 content/schema、15 model SHA、OOF SHAを
  Kaggle実行時に保存する。GPU bitwise一致は主張しない。
- `experiments/exp502_exp501_fixed13_selector_replacement_on_exp413/tests/test_exp502_exp501_fixed13_selector_replacement_on_exp413.py`に8契約testを追加した。
  replacement provenance、final373値配置、fold key/anchor、authorization、親SHA、tail
  report-only、canonical train採用状態を検証する。

### 検証

- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: PASS
- `.venv/bin/python -m py_compile ...`: PASS
- `.venv/bin/ruff check ... --select F821`: PASS
- `.venv/bin/pytest -q tests/test_exp502_...py`: `8 passed`
- exp502 / exp501 / exp413関連test同時実行: PASS
- `task validate-exp ...`: `task` commandが環境にないため未実行
- `make validate-exp EXP=exp502_exp501_fixed13_selector_replacement_on_exp413`:
  strict validation PASS
- `make update-summary`: `experiment_summary.md`の自動管理表を更新
- `__file__`残存: 0

## 2026-08-02 設計確定

- ユーザー指示によりbacklog、steering、実験scaffoldを作成した。
- 置換面は`exp413 nested74 -> exp501 compact77`に固定した。
- 保持面は`exp413 clean273 + signed23`、finalは373列。
- exp413とexp501のfold manifest SHAは
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
  で一致した。
- exp501 compact manifest SHAは
  `32317a715997c7a7e145d7122a8ac37733adb30710e571ccbf11a81c2d79c257`。
- exp413 saved OOF controlはRMSE `7.884802794404715`、OOF SHA
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`。
- exp501 Stage Cのdecision
  `FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR`は維持する。

## GPUコスト契約

Kaggle train前に次を再確認し、2026-08-02にユーザーの実行承認を得た。

| 項目 | 数 |
| --- | ---: |
| active treatment variant | 1 |
| LightGBM config | 3 |
| outer fold | 5 |
| 合計新規GPU booster | 15 |
| exp413 control再学習 | 0 |
| exp501 selector再学習 | 0 |
| exp413 signed selector再学習 | 0 |
| HMM / PF / Beam再実行 | 0 / 0 / 0 |

この範囲に限って正規train notebook採用、package、Kaggle runを承認済み。
inferenceとsubmissionはfalseのまま。

## 変更点

- old selector blockを残さずnew selector blockへ置換する。
- feature orderは`clean273, exp501 compact77, signed23`。
- exp413のfold、TVT model family/config、seed、GPU mode、early stopping、評価scopeを固定する。
- selector blend、weight、gate、feature subset、same-OOF rescueを禁止する。

## 再現性メモ

- seed policy: exp413のseed 42とper-config LightGBM seedを継承する。
- stochastic components: 実行済みGPU LightGBM 15本だけ。
- CPU/GPU runtime: Kaggle T4を正とし、`gpu_repro_guard_dp_threads8`を継承する。
- input SHA: exp413/exp501 kernel version、fold/compact/signed/control SHAをconfigに固定済み。
- feature schema/content SHA: removed74、inserted77、retained273/23、final373と10 fold-role matrixを記録済み。
- model manifest / model SHA: 15 model、grid、個別SHAを記録済み。
- prediction SHA: exp413 controlとexp502 OOFを固定済み。
- submission SHA: 範囲外。
- rerun check: 未実行。GPU bitwise一致は前提にしない。

## 引き続き行わないこと

- current-test inference、submission生成、外部提出

## 次のアクション

FAIL_CLOSEで終了。原因説明が必要な場合のみ、保存exp502 OOFによるfold 3/4・hidden-like
transfer attributionを独立した低優先readoutとして別承認で検討する。
