# exp393_exp347_practical_numerical_equivalence_audit セッションノート

## 目的

exp347のscalar/batched posterior cell最大差`1.4662743e-5`が、最終posterior mean TVTとMAPに実用上無視できる影響しか与えないGPU float32数値差かを、別の事前固定gateで監査する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage 0 FAILを保持したStage A fold 0もFAIL、branch close
- 親: `exp347_prefix_gr_unary_batched_window_exact_ssm`（terminal closeを維持）
- CV / LB: Stage A fold 0 RMSE `22.866144493 ft` / なし
- Notebook: 正規trainはcompact self-contained Stage 0 + Stage A、正規inferenceはfail-closed
- compact source / tests / Kaggle package / Stage 0・Stage A output: あり / あり / あり / あり
- model / frozen prediction / submission: Kaggle outputに各1 / Kaggle outputに1 / なし

## 2026-07-25 設計

- ユーザーはexp347の差がGPU計算順による微小誤差の可能性を再検討したいと判断し、まずbacklog、実験ディレクトリ、steeringを設計まで作成するよう指示した。
- exp347の判定を後付け変更せず、exp393を独立したpractical numerical equivalence auditとして採番した。
- 同じfrozen unaryにscalar FP32、batched FP32 batch 1、production batch 4を適用する。固定先頭4 windowsのみscalar FP64を原因帰属用に追加する。
- 固定16 windowsのposterior mean TVT差RMSE`<=0.001 ft`、p99`<=0.005 ft`、max`<=0.02 ft`、MAP一致率`>=0.9999`を中心に、既存loss/partition/gradient/update/padding/finite gateをAND評価する。
- posterior cell max errorは保存するが、exp347の`1e-6`gateを再分類したりexp393のpromotion gateへ流用しない。
- Stage 0はaudit 1 / fixed16 / temporary neural model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・parent/control再学習0。
- PASSかつ別承認後だけStage A fold 0の1 neural modelを検討する。今回の承認範囲に実装、GPU実行、Stage A、推論、提出は含まれない。
- 直近のKaggle週次GPU quota不足を記録し、実行時期や別環境を独断で選ばない。

## 再現性メモ

- seed policy: seed 42 + stable SHA256 per well/window/mode。
- stochastic components: 将来のtemporary model初期化、CUDA convolution/reduction。比較前にeval/dropout offでunaryを1回生成しfreezeする。
- parallel policy: worker 0、比較modeの並列実行なし、global RNG共有なし。
- deterministic anchor: false。
- 将来記録: parent/source/config/report、raw input、window/boundary/padding、frozen unary、各mode posterior/readout、comparison report、package/kernel/log SHA。
- Stage 0 model/prediction/submission: 生成しない。

## 2026-07-25 実装

- ユーザー指示`exp393を実装してください`を、Stage 0コード、正規Notebook、fail-closed inference、専用test、静的検証の承認として記録した。Kaggle package/push/run、Stage A、推論、提出は承認範囲外のまま。
- exp347 compact self-contained trainから、mask-first input、fixed-window state、temporary neural unary、scalar/batched exact forward-backwardとstructured objectiveをself-containedに持ち込んだ。不要なStage A学習・outer-valid decode orchestrationは持ち込んでいない。
- 親exp347のsource/config/report/fixed16 window/boundary SHAを固定し、reportのterminal FAILとposterior cell差`1.4662742614746094e-05`を実行前に検証する。
- 16 windowsの入力/view/stateをtruthなしで準備し、seed 42 temporary modelをevalにしてunaryを論理的に1回だけ生成・freezeする。その後にscalar FP32、batch-1 FP32、production batch-4 FP32、先頭4 scalar FP64を順次実行する。
- truthはunary freeze後にfixed16のloss/partition/gradient/AdamW 1-step parityだけのために読む。outer-valid wellとの非重複を先に検証し、outer-valid truth accessは0に固定する。
- row comparison、MAP disagreement、runtime、padding、input、unary、Stage 0 reportと各SHAを保存する。posterior cell `1e-6`は`diagnostic_only`で、AND gateにはTVT/MAP、loss/partition、gradient/update、row sum、invalid、finite、truth/model count、memory/runtimeだけを使う。
- inference Notebookはtest dataを読まず、persistent model 0 / trained fold 0 / submission falseを検証して終了する。
- 親compactとの比較: 親4119行/13章に対しexp393 trainは3758行/10章。必要なinput/model/DPを保持し、Stage A/outer-valid評価章をpractical audit/readout/gate章へ置換した。

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <inference.py>
.venv/bin/python -m py_compile <train.py> <inference.py>
.venv/bin/ruff check <train.py> <inference.py> --select F821
.venv/bin/pytest -q tests/test_exp393_exp347_practical_numerical_equivalence_audit.py
結果: 6 passed, 1 skipped
```

- skip 1件はローカル`.venv`にPyTorchがないためのdtype-generalized scalar FP32/FP64実行test。exp347 scalar/batched production source同一性、契約、親SHA、gate exclusion、fail-closed inference、Notebook構造はPASSした。GPU実行はKaggle T4の別承認後に行う。

## 実行量ガード

- Stage 0完了実績: active audit / temporary model / persisted model / trained fold / LightGBM / booster / PF-Beam / control再学習 = `1 / 1 / 0 / 0 / 0 / 0 / 0 / 0`、fixed windows 16、FP64 diagnostic windows 4。
- version 1はmodel/window実行前に停止したため、上記実績との重複はない。
- Stage Aは当初gate FAILで閉じたが、その事実を保持したまま2026-07-25のユーザー明示overrideでfold 0だけ実行対象へ変更した。

## 2026-07-25 Stage 0 実行承認

- ユーザー指示`実行してください`により、Kaggle T4のStage 0 practical numerical equivalence auditだけを実行承認済みとした。
- 実行量: active audit 1 / fixed windows 16 / FP64 diagnostic windows 4 / temporary neural model 1 / persisted model 0 / trained fold 0 / LightGBM config 0 / booster 0 / PF-Beam 0 / parent・control再学習0。
- selected stageは`stage0_practical_audit`。Stage A/B/C、推論、提出は未承認のまま。
- Kaggle kernelはcanonical id/titleを使用し、internet off、NvidiaTeslaT4、private、run-on-pushに固定する。

## 2026-07-25 Kaggle push結果

- credential preflightはOAuth、legacy key、username `kentookumura`を確認した。
- 最初の長いslug `exp393-exp347-practical-numerical-equivalence-audit-train`は`SaveKernel 400`となり、kernel作成・実行とも0。Kaggleのslug長制約を避けるため、仮説や実験番号を変えずcanonical idを`kentookumura/exp393-exp347-practical-equivalence-audit-train`へ短縮した。
- canonical packageはprivate、internet off、`NvidiaTeslaT4`、run-on-push、親kernel source `kentookumura/exp347-prefix-gr-batched-window-ssm-stage0`。embedded configも`selected_stage=stage0_practical_audit`、`run_stage0=true`、Stage A/inference/submission=falseを確認した。
- package SHA256:
  - `kernel-metadata.json`: `380a111e4d3bf54a90babad5786b9cd2b9686ffb65cd4d52ff04363edaa382ba`
  - `config.yaml`: `a66f1d584610edeac7c4233c6d00dfabff2fb29289dade2fb24c0639124eed66`
  - train notebook: `839b0c590a6e89244b81cfda0e650f679d4a8a73408ab2f3046762b74dc84df0`
- canonical pushはKaggleの`Maximum batch GPU session count of 2 reached`でkernel作成前に拒否された。2026-07-25 02:18:33 UTC時点でexp393 statusは404、したがってStage 0 run、temporary model、artifactはすべて0。
- 同時点でT4の`kentookumura/exp372-exp287-exp335-feature-union-train`（id_no `128530478`）が`RUNNING`。CPUのexp358/exp391もRUNNINGだが、exp393を拒否した直接のKaggle応答はGPU session上限である。
- exp372は別実験の15-booster jobであるため停止していない。Colab等へのruntime変更、別slugでの重複push、Stage A/inference/submissionへの進行も行っていない。

## 2026-07-25 GPU枠回復後の再実行

- ユーザー指示`GPUが空いたので実行してください`により、既承認のStage 0だけを同じT4条件で再実行する。
- 2026-07-25 04:23:46 UTC時点でCLI上のexp372 statusは`RUNNING`表示だったが、status反映遅延の可能性があり、ユーザーのGPU空き確認を優先した。
- 元canonical id `kentookumura/exp393-exp347-practical-equivalence-audit-train`はpush前status 404、pull 500。pushを2回確認していずれも`Notebook not found`、API詳細はinvalid kernel/competition/dataset sourceがすべて空、version 0、URL空だった。
- 親`kentookumura/exp347-prefix-gr-batched-window-ssm-stage0`はpull成功、id_no `128239400`、files取得成功のため、親source欠損ではない。
- Kaggleのmine検索ではexp393にref/title/authorを持たない`[Private Notebook]`のghost recordだけが返った。status 404かつversion 0で実行実体はないが、元slugへのcreate/update解決を妨げていると判断した。
- 実験番号・仮説・package内容・T4条件を変えず、短い意味付きretry id/titleを`kentookumura/exp393-exp347-practical-eq-audit-train` / `exp393 exp347 practical eq audit train`とする。旧slugの404、ghost record、retry理由を記録し、別実験には分けない。
- 上記retry idへのversion 1 pushは成功し、id_no `128543320`、private、internet off、`NvidiaTeslaT4`、親kernel source一致をKaggle pullで確認した。
- version 1は約22秒でERROR。setupとStage 0-only execution contractの表示後、temporary model生成・window audit前のparent input guardで停止した。`data.parent_config`のactual SHAはKaggle上のexp347 Stage 0実行時config `376c03da...51f`、expectedは後日更新されたrepo側config `0ab3054f...ebe`だった。他の親source/report/window/boundary SHAに不一致はない。
- 修正は`data.parent_config.sha256`を実行済みStage 0 inputの`376c03da9b6e122cd9fe32c95f3edf079fca5e7126e13aa72700307809bbb51f`へ固定し、ローカル候補も`kaggle/output/stage0_v1/config.yaml`を先頭にするinput identity correctionだけ。gate、window、dtype、batch、padding、kernel、model、実行量は変更しない。
- version 1実績はactive audit 0 / temporary model 0 / audited window 0 / persisted model・fold・LightGBM・booster・PF-Beam・親control再学習各0。technical retryとして同じkernelのversion 2へ進む。
- parent evidence targeted test `1 passed`、strict experiment validation、再prepareをPASS。version 2 package SHAはmetadata `7a812ac14fbfc39f48f645495bfafb9dd7dfa4d014e8703cca39df51db2fe49d`、config `864d60205e360ed2b052eb14ca2f6c2217e56bfa7dd32e26bc0a610c7913d4e8`、notebook `51766184c589f457fd3c70d3a4c081557c6ba4f80dc5fc7ef0070be1276d9c06`。
- 同じkernel idへversion 2 push成功。Kaggle T4で`RUNNING`を確認した。

## 2026-07-25 Stage 0 version 2 完了

- Kaggle private T4 version 2（id_no `128543320`）は`COMPLETE`。audit runtime `0.04458658547 h`、peak GPU memory `0.2416973114 GB`。
- 事前固定13 checksは10 PASS / 3 FAIL:
  - FAIL: posterior mean TVT RMSE `0.007435773763 ft > 0.001 ft`
  - FAIL: posterior mean TVT max abs `0.191623402956 ft > 0.02 ft`
  - FAIL: posterior row-sum max error `2.958618262e-05 > 1e-05`
  - PASS: TVT p99 abs `0.0 ft <= 0.005 ft`
  - PASS: marginal MAP agreement `1.0 >= 0.9999`、disagreement 0
  - PASS: loss/partition `2.384185791e-07 <= 1e-6`
  - PASS: gradient/AdamW update `1.484295353e-08 <= 1e-5`
  - PASS: finite率1、invalid 0、outer-valid truth access 0、Stage A model 0、runtime、memory
- legacy posterior cell max差は`1.519918442e-05`でexp347の`1e-6`診断を満たさないが、事前契約どおりexp393 promotion gateには含めていない。
- report SHA256は`14f646a9d835bf0d724dc1efcd59c9dbaa7fdaa28a56417819a45b85877794db`。13 checksとruntime/memoryの実ファイル確認が必要なため、output archive全体ではなくStage 0 report JSONだけを`kaggle/output/stage0_v2/artifacts/`へ取得した。
- decision=`fail_close_without_threshold_dtype_batch_padding_or_kernel_rescue`。exp347のterminal FAILを維持し、Stage A、推論、提出は0。

## 2026-07-25 ユーザーoverrideによるStage A採用

- ユーザー指示`ずれていようがアイデアを採用してStage Aに進みたいです`を、Stage 0の3件FAILを理解して数値差を受容し、科学的価値をfold 0で検証する明示承認として記録した。
- Stage 0のposterior mean TVT RMSE、TVT max abs、posterior row-sum FAIL、exp347のterminal FAIL、report SHA `14f646a9d835bf0d724dc1efcd59c9dbaa7fdaa28a56417819a45b85877794db`は変更しない。
- 閾値、dtype、batch 4、padding、kernel、window、boundary、objective、architecture、optimizer、10個のStage A science gateは変更しない。
- 実行量はactive variant 1 / architecture 1 / fold 0 / seed 42 / neural model 1 / persisted model 1。LightGBM config 0 / booster 0 / PF-Beam 0 / 親・control再学習0。
- exp347のStage A学習、outer-valid prediction freeze、post-freeze readoutをself-contained trainへ移植した。推論Notebookはfail-closedのままで、Stage B、推論、提出は未承認。
- exp347 Stage 0実績からの予測はp50 `4.741982 h`、保守値`5.108737 h`、実行上限`8.5 h`。
- Jupytext往復、`py_compile`、`ruff --select F821`、専用test `7 passed, 1 skipped`を通過した。skipはローカル環境にPyTorchがないためのdtype実行testで、Kaggle T4実行を正とする。

### Stage A push前ガード

- 実行variant数1、LightGBM config数0、fold数1、合計booster数0、neural model数1、persisted model数1をconfig・package validatorで再確認した。親・control再学習は0。
- canonical kernel `kentookumura/exp393-exp347-practical-eq-audit-train`をpush前にpullし、id_no `128543320`、T4、private、internet offを確認した。statusは`COMPLETE`。
- version 3候補packageはT4、run-on-push、competition source 1、kernel source 3（exp347 / exp209 / exp115）。embedded configは`selected_stage=stage_a_fold0`、`run_stage_a=true`、Stage 0 / Stage B / inference / submission=false。
- package SHA256:
  - `kernel-metadata.json`: `608181c2af9142db709e3b4ac629c7b235b7bc3e09b11c37103934d5e6638511`
  - `config.yaml`: `8f186395d02df96ada76060a304dfb46d9a6520789633a6da2f4dd58497c0b7d`
  - train notebook: `98c4147eb24b9f0142996734b0d5e8a2619295820960c2e5d461bac6cfc680a9`

### Stage A version 3 technical stop

- 同じcanonical kernelへのversion 3 pushは成功したが、約17秒でERROR。Stage A契約表示後、入力読込・model生成・window学習前の共通GPU guardで停止した。
- 原因はStage A追加時に、旧Stage 0専用の`execution.run_stage0=true`検査が`require_kaggle_gpu`へ残ったこと。embedded configの`selected_stage=stage_a_fold0`と`run_stage0=false`は正しく、科学契約やKaggle入力の不一致ではない。
- 修正は共通GPU guardが`validate_selected_stage`の承認済み`stage0_practical_audit`または`stage_a_fold0`を受け入れるようにしたstage dispatch 1点だけ。閾値、dtype、batch、padding、kernel、window、boundary、objective、architecture、optimizer、実行量は変更しない。
- version 3実績はtrained model / persisted model / trained fold / LightGBM / booster / PF-Beam / 親control再学習すべて0。technical retryとしてversion 4へ進む。
- 修正後はJupytext往復、`py_compile`、F821、専用＋notebook test `12 passed, 1 skipped`、strict experiment validationをPASS。version 4候補package SHAはmetadata `608181c2af9142db709e3b4ac629c7b235b7bc3e09b11c37103934d5e6638511`、config `8f186395d02df96ada76060a304dfb46d9a6520789633a6da2f4dd58497c0b7d`、notebook `3d47c58eb932d082de1faecdac36be995d1550b39f3f4876ff54f0f8ff5b3a89`。

### Stage A version 4 実行

- 同じcanonical kernelへversion 4 push成功。2026-07-25 05:03:48 UTC時点でKaggle T4は`RUNNING`。
- version 3の約17秒ERROR時間を越えて`RUNNING`を維持した。実行中の通常logsは空だが、この環境のKaggle CLI既知挙動であり、空ログだけを失敗根拠にしない。
- 完了後に通常logsを取得し、CV、control、10 gate、runtime、model/prediction/manifest SHAを記録する。Stage B、推論、提出は実行しない。

## 2026-07-25 Stage A version 4 完了

- Kaggle private T4 version 4（id_no `128543320`）は`COMPLETE`。fold 0 / seed 42 / neural model 1 / persisted model 1を実行し、LightGBM config・booster・PF/Beam・親control再学習は0。
- 5 epochsを完了し、early-stop window objective最良のepoch 3（`1.5436597614`）を選択した。train `8114.081 s`、全runtime `3.830431 h`、peak GPU memory `7.495397 GB`。
- outer-train fit / early-stop / outer-validは`556 / 62 / 155 wells`。outer-valid `780,457 rows`を全well予測freeze後にtruth読込し、freeze前truth access 0、forbidden neighbor source 0、1 horizontal source/wellを確認した。
- Stage A checksは8 PASS / 3 FAIL:
  - FAIL: real GR RMSE `22.866144493 ft`に対しexp209 `12.671086935 ft`。改善要求`>=0.25 ft`に対し`10.195057557 ft`悪化。
  - FAIL: well RMSE p95 `43.017462701 ft`に対しexp209 `26.301518476 ft`で`16.715944225 ft`悪化。
  - FAIL: maximum well regression `75.227871352 ft > 10 ft`。worst well `44441e54`はreal `76.693842609 ft`、exp209 `1.465971257 ft`。
  - PASS: real RMSEはgeometry `32.465005034 ft`より`9.598860542 ft`良い。
  - PASS: real true-state NLL `14.321157507`はshuffle `23.796372196`より`9.475214690`良い。
  - PASS: within10 mass `0.517559900`はshuffle `0.181425730`より`0.336134170`高い。
  - PASS: target-in-grid 1、prefix clamp error 0、finite coverage 1、runtime、memory。
- hidden-like spatialはreal `23.181685534` vs exp209 `12.761284104`、hidden-like typewell-purgedは`22.525247782` vs `12.046808374`、distance 1000+は`24.761298515` vs `13.878414072`。すべてexp209より約10–11 ft悪く、全体RMSEだけの偶発ではない。
- decision=`close_stage_b_without_exp347_rescue_grid`。Stage B、推論、提出、同family rescue gridを実行しない。Stage 0の3 FAILとexp347 terminal FAILも維持する。

### Stage A生成物とSHA

- model: `3c71deec787ea236a562d3e0aa9add68e792a062b5428c6b3921592cbd3ce598`
- frozen prediction gzip / decompressed: `14b3575f98b98c0c5172ad3b7fc098361d7ee990a2130c7d6454d8b38183838b` / `6c38315da08e0b0c2c14f62e6f824b95465aa13ce064e29ff10e36f880d01c1d`
- model manifest: `9d8b1cceffef5c80a3046fc6bebedd2c53ab9e87c7079e8951abd4ff17fdbcf5`
- Stage A metrics: `7ec9077952b37a9ac87048ca1531c795bae3b31274580c426d20db7063e2cd45`
- summary: `f55f24931de94ba994e0d2c1ad09e66b4ade39db54a133019240e25aa3ae23e3`
- input / window schedule / teacher boundary / emission posterior manifest: `fcd4c46a...6749` / `476e3b03...389c3` / `e568498d...89d1` / `2703fc7f...e75`
- output archive全体は取得せず、小型metrics/manifest 12 filesとkernel logだけを`kaggle/output/stage_a_v4/`へ取得し、ログ表示SHAと実ファイルSHAの一致を確認した。model、frozen prediction、validation readout本体は大きな後続入力が不要なFAIL branchのため取得していない。
- 記録更新後の専用＋notebook testは`13 passed, 1 skipped`、strict experiment validationとtemplate validationもPASSした。

## コマンドログ

```text
make new-steering EXP=exp393_exp347_practical_numerical_equivalence_audit
make new-exp EXP=exp393_exp347_practical_numerical_equivalence_audit
```

## 次のアクション

1. exp393を`stage_a_failed_branch_closed`として終了する。
2. Stage B、推論、提出、exp347 rescue gridへ進まない。
3. 次の実験はこのneural unary familyの微調整ではなく、既存の高優先度backlogから選ぶ。
