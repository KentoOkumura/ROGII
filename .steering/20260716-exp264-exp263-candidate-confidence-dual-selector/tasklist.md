# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- hard selector、Viterbi、candidate TVTのsoftmax平均。

## 完了

- 2026-07-21、`U=TVT+Z`の符号規約を物理モデル解説へ追記した。`Z`は地下ほど負、`TVT`は下向きに正であるため、
  正の下向き深度`D=-Z`では`U=TVT-D`になることを、実例と全773 wells・5,091,482隣接差分で確認した。
- 2026-07-20、horizontal/typewellでGR補完方法が異なる理由を物理モデル解説へ追記した。観測列と参照曲線の役割、
  アルゴリズム上のNaN処理要件、公開実装からの系譜を分け、現行差が最適性を検証した結果ではないことも明記した。
- 2026-07-20、物理モデル解説へ6 primitive別のhorizontal/typewell GR欠損補完とGRスムージング一覧を追記した。
  補完、TVT grid再サンプリング、GR特徴量の局所平滑化、TVT状態平滑化を区別し、固定blendの継承関係も明記した。
- 2026-07-20、物理モデル解説へGR観測モデルの使用箇所、内部TVT仮説の生成方法、モデル別`σ_GR`推定、
  二乗誤差の統計的前提とデータ適合性、HMM 0.35/41およびK16の継承経緯・妥当性を追記した。
  K16処理をMermaidで図解し、raw train 773 wellsの既知prefix/解像度を読み取り専用で集計した。
- 2026-07-20、exp264 の物理・幾何候補に焦点を当てた `physical_model_summary.md` を追加した。
  共通の `U = TVT + Z` 状態、GR観測モデル、6 primitive、固定blend、selector特徴量、有効な修正版精度、
  hard selectorとworst-well guardの限界を実装・記録と照合し、READMEから参照できるようにした。
- OOF診断2本のKaggle実行について、学習variant 0、model config 0、fold学習0、booster 0、
  parent/control再学習なし、GPU 0、competition submitなしのscopeで明示承認を受領した。
- corrected Stage C v6 / Stage D v3を使うselector-confidence notebook、LikPF 128-path notebookを
  Jupytextから生成し、Kaggle用private/CPU/internet-off packageを準備した。
- corrected Stage D v3 OOFからviewer用`id,tvt` CSVを生成した。3,783,989 unique ID / 773 wells、
  NaN/Inf 0、RMSE 8.460811、viewer loader互換、出力SHA `9fe0cfce...e04b`を確認した。
- Stage C v6 strict nested outer-valid score 45,407,868行を入力に固定し、12候補順・2 legal domain・
  outer fold・inner model count・SHAをfail-closedで検証する実装とした。
- py_compile、ruff F821/F401/E9、Jupytext test、strict exp validation、exp264 15 tests、
  Kaggle notebook 4 testsをPASSした。notebookはローカル実行していない。
- 2026-07-19のOOF診断追加要件をrequirements/designへ固定した。旧Stage D v2を禁止し、final overlayとviewer CSVを
  corrected Stage D v3 OOF SHA `b11c5005...9ae2`へ固定した。

- 修正版Stage D canonical Kaggle T4 version 3を30/30 boostersで完走し、status `COMPLETE`、
  30 models、3,783,989 rows / 773 wells、25 Stage C partition byte SHA preflightを確認した。
- clean 273 control 10.476169に対して347列add-only 8.460811、delta -2.015358、5/5 folds改善。
  near / mid / 1000+、hidden-like 2面も改善した。
- worst `70925e23`が+14.482873悪化し、事前上限+0.25を超えたため総合guard FAIL。
  corrected inference、hard selector、Viterbi、softmax TVT平均、submissionを実行しない。
- small output、metrics/OOF/model manifest SHA、全74 compact特徴の正規化重要度・説明を保存した。
- 修正版Stage D packageのT4/internet off、3 kernel sources、bootstrap config/pipeline/train SHA、
  Stage C v6固定SHA、clean 273 allowlist、30-booster cost contractを監査し、canonical kernel version 3へpushした。
- post-push pullでid_no `127577193`、T4、source、remote bootstrap SHAを再確認し、local run gateを閉じた。
- 修正版Stage Dの2 variants × 3 configs × 5 folds = 30 GPU boostersと、clean 273 matched control
  15本の再学習理由について2026-07-18にユーザー承認を受領した。inference/submissionは含めない。
- 修正版Stage C canonical Kaggle CPU version 6を40 CPU boostersで完走。score guardはexpected-error
  MAE 3.798819、within10 logloss/Brier 0.359412/0.111830で全指標5/5 folds改善しPASS、nested leakageもPASS。
- hard top1は8.652532でfixed比+0.414200、改善1/5 foldsのためFAIL。hard selector、Viterbi、
  softmax TVT平均、submissionの禁止を維持した。
- 40/40 model byte SHA、40期待組合せ、25 partition manifest、18,919,945 compact rows、
  45,407,868 outer-valid candidate-long rows、88列schema SHAを監査した。Stage D fit前に25 Parquet本体の
  byte SHAをKaggle入力上で全件再検証する。
- 修正版Stage C packageのbootstrap config、CPU/internet off、exp263単一source、88列schema契約、
  40 CPU boostersを検証し、canonical kernel version 6へpush。`RUNNING`を確認してlocal gateを閉じた。
- 修正版Stage Cの1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boosters、
  control/親再学習0本の実行承認を2026-07-18に受領した。
- 修正版Stage B version 5を10 CPU boostersで完走し、expected-error MAEとwithin10 logloss/Brierの
  score guardをpooled・5/5 foldsでPASSした。
- hard top1はfixed比+0.348673、0/5 folds改善、hidden-likeとworst-wellも悪化してFAILと確定した。
  selector scoreはcompact内部表現に限定し、hard selectorは禁止を維持する。
- candidate-long 45,407,868行、compact 3,783,989行×74特徴、10/10 model SHAを監査し、
  欠損・nonfinite・確率範囲・fold対応違反0を確認した。
- 全88特徴の説明・重要度・重複・相関を`selector_feature_readout_corrected_stage_b_v5.md`へ確定した。
- 修正版Stage B packageのCPU/internet off/source、config、notebook、pipeline、clean 273 allowlist SHAを
  照合し、canonical kernel version 5をpush。`RUNNING`を確認し、local run gateを閉じた。
- ユーザー選択によりdownstreamをclean 273列に固定し、380列のfold-safe再生成案をスコープ外とした。
- clean 273 allowlist SHA256 `d01a73cc...77bf`、source 380列、selected 273列、compact add-only
  347列のfail-closed契約をconfig/実装/テストに追加した。
- 修正版Stage Bの1 variant × 2 objectives × 5 folds = 10 CPU boosters、control/親再学習0本の
  実行承認を2026-07-18に受領した。
- 修正版Stage A version 4をKaggle CPUで0 booster実行し、600,000 rows、150列から88列を
  logical schema SHA `aaef4ffd...ddd3a4`で凍結した。raw context availabilityはtrain 773/773・
  current-test 3/3 fileでPASS、採用側完全重複0、高相関14組report-only、sparse採用特徴25列。
- version 4の小規模生成物を`kaggle/output/stage_a_v4/artifacts/`へ取得し、catalog/schema/raw availability/
  reproducibility manifestのSHAを記録した。
- feature-level availability auditを実装し、selector 12/100、exp218 downstream 107/380を無効と確定した。
- selector raw allowlistを`MD/X/Y/Z/GR`へ縮小し、actual train/current-test全file headerのfail-closed gateを追加した。

- Stage A/B/CとStage D compact add-only OOFをfeature availability leakageで全無効化し、旧score/RMSE/重要度を意思決定根拠から除外した。
- exp264 steeringと実験ディレクトリを作成した。
- 候補source of truthをexp263へ固定した。
- 6 primitive + 5 pair + fixed 1の12 score candidateを固定した。
- candidate ID 12 one-hot、native confidence + universal proxy、2 legal domainを固定した。
- exp251 v4 295列からのfeature整理規則を固定した。
- dual score、compact adapter、nested stacking、stage別booster数を固定した。
- HMM+LGB、Viterbi、hard-path提出、softmax TVT平均をscope外に固定した。
- 再現性設計を`design.md`と`config.yaml`に記録した。
- Stage A feature builder、exp263 formula loader、raw row/typewell context、confidence coverage、重複/相関監査を実装した。
- exp251 v4 295列のretain/recompute/remove/defer分類表生成を実装した。COPCFは同等raw-test generatorを接続するまでdeferする。
- Stage B dual-objective LightGBM、streaming OOF score、calibration、candidate/distance/by-well readout、feature importanceを実装した。
- 2 legal domainからの74列compact adapterとcurrent-test 5-fold model ensembleを実装した。
- exp264 contract unit test 4件を追加した。
- Jupytext train/inference再生成、Jupytext test、py_compile、ruff、repo全67 tests、template/config/exp strict validationを通した。
- exp264の12候補がすべてexp263 Stage 1でcurrent-test生成済みであることを実出力で再確認し、追加inventoryの旧`train_only_*`表記を`stage0_oof_only_not_in_current_stage1_*`へ修正した。契約テストとKaggle packageも更新した。
- Kaggle CPU Stage A v1を0 boosterで完走した。600,000 candidate-long rowsを監査し、162候補から100特徴を採用、全欠損41・定数5・完全重複16を除外、高相関35組をreport-onlyで記録した。feature schema SHAは`766cfcf1...d4deb`。
- 採用100列中21列がnative confidence依存と確定したため、Stage Bの前提としてexp263 Stage 1 current-test namespaced confidence parityを追加する。
- exp263 Stage 1 v3で21 confidence列、旧15値列、submissionのparityを確認し、current-test confidence前提を満たした。
- Stage A用Kaggle train packageをcanonical id/title、CPU、internet off、`run_approved=false`で準備した。pushは未実行。
- Stage Bの1 variant、2 objectives、5 folds、10 CPU boosters、control再学習0についてユーザー承認を受領した。
- Stage B packageのsource/package SHA、metadata、input sourceを照合し、canonical Kaggle train version 2をpushした。
- Kaggle CPU Stage B version 2を10 boostersで完走した。score guardは3指標すべて5/5 folds改善でPASS、hard top1はfixedより+0.124512でFAIL。
- 独立hidden-like post-hoc readoutを実施し、spatial +0.438111、typewell-purged +0.407604の回帰を確認した。
- diagnostic CSV/JSON/PNGと10 modelを取得し、manifestと全model SHAを照合した。
- 全100特徴の説明・objective×fold重要度・16完全重複・41全欠損・5定数・35高相関組を`selector_feature_readout.md`へ記録した。
- Stage Cの1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boosters、親/control再学習0についてユーザー承認を受領した。
- outer-train内のwell-row-balanced inner 4-fold、inner OOF train feature、4-model ensemble outer-valid feature、25 partition compact出力、40 model manifest、outer-valid score監査を実装した。
- exp264契約テストを6件へ拡張し、Stage C inner foldのwell disjoint・coverage・determinismを確認した。
- repo全78 tests、strict validation、Jupytext、py_compile、F821、bootstrap/config/package SHA監査を通した。
- 既存canonical kernel `kentookumura/exp264-exp263-confidence-dual-selector-train`へversion 3をpushし、
  id_no `127485868`のままStage C notebookがRUNNINGであることを確認した。
- Kaggle CPU Stage C version 3を40 boostersで完走した。score guardはexpected-error MAE、within10
  logloss/Brierの3指標すべて5/5 folds改善でPASS、nested leakage auditもPASSした。
- 40 model組合せと一意SHA、25 compact partitions、18,919,945 compact rows、45,407,868 outer-valid
  candidate-long rowsをmanifest/logで確認した。manifest SHAはmetrics/reproducibility記録と一致した。
- Stage C hard top1は8.420613でfixed 8.238332より+0.182281悪く、hard inference不採用を維持した。
- Stage D 30 GPU boostersとmatched control 15本の再学習についてユーザー承認を受領した。
- Stage DのStage C SHA/25 partition検証、exp218 380列再構築、matched control 380列、compact
  add-only 454列、30-model manifest、OOF/importance/bucket/hidden-like/by-well readoutを実装した。
- cost contractを2 variants × 3 configs × 5 folds = 30 GPU boostersへfail-closedで固定し、
  exp264契約テストを9件へ拡張した。Jupytext canonical train notebookもStage D対応へ同期した。
- private/T4/internet off、Stage C/exp072/exp145の3 input、bootstrap SHAを監査し、別kernel
  `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train` version 1を開始した。
- push後のid_noは`127577193`、statusは`RUNNING`。重複実行防止のためlocal run gateを閉じた。
- version 1は`compact_meta_schema.json`のbyte SHAとJSON内logical SHAを混同して22.5秒で
  学習前ERROR。booster学習は0本だった。
- byte SHA `e3a67761...`とlogical SHA `23614916...`を別契約へ修正し、対象テスト9件、
  strict validation、bootstrap SHAを再確認した。同じkernel version 2をT4明示で開始し、
  v1の失敗時刻を超えた後も`RUNNING`であることを確認した。
- Kaggle Stage D version 2で30/30 GPU boostersを完走したが、後にfeature availability leakageで
  compact add-only 7.805644と全比較を無効化した。
- near/mid/1000+は-0.222916/-0.419414/-0.807155、hidden-like 2面は-1.174830/-1.193025でPASSした。
- worst-well `70925e23`は+17.446742で上限+0.25を超え、Stage D総合guardはFAILと確定した。
- OOF 3,783,989行のfold coverage、欠損/nonfinite 0、pooled RMSEを再計算し、30/30 model SHAと
  reproducibility manifestの8/8 output SHAを照合した。
- 全74 compact特徴の説明と15-model正規化gain/split重要度を`stage_d_feature_importance_readout.md`へ記録した。
- global改善を理由に事前guardを緩めず、compact inference、hard inference、Viterbi、submissionを不採用として実験を閉じた。
- その後2026-07-18にユーザーから推論へ進む明示指示を受領した。guard FAILは保持し、保存済みmodelによるhidden-safe推論成果物生成のみを例外として再開した。
- Stage C version 3の40 selector modelをpagination込みで全件取得し、40/40 byte SHA一致を確認した。推論bundle SHAは`1697e1f7...6b21`。
- Jupytext inferenceをhidden-safe exp263 source-port + Stage C 40 selector + Stage D add-only 15 TVT model構成へ更新した。
- ipynb変換、構文、F821/F401/E9、targeted 10 tests、strict experiment/project validation、Kaggle package SHAを確認した。
- 21.75MBのarchive埋め込みpackageはKaggle API 400でversion作成前に拒否された。Stage C bundleをprivate Datasetへ分離し、416KBのpackageへ再生成した。
- ghost状態の旧slugを避け、新slug `exp264-stage-d-hidden-safe-inference`へversion 1をpushし、RUNNINGを確認した。
- version 1の起動ログを確認し、model推論前のmanifest SHA guardでERRORになったことを記録した。正しいSHAはStage D reproducibility manifestとローカル実ファイルで二重確認した。
- 正しいmanifest SHAでversion 2をpushし、RUNNINGを確認した。
- version 2は395.586秒、selector推論直前に100列一律finite guardでERROR。Stage A採用100列中29列は
  学習時からNaNを持つため、入力parity違反ではなく推論guardの実装不整合と確定した。
- Stage A feature catalog SHA `83c8b953...639d`を推論packageへ追加し、期待NaN保持、`±inf`禁止、
  training-dense列の新規NaN禁止、`conf__`/`formula__`構造欠損率一致を検証するhelperと回帰テストを追加した。
- canonical inference notebookとKaggle packageを同期し、targeted 12 tests、repo 146 tests、Jupytext、
  py_compile、ruff、strict experiment/project validationをPASSした。version 3 package notebook SHAは`9a01bea9...6d63`。
- direct formation 12特徴を除いたStage C v6（88列）とclean 273 downstreamのStage D v3を正として、
  corrected inference version 4を実装・package化した。Stage C 40-model bundleはprivate DatasetへSHA固定した。
- bootstrap 35 filesのcatalog/allowlist/availability SHA、py_compile、ruff、Jupytext、strict validation、
  exp264対象15 tests、repository全161 testsをPASSし、既存private CPU/internet-off kernelへversion 4をpushした。
- version 4は424.511秒でCOMPLETE。12 candidates / 21 confidence / selector 88列・40 models /
  compact 74列 / clean base 273列 / final 347列 / TVT 15 models / training 0 booster / formula parity 0を確認した。
- `submission.csv`はsampleとのheader・row・ID order一致、duplicate/NaN/Inf 0でsubmit-check PASS。
  reference submission ref `54818932`を1回提出した。worst-well guard FAILは維持する。
- scoring完了連絡後にref `54818932`のCOMPLETE / Public LB 7.562を取得した。同一runの自動ref
  `54818883`も7.562。直前ML anchor exp274 7.715を-0.153改善してexp264を新ML LB anchorに更新した。
  別routeのexp082 ensemble 7.601も-0.039で上回るが、ensemble anchorはexp082に維持した。
- PF/HMM/Beam候補を補助meta featureとする主目的基準に合わせ、routeを`ml_model`へ修正した。
  worst-well +14.482873 guard FAILは維持し、train-side採用とは分離した。
- selector-confidence v2の描画差分を無効化し、selector結果だけexp264のprimary predicted-error top1へ
  更新したまま、exp238の3段構成・比較パス・色・線種・exact HMM ±2sigma帯へ完全復元した。
- 修正版selector version 3を同じcanonical Kaggle slugで完走し、3,783,989行 / 773 wells / 773 plots、
  plot manifest、summary SHA、代表PNGを確認した。
- LikPF version 1を500 particles × 128 seeds × 773 wellsで完走し、773 plots、保存済みmeanとの
  全well exact parity、最大絶対差0、plot manifest、summary SHA、代表PNGを確認した。
