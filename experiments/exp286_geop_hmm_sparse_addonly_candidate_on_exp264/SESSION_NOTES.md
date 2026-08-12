# exp286_geop_hmm_sparse_addonly_candidate_on_exp264 セッションノート

## 目的

Stage 0でoracle headroomが確認できたexp279 `geop_hmm`を、gateではなく修正版exp264の正式な
13番目candidateとして他候補と同じcandidate-long情報付きで追加し、selector再学習が12候補版より
改善するかを確認する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage D T4 version 1完了・pooled改善・安定性guard FAIL
- CV / LB: Stage D full13 add-only `8.4037839136` / submissionなし
- inference / submission: disabled / disabled
- fixed top-25% gate: disabled。Stage 0で失敗したsparse gate分岐だけclosed

## ユーザー指示の解釈

- 固定gateのgain保持率27.71%はselectorに候補を追加した効果ではない。
- ユーザーは「selectorに追加すると改善するか」を実行するよう明示した。
- 追加時は他候補と同様にID、kind/family、availability、信頼度、generic proxyを持たせる。
- この指示をStage A + Stage BのKaggle CPU実行承認とする。Stage C/D、GPU、inference、
  submission、親/control再学習までの承認とは解釈しない。

## Stage B実行コスト契約（push前固定）

- active variant: 1
- LightGBM config: 1（exp264 corrected Stage B v5のstandard config）
- objectives: 2（`pred_abs_error` / `p_within10`）
- outer folds: 5
- 合計selector boosters: `1 x 2 x 5 = 10 CPU boosters`
- Stage A feature audit: 0 booster
- parent/control再学習: 0 booster
- HMM / PF再生成: `0 / 0 well-runs`（保存済みexp279 OOFを読む）
- Stage C nested selector: 0（このStage B runのscope。後に別承認で40 CPU modelsを実行）
- Stage D downstream: 0（このStage B runのscope。後に別承認で15 GPU modelsを実行）
- inference / submission: `0 / 0`
- runtime: Kaggle CPU、GPU/TPU/internet off

## 候補情報契約

- candidate ID: `geop_hmm`、score candidate順の13番目。string IDをartifactに保持し、modelには
  `id__candidate__geop_hmm` one-hotを使う。ordinal candidate indexは禁止。
- kind: `primitive`。
- family: `geop_centered_exact_hmm`。
- availability: exp263/exp279 keyが一致する3,783,989 OOF rowsすべてでtrue。gateなし、補完なし。
- generic proxy: value/finite、last-anchor差、local shape windows 32/128/512、bank median/range/std、
  candidate disagreement/rank、primary/fixed domain統計を共有pipelineで全候補同様に生成。
- native confidence:
  - `geop_hmm_std -> sigma_tvt`
  - `geop_hmm_loglik -> source_loglik`
  - `geop_hmm_loglik / evaluation_rows_in_well -> loglik_per_row`
  - `candidate_finite_source`
  - `confidence_valid`
- source loaderは`id, well, row_idx, fold, geop_hmm, geop_hmm_std, geop_hmm_loglik`だけを読む。
  `true_tvt_readout_only`、error、oracleは読み込まず、feature/confidenceへ渡さない。
- primary legal domain: 11→12、fixed comparison domain: 7→8。既存12候補、formula、fallbackは不変。

## 比較契約

再学習しないexp264 corrected Stage B v5をparent12 baselineとする。

- parent hard primary OOF RMSE: `8.587004386703422`
- parent fixed fallback OOF RMSE: `8.238331546485645`
- parent model count: 10
- selector metrics SHA: `576120d677bdd76bd7c2d40bf73d96bc37c759d79a8acd9a8a58faee43e4bf54`
- candidate metrics SHA: `54a3b160a71a2a1415b7c30c8dd3a3a624a27e00ba990cb0b3bf539001347d3f`
- candidate score OOF SHA: `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`

13候補追加guardは次を全て必須にする。

1. hard primary OOF RMSEがparent12より小さい。
2. hard primary fold RMSEが3/5 folds以上でparent12より小さい。
3. `pred_abs_error` objectiveで`geop_hmm`が1行以上選択される。
4. 13候補のMAE/logloss/Brier prior比較score guardがPASSする。
5. fixed fallback OOF RMSEがparent12と1e-9以内で一致する。
6. `id__candidate__geop_hmm`が採用schemaにあり、4 native confidence + validityが全fold coverage 1.0。

hard/fold pathとshared-12 candidate/fold scoreは直接比較する。all-13 pooled candidate-long scoreは
12→13候補で評価行母集団が変わるため補助指標と明記する。

## 実装変更

- `candidate_contract.yaml`を追加し、core12を不変のまま`geop_hmm`を13番目へ登録。
- `src/geop_hmm_selector_audit.py`を追加。exp279 SHA/schema/key/foldを検証し、FoldBundleへ候補値・
  availability・native confidenceをappendする。
- shared `run_stage_b`へ任意`cache_factory`を追加。既存呼び出しはdefaultのexp263 core12 cacheを維持。
- train Jupytext sourceへStage Bの入力/compute/candidate契約表示、Stage A/B orchestration、
  parent比較、選択率・importance readoutを追加。
- parent metrics/candidate metricsとexp251 selected schemaをKaggle bootstrapへ同梱する。
- 専用testに13候補contract、bundle append、confidence、fold mismatch、raw-test-only schema、
  exact compute scopeを追加。

## push前検証ログ

```text
.venv/bin/python -m py_compile ...
.venv/bin/ruff check --select F821,F401,F841,E722 ...
.venv/bin/pytest -q tests/test_exp286... experiments/exp264_exp263_candidate_confidence_dual_selector/tests/test_exp264_candidate_selector_pipeline.py
# 29 passed

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
make validate-exp EXP=exp286_geop_hmm_sparse_addonly_candidate_on_exp264
# strict validation passed
```

## Stage 0履歴

- canonical kernel: `kentookumura/exp286-geop-hmm-sparse-addonly-exp264-train`
- version 1 / id_no `127856113` / COMPLETE
- 3,783,989 rows / 773 wells / 0 booster
- full 13候補unionはrow / block512 / whole-well oracleを5/5 foldsで改善。
- whole-well full SSE gain `3,254,925.793`、固定gate sparse gain `901,971.228`、保持率
  `27.710961% < 50%`でsparse gate guard FAIL。
- technical/input/artifact SHAはPASS。固定gateの救済gridは行わない。
- この失敗は全候補をselectorへ追加するStage B結果ではないため、ユーザー明示指示でfull-all-well
  Stage Bを同一expのversion 2として再開する。

## 次のアクション

1. packageを`src`込みでprepareし、private/CPU/internet-off、competition source、2 kernel sources、
   bootstrap 3 files、exact 10-booster configを監査する。
2. canonical同一slugへversion 2をpushし、直後にlocal approvalをfalseへ戻す。
3. COMPLETEまで監視する。technical errorだけ同一slugで最小修正し、科学条件は変更しない。
4. outputを取得し、10 models、feature/ID/confidence coverage、selector parent comparison、OOF/model SHAを監査する。
5. 実測でREADME/result/metrics/summary/directionを更新する。Stage C/D/inference/submissionへは進まない。

## Kaggle Stage B version 2実行ログ

- credential check: OAuth、username、legacy key OK。headless API tokenは未設定だがKaggle CLI OAuthで認証。
- canonical version 1 metadataをpullし、id_no `127856113`、private CPU、internet offを確認。
- package: private CPU、internet off、competition source 1、kernel sources 2、bootstrap 3 files、
  `src` helper/pipelineとcandidate contract同梱、exact 10-booster scopeを確認。
- package SHA: config `a35fff2c...7aac`、train source `b2aac178...56ca`、candidate contract
  `8ad970a5...df86`、geop helper `705d392f...67eb`、selector pipeline `60c6ca55...03e8`、
  notebook `c5d7ead9...f1b9`。
- 2026-07-19 17:21:53 JST、canonical同一slugへpushし`Kernel version 2 successfully pushed`。
- push直後status: `KernelWorkerStatus.RUNNING`。
- 重複push防止のためlocal `execution.run_approved` / `kaggle_push_approved`をfalseへ戻した。
- version 2は約12秒で`FileNotFoundError: raw train/test directories were not found unambiguously`。
  competition sourceのmount rootが想定したdirect childではなく、selector Stage Aや学習には未到達、
  trained model 0。Kaggle file inventoryでは`train/`と`test/`自体の存在を確認した。
- technical retryは`/kaggle/input`以下の`sample_submission.csv`を再帰検索し、同じrootの`train/test`と
  horizontal CSVを確認するpath resolverだけを変更する。候補、confidence、fold、LightGBM、guard、
  10-booster scopeは変更しない。同一canonical slug version 3として再実行する。
- path修正後にJupytext、構文、ruff、対象29 tests、strict validation、package exact-cost監査を再実行しPASS。
- version 3 package train SHA `166f4c71...1df6`。2026-07-19 17:27:12 JSTに同一canonical slugへpushし、
  `Kernel version 3 successfully pushed`、直後status RUNNING。local approvalは再びfalseへ戻した。
- version 3はraw path、candidate contract、exp263/exp279/parent artifact SHAを通過。ログ上も13 candidate
  order、`primitive / geop_centered_exact_hmm`、native confidence mapping、exact 10-booster scopeを確認した。
- 約79秒でStage A最初のfoldにて`exp279 geop source is missing exp263 IDs`。exp279 CSVの`fold`は
  exp263 selector outer-fold partitionと別定義で、source fold行数もexp263 fold行数と異なる。model生成0。
- version 4はexp263 `exp226_k16` reference partitionから773 wellのouter-fold assignmentを読み、
  exp279をそのwell mappingでchunk materializeする。exp279自身のfoldはprovenanceとして保持し、
  selector splitには使わない。候補値、confidence、selector、guard、model数は不変。
- version 4の対象30 tests、strict validation、package監査をPASS。helper SHA `3e00b23c...499e`。
  2026-07-19 17:32:01 JSTにcanonical同一slugへpushし、直後status RUNNING。local approvalはfalseへ戻した。

## Stage B version 4完了・生成物監査

- 2026-07-19 17:59:43 JSTに`KernelWorkerStatus.COMPLETE`。summaryまで1,518.253秒。
- 実行scopeは1 variant / 1 config / 2 objectives / 5 folds = 10 CPU models。manifestと選択downloadした
  10 model実体のSHAを全件照合した。このStage B runでは親/control再学習、HMM/PF、Stage C/D、
  GPU、inference、submissionは0だった。
- exp279はexp263 well outer-foldへ分割し、fold rows `757,738 / 756,650 / 756,255 / 757,101 / 756,245`で
  exp263 reference partitionと一致。exp279自身のfold一致率は18.7412%で、別splitであることを記録した。
- feature schemaはparent88列 + `id__candidate__geop_hmm` = 89列。parent-only欠落0、training-only formation hit 0。
- `geop_hmm` availability、confidence validity、`sigma_tvt / source_loglik / loglik_per_row / finite`は
  5 fields x 5 foldsすべてcoverage 1.0。truth columns loadedは空。
- hard primary OOF RMSEはparent12 `8.5870043867`からnew13 `8.4777396073`へ
  `-0.1092647794 ft`改善。fold deltaは`-0.295391 / -0.159647 / -0.299858 / +0.081592 /
  +0.139340`で3/5改善。
- `geop_hmm` selectionはpred-error 737,876行（19.50%）、p-within10 206,557行（5.46%）。全foldで選択あり。
- score guardはexpected-error MAE、within10 logloss/Brierすべてpooled + 5/5 folds PASS。
  shared-12平均deltaも`-0.021569 / -0.002555 / -0.000916`で3指標改善。
- bucket delta new13-parent12はnear `-0.054761`、250-500 `-0.016057`、500-1000 `+0.054285`、
  1000+ `-0.127562 ft`。402/773 wells改善、371悪化、median `-0.016230 ft`。
- selector-addition guardは全項目PASS。ユーザーの問い「追加するとselectorは改善するか」にはyes。
- 一方fixed fallback `8.2383315465`に対しnew hardは`+0.2394080608 ft`、fixed比較のhard guardはFAIL。
  inference/submissionには採用せず、Stage C/Dも自動実行しない。
- 小さいJSON/CSVと10 model txtだけを選択download（9.3MB）。巨大candidate/compact parquetはdownloadせず、
  Kaggle実行ログのfile SHAとmetrics/manifest内SHAを証拠にした。
- primary SHA: candidate score `0baef319...e678`、compact `78fcca2d...cf27`、model manifest
  `fb1415c2...018a`、feature schema logical `187200b9...3281`、comparison `86a5d74c...1c1d`。

## 完了判断

今回依頼されたStage B比較は完了。固定sparse gateはnegative、full candidate selector additionはparent12比positive、
hard selector deployabilityはfixed fallback比negativeという3点を分離して記録する。次に進むなら同じexp286の
Stage C nested compact 40 CPU modelsだが、別実装・別承認が必要であり現runのscope外とする。

## Stage C/D追加承認（2026-07-19）

- ユーザーは、元selector比のStage B改善を根拠にStage Cの後Stage Dまで実行するよう明示した。
- Stage C: 1 variant / 2 objectives / 5 outer / 4 inner = 40 CPU models。
- Stage D: full13 compact add-only 1 variant / 3 configs / 5 folds = 15 GPU models。
- 親/control再学習は0。保存済みexp264 Stage D control 273列とparent12 compact add-only 347列をbaselineにする。
- full13 compactは77列、Stage D新規surfaceはclean base 273 + compact 77 = 350列。
- HMM/PF再生成、inference、submissionは0。Stage C生成物のscore/leakage guard通過後にだけStage Dを実行する。

## Stage C実装・初回push待機

- `run_stage_c`へ既存のfull13 cache factoryを渡すwrapperを追加し、Stage A 89列schema固定後に
  outer 5 x inner 4 x 2 objectives = 40 CPU modelsと25 compact partitionsを生成する構成にした。
- parent exp264 Stage C v6 metrics/fold metricsをSHA固定し、hard RMSEのparent12/new13比較を保存する。
- Stage C packageはprivate、CPU、internet off、competition source 1、kernel source 2、40 models、
  control再学習0を確認。package SHAはconfig `50a5b41...ce0f`、notebook `85aa9d3d...02af6`、
  geop helper `92670ad3...ea58`、metadata `30d42a64...e84c`。
- 2026-07-19 18:25頃の初回pushはKaggleの`Maximum batch CPU session count of 5 reached`で
  version生成前に拒否された。別実験5件がRUNNINGであり、ユーザーの他実行を勝手に停止せず枠待ちとする。
- 待機中にStage D add-only-only 15 GPU model実装を追加。parent/controlは再学習せず、保存済みexp264
  Stage D metrics/fold/bucket/hidden/by-wellを全ファイルSHA固定してparent12との比較に使う。
- CPU枠が1つ空いた後、2026-07-19 18:33:29 JSTに同一canonical slugへStage C version 5をpush。
  `Kernel version 5 successfully pushed`を確認し、重複push防止のためlocal approvalをfalseへ戻した。

## Stage C version 5 technical failure / version 6 retry

- version 5はouter 5 x inner 4 x 2 objectivesの40 CPU modelsをすべて学習した後、親exp264
  `nested_selector_metrics.csv`のfold列を`outer_fold`として読んだため、親比較で`KeyError: outer_fold`になった。
- exp264の保存形式は`fold`、exp286内部の意味はouter foldであり、科学条件やsplitの不一致ではない。
  failed kernelはStage C成果物を公開しなかったため、Stage D入力として再利用できない。
- `fold` / `outer_fold`を共通の`outer_fold`へ正規化し、SHA、必須列、0..4 fold inventoryを
  40-model学習前にpreflightするよう修正した。候補、confidence、fold割当、objective、LightGBM、
  model数は変更しない。同一canonical slug version 6としてtechnical retryする。
- 対象18 tests、F821、構文、Jupytext test、strict experiment/package validationをPASS。package SHAは
  config `1f53d7ca...4042`、notebook `6826bb59...4c1e`、helper `1c1f01f1...7e32`、metadata
  `30d42a64...e84c`。
- 2026-07-19 20:36:46 JSTにversion 6をpushし、直後status `RUNNING`。pullしたmetadataでid_no
  `127856113`、private CPU、internet off、competition source 1、kernel sources 2を再確認した。
  重複push防止のためlocal approvalはfalseへ戻した。

## Stage C version 6完了・Stage D入力監査

- ユーザーの完了連絡後にKaggle status `COMPLETE`を確認。summaryまで`4,108.033秒`、実行scopeは
  1 variant / 2 objectives / 5 outer / 4 inner = 40 CPU models、親/control再学習0だった。
- 40 model manifest（`p_within10` 20 + `pred_abs_error` 20）、25 compact partitions、
  18,919,945 compact rows、49,191,857 outer-valid candidate-long rowsを確認した。
- score guardは3指標すべてpooled + 5/5 folds改善でPASS。outer-valid除外、inner train/valid well分離、
  outer-train=`inner_oof`、outer-valid=`four_inner_model_ensemble`のleakage auditもPASS。
- Stage C hard RMSEはparent12 `8.6525319556`からnew13 `8.4486821528`へ`-0.2038498028 ft`改善。
  fold deltaは`-0.304280 / -0.166570 / -0.592312 / +0.055538 / -0.001129`で4/5改善した。
- compact featureは74から77へ増え、`geop_hmm`の2 objective scoreとprimary error top1 flagを含む。
  `geop_hmm` native confidence 5 fieldsは全5 folds coverage 1.0、truth columns loadedは空。
- Stage D固定SHAはmetrics `8f69a1a4...81d61`、model manifest `c22b98eb...19b0e`、
  compact manifest `cddfe5c6...3c7b5`、compact schema file `68335b9f...d35e`、logical
  `73e7efd5...310ec`。巨大partition/modelはdownloadせず、小型JSON/CSVだけ608KB取得した。

## Stage D push前GPUコスト・再現性監査

- 実行対象は`selector_compact_addonly` 1 variantだけ。3 LightGBM configs x 5 outer folds =
  15 GPU boosters。clean273 controlとparent12 compact347 add-onlyは保存済みexp264 Stage D v3を参照し、
  再学習0。最終surfaceはclean base 273 + full13 compact 77 = 350列。
- Stage C、exp072 full-replay cache、exp145 learned-likelihood cacheの3 kernel sourcesを使用する。
  exp218 source/config、clean273 allowlist、hidden assignment、親Stage D小型5成果物はbootstrap SHA固定する。
- LightGBM modeは`gpu_repro_guard_dp_threads8`で、`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、`n_jobs/num_threads=8`。GPU bitwise deterministicとは扱わず、model/OOF SHAを記録する。
- Stage D postprocessでStage C SHA辞書の参照キー誤りをpush前監査で修正した。対象18 tests、py_compile、
  F821/F401/E9、strict experiment validationをPASS。inference/submissionは0。

## Stage D Kaggle T4 version 1開始

- packageはcanonical `kentookumura/exp286-geop-hmm-sparse-addonly-exp264-tvt-train`、private、T4、
  internet off、competition source 1、kernel sources Stage C v6 / exp072 / exp145の3件。
- package SHAはconfig `720bb2ef...e98c`、train source `3de765a8...e47b`、notebook
  `1419fbfe...9f36`、helper `34f3c1d7...fc4b`、metadata `f85a9156...afd0`。
- 2026-07-19 21:56:44 JST、`--accelerator NvidiaTeslaT4`を明示してversion 1をpush。直後status
  `RUNNING`。pullしたmetadataでid_no `127886849`、T4、private、internet off、3 kernel sourcesを確認した。
- 実行scopeはfull13 compact350 add-only 1 variant / 3 configs / 5 folds = 15 GPU boosters。
  親/control再学習、Stage C再学習、HMM/PF再生成、inference、submissionはすべて0。
- 重複実行防止のためlocal `execution.run_approved` / `kaggle_push_approved`はfalseへ戻した。

## Stage D Kaggle T4 version 1完了・監査（2026-07-20）

- ユーザーの完了連絡後、canonical kernelのstatusが`KernelWorkerStatus.COMPLETE`であることを確認した。
  summary出力時刻はsession開始から`9,600.936秒`。実行scopeは承認どおりfull13 compact350
  add-only 1 variant / 3 configs / 5 folds = 15 GPU boosters、parent/control再学習0だった。
- model manifestは15 entriesで、全modelが`device_type=gpu`、`gpu_use_dp=true`、
  `deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`を保持した。
- parent12 compact add-only `8.4608112376`に対し、full13は`8.4037839136`、
  delta `-0.0570273240 ft`でpooled RMSEは改善した。
- fold deltaは`+0.019882 / +0.141198 / -0.456150 / -0.123783 / +0.105954 ft`で、
  改善は2/5 folds。事前条件3/5を満たさなかった。
- distance bucketはnear `-0.030259`、mid `-0.046842`、1000+ `-0.060865 ft`、
  hidden-likeはspatial `-0.171191`、typewell-purged `-0.163713 ft`で全て改善した。
- well単位は373改善 / 400悪化、median delta `+0.025209 ft`。worst `2d35f86d`は
  parent12 `9.612724`からfull13 `15.475557`へ`+5.862833 ft`悪化し、上限`+0.25 ft`を超えた。
- よってpooled/near/1000+/hidden-likeはPASS、fold数/worst-wellはFAIL、総合promotion guardはFAIL。
  これはpooled RMSE悪化ではなく、平均改善のfold/well安定性不足を意味する。
- compact 77列のgain shareは`76.4885%`、名前に`geop_hmm`を含む3特徴は`0.3996%`。
  top gainは両bank・両objectiveのtop1-minus-anchor群だった。
- ローカル取得は小型9成果物だけに限定した。metrics `1c4dfcd1...4d2421`、fold comparison
  `bcde3255...00450`、model manifest `1ad06b7e...0a22e`、reproducibility manifest
  `d77a6fb6...e774d`をKaggleログ記載SHAと照合した。OOF本体SHAは`0769e600...ec0ae`。
- fixed fallback `8.238332`はStage B hard-selector診断であり、Stage Dの直接比較は保存済みparent12
  compact add-only `8.460811`。結果記録では両者を混同しない。
- exp286はStage Dまで完了。train-side promotion、inference、submissionは行わない。
