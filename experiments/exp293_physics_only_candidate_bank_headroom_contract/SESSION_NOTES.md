# exp293_physics_only_candidate_bank_headroom_contract セッションノート

## 目的

物理モデル単体Public LB 6.5を目指す系列の第1段階として、exp263でcurrent-test再生成済みの
deployable12 candidate bankが、H512 block単位でも目標を支えるoracle headroomを持つか監査する。
初回turnでbacklog、steering、実験scaffold、Stage 2/3/4の固定分岐を作り、追加承認turnで
compact self-contained trainとfail-closed inference候補、contract testsまで実装した。Kaggle実行は行わない。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 2完了・support PASS
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- anchor OOF / Public LB: `8.2383315465 / 7.800`
- 目標: 物理モデル単体Public LB `6.5`
- CV/LB: H512 oracle RMSE `3.6837626642` / 提出対象外
- 実装承認: 2026-07-19ユーザーメッセージ「実装に進んでください」
- Kaggle push承認: 2026-07-19（実行済み・承認消費済み）

## 2026-07-19 設計確定

### 作成コマンド

```bash
make new-steering EXP=exp293_physics_only_candidate_bank_headroom_contract
make new-exp EXP=exp293_physics_only_candidate_bank_headroom_contract
```

### 固定した実行契約

- primary candidate bank: exp263 Stage 1 deployable12。
- candidate内訳: 6 primitive + 5 fixed 50/50 pair + `exp226_w500_50_50`。
- audit粒度: row / H128 / H256 / H512 / whole-well、primary H512。
- anchor: `exp226_w500_50_50` OOF RMSE `8.2383315465`。
- support PASS: H512 pooled `<=5.5`、全fold `<6.5`、全fold anchor改善、6.5必要回収率finiteかつ`<=1.0`。
- active audit contract / LightGBM config / trained fold / booster: 実装契約`1 / 0 / 0 / 0`、実行済みauditは0。
- HMM/PF再生成、GPU、inference、submission: `0 / 0 / 0 / 0`。
- 設計確定時点のnotebook実装、補助コード、test、package、Kaggle run: なし。

### 分岐契約

- exp293 support PASS -> Stage 2 latent-registration GR evidence。
- exp293 support FAIL -> Stage 4 candidate birth。
- Stage 2 PASS -> Stage 3 joint physical semi-Markov smoother。
- Stage 2 FAIL -> stop。Stage 4へ自動分岐しない。
- Stage 4 PASS -> exp293と同じoracle contractを新bankで再監査してからStage 2。

詳細は`downstream_branch_contract.md`を正とする。

## 変更点

- `KAGGLE_DIRECTION.md`未着手backlogへexp293を最優先physical-only 0-booster設計として追加した。
- steering requirements/design/tasklistの未記入placeholderを解消した。
- `config.yaml`へcandidate identity/formula、oracle定義、support条件、分岐、再現性を固定した。
- README/result/metricsを初回design-only状態へ更新した。
- 初回turnでは実装を行わず、追加承認後も正規notebookはtemplateのまま維持した。

## 再現性メモ

- seed policy: primary auditはRNGなし、fold/well/row/candidate stable order。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle private CPU single process、GPU/AMP/internet offでversion 2完了。
- Kaggle kernel id / version: 未作成。
- input SHA: 実行時にexp263 manifest/catalogとprimitive gzip raw/decompressed SHAを記録する。
- feature content SHA: 実行時にdeployable12 candidate content、bank manifest、block assignmentを記録する。
- model manifest / prediction / submission SHA: 生成しないため対象外。
- rerun check: 未実行。deterministic submission anchorではない。

## 次のアクション

1. exp293はsupport PASSとして完了扱いにする。
2. 固定契約どおりStage 2 `prefix_calibrated_latent_registration_gr_evidence`だけを次のbacklogとする。
3. Stage 2のsteering/実験化は別承認を待つ。exp293 rerun、inference、submissionは引き続き禁止する。

## 2026-07-19 compact self-contained実装

### 実装物

- `exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_train.py/.ipynb`
- `exp293_physics_only_candidate_bank_headroom_contract_compact_selfcontained_inference.py/.ipynb`
- `tests/test_exp293_physics_only_candidate_bank_headroom_contract.py`

train候補はexp263 candidate partitionをfile/schema/logical content SHA付きで読み、6 primitive、5 fixed pair、
`exp226_w500_50_50`をfixed float32順で構築する。candidate contentとH128/H256/H512/whole-well assignmentを
gzip raw/decompressed/logical SHAとともにfreezeし、bank再hash一致後だけraw horizontal trainのsuffix TVTを読む。
oracleは12候補のwide squared-error matrixを作らず、100,000 row chunkでrow minimumとgroup SSEを集約する。

inference候補はconfig上のdisabled guardを検証後に`RuntimeError`で停止し、raw test、prediction、submissionを
一切作らない。

### 親notebookとの比較

exp263にはcompact self-contained版がないため、通常train 335行を章立て参照元とした。exp293 train候補は
1,946行、8章、17 notebook cellsで、runtime/config、input/SHA、candidate再構成、freeze、truth loader、
oracle集約、guard、生成物保存をnotebook内に展開した。同一exp helper import、`__file__`、model fitはない。

### 検証

```text
dedicated pytest: 11 passed
repository pytest (実装直後のsnapshot): 298 passed
repository pytest (最終再実行): 296 passed / 2 failed（今回の対象外であるexp294のみ）
strict validate-exp: PASS
validate-template: PASS
Jupytext --test train/inference: PASS
py_compile train/inference: PASS
Ruff train/inference/tests: PASS
```

正規notebookと`settings.py`は標準scaffoldから変更していない。Kaggle package、push、run、output取得、
inference、submissionは実施していない。

最終の全体再実行では、同一workspaceに追加されたexp294の`run_stage0`設定とtest期待値の不一致により
exp294専用test 2件だけが失敗した。exp293専用testは再実行でも11件すべて通過しており、exp294の
ファイルは本実装では変更していない。

## 2026-07-19 Kaggle実行承認

- ユーザーの「実行してください」を、compact train候補のcanonical採用と固定済みKaggle CPU audit
  1回の承認として反映した。inference、submission、候補追加、weight調整は承認範囲外のまま維持する。
- push対象はactive audit 1、LightGBM config 0、evaluation fold 5、trained fold 0、booster 0、
  HMM/PF well-run 0、control/parent再学習なし、private CPU、GPU/TPU/internet off。
- credential preflightはAPI token未設定、OAuth credentialとlegacy credentialはOK。
- 親kernel `exp263-last-anchor-pair-cache-train`（id_no `127474050`）と
  `exp115-hidden-like-spatial-holdout-from-ppt-train`（id_no `124519917`）のmetadata取得に成功した。
- canonical kernelは、長いfull slugでの既存SaveKernel 400事例を避け、意味を保持した
  `kentookumura/exp293-physics-bank-headroom-audit-train` / title
  `exp293 physics bank headroom audit train`を使用する。push前pullは403で、既存resourceなしと判断した。
- compact train notebookをcanonical trainへ採用した。両者はbyte-identical、17 cells、code cell 8、
  cell output 0、execution count 0。canonical notebook SHAは
  `a5d4985b271c63667d4b3e7521de31c05638278b2a9955e29b28e0575acaf589`。push承認をtrainと
  inferenceで分離するguard追加後のSHAである。
- push前検証は専用test 11/11、repository全体298/298、strict experiment/template validation、
  Jupytext train/inference、RuffがすべてPASSした。
- strict packageはbootstrap 1 + canonical 17 = 18 cells。private CPU、GPU/TPU/internet off、
  run-on-push true、competition source 1件、kernel source 2件を確認した。
- executed config SHA: `9a22bec50f58281a22ab0beb48c37c72023cd1f39b3851314ecc8eac46eb9b06`。
- compact train source SHA: `ec589d045ac042a72cd7a74566a024141983efb33aaf4adb705de40885e6b183`。
- packaged notebook SHA: `04086df4088a3cc199458fecc88ab658c21af2f5bc37178e9954844439731a97`。
- kernel metadata SHA: `0a02d65a57d73e6dbf6e7b8773d6383b283578f33b1cd355608fcce19d3f3c59`。
- bootstrap ZIP SHA: `0d6a1795225952c327dddbe65ce0d0faf92ca2eb14ffa2b1d0036a51063bbbcb`。
  loose/package/bootstrap内configとcompact train sourceはbytes一致した。

### Kaggle version 1失敗

- canonical kernel version 1 / id_no `127891171`のpushは成功したが、runtime約9秒で
  `FileNotFoundError: downstream_branch_contract.md was not found`により停止した。
- bootstrapとconfig SHA表示までは成功し、candidate入力読込、truth読込、oracle計算には到達していない。
- 原因はpackage builderがMarkdown文書をbootstrap対象に含めない一方、実行セルが正規分岐文書の
  file SHAを直接要求したこと。
- scientific contract、candidate、閾値は変更せず、正規文書SHA
  `025a81e634b9a46504314bfd9e273bc2e36ed18dd5fa744e7fd3a8b614713819`をtrain内へ固定し、
  local文書が存在する場合は一致を強制、Kaggle package内では固定SHAを使用するversion 2修正とした。
- version 2修正後も専用test 11/11、repository全体298/298、strict validation、Jupytext、RuffがPASS。
- version 2 config SHA: `bb75990e6a144c27d87e6da37db17babe645922ed809a035e2b6c6f02770222d`。
- version 2 compact train source SHA: `dcb8743f9519c5363c9d88695da9630d66e6e22d9058ce00d565476d884b7ae6`。
- version 2 canonical notebook SHA: `bdaf05379acf5f691920e533d8d455d91fce960811949435769d8617b6b25cd3`。
- version 2 packaged notebook SHA: `795c963975d2f3b3e2874ed9d4909560664fd2f5fd63ab11aea0044f5d7fde58`。
- version 2 bootstrap ZIP SHA: `b244f4b515ade6a6af4015beeb39360e2e65c7ffde51cf0e90cc15f790376a24`。
  loose/package/bootstrap内config/sourceのbytes一致を再確認した。

### Kaggle version 2完了

- canonical kernel `kentookumura/exp293-physics-bank-headroom-audit-train` version 2、id_no
  `127891171`が`COMPLETE`。audit runtimeは約200.094秒、notebook全体は約209.393秒。
- 3,783,989 rows / 773 wells / 12 candidates、finite coverage 1.0。technical checks 10件と
  scientific checks 5件はすべてPASSした。
- oracle RMSEはrow/H128/H256/H512/whole-wellで
  `3.446407 / 3.492440 / 3.552829 / 3.683763 / 4.784904`。
- primary H512のanchorは`8.238332`、必要headroom回収率は`0.471825`。
- H512 fold RMSEは`3.998262 / 3.745013 / 3.317686 / 4.117908 / 3.141067`で、
  全foldが6.5未満かつanchor改善。
- H512の1000+ / hidden-like spatial / hidden-like typewell-purged oracleは
  `4.009694 / 3.540513 / 3.531630`で、全risk面でanchorを改善。
- H512 by-well p95は`7.080306`、worstはwell `91b301ce`の`25.698447`。
- candidate bank content SHAは
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`、truth SHAは
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`、oracle readout SHAは
  `69d14a236205eaa1aaafa09abf9bb9b1984797fec54f2fb6533f8243f0a97003`。
- outputのSHA manifest 11件を取得物へ再計算し、file SHA不一致0、gzip decompressed SHA不一致0。
- oracle/selected row prediction、model、inference、submissionは生成していない。
- 固定support判定はPASS。Stage 4は開始せず、次はStage 2
  `prefix_calibrated_latent_registration_gr_evidence`だけを設計する。
