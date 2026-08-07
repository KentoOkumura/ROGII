# exp372_exp287_exp335_feature_union_on_exp264 セッションノート

## 目的

corrected exp264の`clean273 + saved74`へ、exp287のfold-safe formation 74列と
exp335のstrict-nested signed residual 23列を同時に追加し、単独親では拾えなかった
foldごとの相補性が444特徴のdownstream LightGBMで改善につながるか検証する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle T4 version 2完了、科学gate FAIL維持、推論・scoring完了
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 統合元: `exp287_fold_safe_formation_74_addonly_on_exp264` /
  `exp335_signed_residual_meta_on_exp264`
- CV / Public LB: `8.071563864946972` / `7.587`
- Notebook: compact self-contained train候補を正規Notebookへ採用済み
- shared pipeline / tests / Kaggle package: あり / あり / あり
- completed GPU boosters: `15 / 15`
- technical / incremental / tail / promotion gate: `PASS / FAIL / FAIL / FAIL`
- implementation / completed train / inference / observed submission: `true / true / true / true`

## 2026-07-24 設計確定

- ユーザー指示「3から進めていい」により、事前提案した0-booster相補性診断を
  必須の先行gateにせず、直接の444特徴unionを新規expとして設計する。
- 特徴順序は
  `clean273 + saved exp264 compact74 + saved exp287 formation74 + saved exp335 signed23`
  に固定する。
- train時はexp287保存済みformation 10 partition、exp335保存済みsigned 25 partition、
  exp264保存済みcompact 25 partitionをmanifest・partition SHA検証後に使う。
  formationやsigned feature、親selector、signed selectorは再生成・再学習しない。
- 変更するvariantは`formation74_signed23_union_addonly`の1件だけ。
  LightGBM family、3 config、5 outer folds、target、seed、early stoppingは親と同一にする。
- 将来の学習量は
  `1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters`。
  exp264 control、exp287/exp335単独親、selectorの再学習はすべて0。
- 比較controlは保存済みexp264 / exp287 / exp335 OOFとし、学習特徴には使わない。
  best standalone CVはexp287 `8.136708220359452`、Public-LB referenceはexp335
  `7.517`だが、Public LBは設計・gate選択に使わない。
- incremental utilityはpooled `<=8.116708220359452`、best-of-exp287/335 fold比
  `<=+0.02 ft`を4/5 folds、5固定scopeすべて`<=+0.02 ft`、formation/signed両familyの
  nonzero gainを要求する。
- promotionは上記に加え、exp264比by-well p95非悪化、worst`<=+0.25 ft`、
  clean273比`+1/+3/+5 ft`悪化well数`<=135/39/14`をすべて要求する。
- scientific FAIL時は同じOOFでのfeature/config/weight/gate救済を行わず閉じる。
  inferenceは将来の別明示overrideなしには行わない。

## 再現性

- seed: 42
- stochastic component: 将来のGPU LightGBM学習のみ
- 保存入力:
  - exp264 compact manifest SHA:
    `f4855726de446b8308a8acf80d6ff6cd6a789f18ef90e165b98fa05d12aecf1c`
  - exp287 formation manifest SHA:
    `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
  - exp335 signed compact manifest SHA:
    `237486930a0e6f7479d40d2b2d2ccb8e033e3787eb273c406d1eb5a3fc8a6b64`
  - exp264 / exp287 / exp335 OOF SHA:
    `b11c5005...39ae2` / `8f026c5c...c3913` / `8b28a3f2...769b1`
- GPU LightGBMのbitwise reproducibilityは主張しない。kernel version、
  package/bootstrap SHA、444-feature schema/content SHA、15 model SHA、OOF SHAを記録済み。

## 実行量ガード

- active variant: `1`
- LightGBM config: `3`
- outer fold: `5`
- planned / completed GPU boosters: `15 / 15`
- parent/control・standalone親・selector再学習: `0`
- formation/signed train feature再生成: `0`
- inference / submission: `0 / 0`
- Kaggle GPU quotaは設計時点で利用不可。将来の実装後も、Kaggle quota回復または
  Colab実行方針とrun承認が別途必要。

## コマンドログ

```text
make new-steering EXP=exp372_exp287_exp335_feature_union_on_exp264
make new-exp EXP=exp372_exp287_exp335_feature_union_on_exp264 SOURCE=templates/experiment
make validate-exp EXP=exp372_exp287_exp335_feature_union_on_exp264
make update-summary
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp372 --root .
```

初回strict validationはREADMEの必須`## 所見`節がないためFAILした。設計内容を変えず同節を
追加し、再実行でstrict PASSした。design contract auditは444特徴、15 boosters、
control/selector/standalone再学習0、全承認flag false、train/inference Notebook各6-cell
template placeholderを確認してPASSした。summary更新後、doc reviewerはcore evidence
categoriesが全体として存在することを確認した。

設計確定ターンでは、コード実装、Jupytext source作成、Notebook採用、
Kaggle package/push/run、ローカルNotebook実行、推論、提出は行っていない。

## 2026-07-24 実装

- ユーザー指示「exp372を実装してください」を実装承認として記録した。
- `src/feature_union_pipeline.py`へ次を実装した。
  - corrected exp264 saved74、exp287 formation74、exp335 signed23のmanifest/schema検証。
  - 全partition byte SHAとformation logical float32 content SHAのfit前検証。
  - target/errorを開く前の
    `clean273 -> saved74 -> formation74 -> signed23 = 444` schema freeze。
  - fold/roleごとの`id/well/fold/role` exact alignmentとfinite matrix check。
  - 保存済みexp264 / exp287 / exp335 OOFのSHA・fold・truth alignment。
  - 1 variant × 3 configs × 5 folds = 15 modelの学習・OOF・importance・SHA保存。
  - technical / incremental utility / tail promotionの固定AND gate。
  - added family間のduplicate/correlation report。pruneやgridは行わない。
- `*_compact_selfcontained_train.py`と変換先`.ipynb`を別名で作成した。
  正規`*_train.ipynb`は既存placeholderを維持した。
- `tests/test_exp372_feature_union_on_exp264.py`を追加し、cost、444列順序、
  alignment、matrix組み立て、duplicate report、独立gate、承認flag、入力SHAを検証した。
- 親compactとの比較:
  - exp335 Stage S compact: 398行、markdown section 9相当。
  - exp372 train候補: 8役割章、入力契約・schema freeze・学習・gate・重要度・再現性を展開。
  - 同一実験helperを呼ぶだけの薄いNotebookではなく、上位orchestrationをセルへ展開した。
- 実装後もtrain run、正規Notebook採用、package、push、inference、submission flagはfalse。

検証:

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp372_exp287_exp335_feature_union_on_exp264/\
exp372_exp287_exp335_feature_union_on_exp264_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <same-source>
.venv/bin/python -m py_compile src/feature_union_pipeline.py <train-source> <test-source>
.venv/bin/ruff check src/feature_union_pipeline.py <train-source> <test-source>
.venv/bin/pytest -q tests/test_exp372_feature_union_on_exp264.py
.venv/bin/pytest -q tests/test_exp264_candidate_selector_pipeline.py \
  tests/test_fold_safe_formation_pipeline.py \
  tests/test_exp335_signed_residual_meta_on_exp264.py \
  tests/test_exp372_feature_union_on_exp264.py
make validate-exp EXP=exp372_exp287_exp335_feature_union_on_exp264
make validate-template
make test
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp372 --root .
make update-summary
```

- Jupytext `--test`: PASS
- py_compile / Ruff: PASS
- 専用pytest: `8 passed`
- exp264/exp287/exp335/exp372関連回帰: `43 passed`
- strict experiment validation: PASS
- template validation: PASS
- doc reviewer: core evidence categories present
- 全体suite: `817 passed / 6 skipped / 2 failed`。失敗2件はいずれも既存exp296の
  完了後configに対して旧`kaggle_cpu_*` statusと`run_variant=true`を期待する
  `tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`であり、exp372差分外。

ローカルNotebook本体の実行とモデル学習は行っていない。

## 2026-07-25 train実行承認

- ユーザー指示「実行してください。」を、実装済みcompact self-contained train候補の
  正規Notebook採用、Kaggle package作成、push、および1回のKaggle T4 train run承認として
  記録した。
- 実行対象は`formation74_signed23_union_addonly`の1 variant、
  LightGBM 3 configs、5 outer folds、合計15 GPU boosters。
- exp264 control、exp287/exp335 standalone、parent/signed selectorの再学習は0。
  保存済みformation/signed train featureの再生成も0。
- 正規kernel ID / titleは
  `kentookumura/exp372-exp287-exp335-feature-union-train` /
  `exp372 exp287 exp335 feature union train`。
- runtimeはKaggle `NvidiaTeslaT4`、internet無効、`run_on_push=true`。
- Jupytext起点の正規train Notebookは18 cells（code 8 / markdown 10）で、
  SHA256は`ecbfb1d3d581af793e74961a268c48473de4a7c7e4844c1234157b8bcbe321f1`。
- push packageはbootstrap cellを加えた19 cells、support files 29件。
  - package Notebook SHA256:
    `6430e1401bd018634d8e08a385dc92b342128613c19162046ca6b72baaac95d6`
  - `kernel-metadata.json` SHA256:
    `ad01345b0a3516b3cdd19892bb360b7b6d8f2526545bdbbe66ca88d374059cf5`
  - embedded `config.yaml` SHA256:
    `ea7cee1284ec8e0a0f1103ef536498bc63586c5ac36d326bb4d93ce5ad81058b`
  - embedded `src/feature_union_pipeline.py` SHA256:
    `75f3e0388f43decadcb37b18faac14c08c61148c30906e19e9c484e2dd0dbc1d`
  - embedded Jupytext source SHA256:
    `e2b182ec17b8e330a2bc5faebed5540da3e7970c5f9895721fef9febd617da80`
- metadataのID/title、T4、internet無効、run-on-push、7 kernel sourcesと、
  embedded configの`1 / 3 / 5 / 15 / retraining 0`をpush直前に照合した。
- 2026-07-25 01:09:39 UTC（10:09:39 JST）に正規IDへT4指定で初回pushし、
  Kaggle kernel version 1を開始した。
- pull-back metadataはcanonical ID/title、`enable_gpu=true`、
  `machine_shape=NvidiaTeslaT4`、internet無効、7 kernel sourcesを保持していた。
- pull-back Notebookは19 cells / support files 29件で、embedded config SHA
  `ea7cee128...1058b`とpipeline SHA`75f3e038...bc1d`がpush前packageと一致した。
  Kaggle正規化後のNotebook SHAは
  `af2515fb8049c8288eec866d9902b2fd716ab0841b48b8f9f714f8dec36077f2`。
- push後は重複実行防止のためローカル`execution.run_train=false`へ戻した。
- version 1はKaggle log時刻619.970949秒で`ERROR`となった。prefitで
  `load_parent_compact_fold`からexp264 `load_stage_d_compact_fold`へ渡したevidenceに
  `compact_features`がなく、`KeyError`で停止した。LightGBM boosterは0本。
- 完全ログを`kaggle/output/train_v1/logs.json`へ保存した。SHA256は
  `89d00097a675bb0373f3e1d4b464aeaa6a4d1d403f89043e369d3458607da8d8`。
- 原因はexp372 verifierの標準化key `features`と、再利用したexp264 loaderの
  `compact_features`契約の境界adapter不足。74 unique列を確認後にkey変換する修正と
  専用回帰testを追加した。
- 修正後検証は専用`9 passed`、exp264/287/335/372関連`44 passed`、
  py_compile / Ruff PASS。
- steeringのtechnical retry別承認ルールに従い、
  `kaggle_push_approved=false`、`train_run_approved=false`、
  `technical_retry_approved=false`へ戻した。version 2は未push。
- technical failureとretry承認待ちを正規Jupytext sourceへ反映して18-cell Notebookを
  再生成した。修正版正規Notebook SHA256は
  `56247754817d1b26ef09b7b82808293ea7327bdb2e0d640247ef229720121ef9`。
- ユーザー指示「再実行してください」をversion 2のtechnical retry承認として記録した。
  実行契約はversion 1と同一の1 variant / 3 configs / 5 folds / 15 GPU boosters。
  exp264 control、exp287/exp335 standalone、selector再学習、formation/signed feature再生成は0。
  canonical kernel ID/titleとKaggle T4、internet無効も変更しない。
- version 2のpackage、同一canonical IDへのpush、train runだけを承認範囲とする。
  inference、submission、scientific FAIL後のsame-OOF rescueは未承認のまま。
- retry承認を正規Jupytext sourceへ反映し、18-cell Notebookを再生成した。
  version 2正規Notebook SHA256は
  `33f0978521812059e65c80b80da21bb2b7f661da28c05de6306275c5201cc95a`。
- version 2再push前に同一canonical IDをpull-backし、version 1のKaggle
  `id_no=128530478`、T4、internet無効、7 kernel sourcesを再確認した。
- version 2 packageはbootstrap cellを加えた19 cells、support files 29件。
  - package Notebook SHA256:
    `f8dab2b5044ec22691d9c99b20f4c7d8a419867da7ec55200ece63f2d2315ecf`
  - `kernel-metadata.json` SHA256:
    `ad01345b0a3516b3cdd19892bb360b7b6d8f2526545bdbbe66ca88d374059cf5`
  - embedded `config.yaml` SHA256:
    `7290a1c8848cd3d5f6b7a8a4ef93ec3d5699c9c0f6e25c3d42042d82f8d45e47`
  - embedded fixed `src/feature_union_pipeline.py` SHA256:
    `7fa27865dbcf12d9987547452742ad1352d0c4ecfaaf2f5f9bfc5cc427caa4f7`
  - embedded Jupytext source SHA256:
    `fd1c9004740abc2b9b45988884c267ec0f41696067a7626031f4d11d056b41b0`
- version 1 pipeline SHA`75f3e038...bc1d`とは異なる修正版が埋め込まれていること、
  metadataのID/title/T4/internet/run-on-push/7 sourcesと、embedded configの
  `1 / 3 / 5 / 15 / retraining 0`をpush直前に照合した。
- 2026-07-25 01:34:15 UTC（10:34:15 JST）に同一canonical IDへT4指定で
  version 2をpushした。
- pull-backは`id_no=128530478`、canonical ID/title、T4、internet無効、
  7 kernel sources、19 cells / support files 29件を保持した。
- pull-back embedded config/pipeline SHAはpush前と一致した。Kaggle正規化後の
  Notebook SHAは`4226b6db045106dc639c3c983ef780493b48e636d1e954e3a7e6f38910edba19`、
  metadata SHAは`ce342e24158bf1d93076ccc886862620900667224b598df995e3c1370fe243d1`。
- version 2 push後は重複retry防止のためローカル`run_train=false`、
  `kaggle_push_approved=false`、`train_run_approved=false`へdisarmした。
- Kaggle quotaで拒否された場合、ユーザー確認なしにColabへ切り替えない。
- inference、submission、scientific FAIL後のsame-OOF rescueは未承認のまま維持する。

## Version 2 push時点のアクション（完了）

1. retry承認を反映した正規train Notebookを再生成する。
2. 静的検証後に修正版Kaggle T4 packageを作成し、bootstrap/metadata SHAを確認する。
3. `1 / 3 / 5 / 15 / control 0`契約のままversion 2をpushし、完了まで監視する。

## 2026-07-25 Kaggle train version 2完了

- canonical kernel
  `kentookumura/exp372-exp287-exp335-feature-union-train` version 2
  （id_no `128530478`）はKaggle `NvidiaTeslaT4`、internet無効でCOMPLETEとなった。
- log最終時刻は`18425.058989808 sec`。固定契約どおり
  1 variant / 3 configs / 5 foldsの15/15 GPU boostersを完了した。
- exp264 control、exp287/exp335 standalone、parent/signed selectorの再学習は0。
  formation/signed train feature再生成も0。
- pooled union RMSEは`8.071563864946972`。best standalone exp287
  `8.136708220359452`から`0.06514435541248 ft`改善した。
- fold ensemble RMSEは
  `7.822580 / 8.676376 / 7.662126 / 7.847103 / 8.306610`。
  fold別best standaloneとの差`<=+0.02 ft`はfold 0/2/3/4の4/5でPASSした。
- technical gateは11/11 PASS。3入力manifest/partition SHA、formation logical SHA、
  3,783,989 rows / 773 wells、id/well/fold/role alignment、schema freeze前truth/error読込0、
  444 unique features、finite matrix、15 unique model slotsを確認した。
- incremental utility gateはFAIL。pooled/fold条件とformation/signed両familyの
  positive total gain・5/5 positive foldsはPASSしたが、固定scopeの
  `mid_250_1000`がexp335比`+0.04839954514187905 ft`で上限`+0.02 ft`を超えた。
- tail promotion gateは全条件FAIL。
  - exp264比by-well delta p95: `+2.19802617730974 ft`
  - exp264比worst-well delta: `fb03ae90 +13.023263265570503 ft`
  - clean273比悪化well数:
    `+1/+3/+5 ft = 157/53/23`（上限`135/39/14`）
- promotion gateはFAIL。事前登録の
  `close_without_same_oof_rescue`を適用し、本branchをterminal closeした。
  inference、submission、same-OOF feature/config/weight/gate rescueは行わない。
- OOF、metrics、model manifest、SHAの実ファイル確認が必要なため、version 2 outputを
  `kaggle/output/train_v2/artifacts/`へ取得した。28 files / 457,240,998 bytes。
  reproducibility manifest記載の主要10成果物と15 model fileのSHAは全件一致した。
- SHA256:
  - logs:
    `0f8134297af145f7d4cb2da9bed6fef7c795bac9201e38adb151b16251a5704f`
  - feature schema:
    `049800d626b04f16fbf08eb33e8a980ecbe62008402ff7b24f3e77e04e6ef4e9`
  - model manifest:
    `e0d7f85c34d5c64410fe1b2e641669ee1887346a4cbd754579d0dd7e15875b5a`
  - OOF:
    `635dea78b9bf7ad07a1bef267d37e4e2d1707f648799c1590715d4255c02e6f8`
  - reproducibility manifest:
    `90eeede79b13d39ec3fcf6cb08268e6b396db42cb0e1f9e62fd9dca712ccdb5d`
- GPU LightGBMのbitwise reproducibilityは主張しない。submission生成はfalse。

## 最終記録・検証

1. 実験状態、CV、gate、model/OOF/SHAを実験文書と台帳へ反映した。
2. 完了済みunionを`KAGGLE_DIRECTION.md`のアイデアバックログから削除し、
   科学的negative resultを判断メモへ移した。
3. Jupytext、構文、Ruff、専用9 tests、関連44 tests、strict validationをPASSした。
   実験文書レビューもcore evidence categoriesの存在を確認した。

## 2026-07-25 推論override・実装

- ユーザー指示「推論に進んでください。」を、固定科学gate FAIL後のsaved-model inference
  overrideとして記録した。trainのincremental/tail/promotion FAILは維持する。
- 承認範囲はKaggle private CPU inferenceと提出形式検証用`submission.csv`生成まで。
  外部competition submit、再学習、same-OOF rescue、gate再分類、Colab fallbackは含まない。
- 実行量:
  - raw-test candidate: 12
  - saved parent selector: 40 models
  - saved signed selector: 20 models
  - saved union TVT: 15 models
  - current-test formation生成: 1
  - fitted model / trained booster / control retraining: `0 / 0 / 0`
- raw testからcandidate/confidence、clean273、outer別saved74、outer別signed23、
  all-train-reference formation74を同一runで再生成し、
  `clean273 -> saved74 -> formation74 -> signed23 = 444`で15 union modelへ渡す。
- exp335 compact inferenceを主構成参照、exp287 formation inferenceを追加構成参照として、
  8役割章・1,584行のJupytext percent sourceを作成した。
  - exp287 compact inference: 1,349行 / 8役割章
  - exp335 compact inference: 1,448行 / 7役割章
  - exp372 union inference: 1,584行 / 8役割章
- 18 cells（code 8 / markdown 10）の候補Notebookへ変換し、静的検証後に正規
  `*_inference.ipynb`へ採用した。
- SHA256:
  - Jupytext source:
    `37128c1426bfa1b12c9227e1c01fe89ffe2bbd88c817da07b51c661c61eaaec9`
  - canonical inference Notebook:
    `2d7b592542f36f07b411d7450875ddb67a8a7764880ad34665d27a2a6bc11a44`
- Jupytext `--test`、py_compile、Ruff、専用11 testsはPASS。
- canonical inference kernel ID/titleは
  `kentookumura/exp372-exp287-exp335-feature-union-inference` /
  `exp372 exp287 exp335 feature union inference`。
- 次はcredentials、strict validation、package bootstrap/model/source contractを確認後、
  CPU run-on-pushで実行し、完了まで監視する。
- credentialsはOAuthとlegacy CLI credentialが利用可能。API Tokenは未設定だが、
  Kaggle CLIのOAuth認証を使える。
- strict experiment validation、project config validation、専用11 tests、関連46 tests、
  Jupytext、py_compile、RuffをPASSした。
- bootstrap dependency 18件はmissing 0 / destination重複0。
  inference kernel sources 9件、private dataset source 1件、competition source 1件。
- exp372 train version 2のunion model manifest SHAと15 model file SHAを再検証し、
  mismatch 0、feature count 444、feature schema SHA一致を確認した。
- CPU inference packageはbootstrap cell込み19 cells、support files 42件。
  - package Notebook SHA:
    `59838d3ef7e899c1d85378dee60675aa20ef5339dab4f02703829b0d0023e0e6`
  - metadata SHA:
    `65824e371e7639a8b85e84220180aaefee5e4cc37056ff11327993ff3564dfb6`
  - embedded config SHA:
    `00491716b604e31701d380187d3961666afdde0dd01a991903f0dc1e7f986f0e`
  - embedded support ZIP SHA:
    `316796311ff5e7b66fd913f3f897552716e94354dba9aec5b7e3369c62bb0402`
- metadataはcanonical ID/title、private、CPU、internet off、run-on-push、
  9 kernel sources、1 dataset sourceを保持する。
- embedded configは`run_inference=true`、competition submit false、
  model counts `40 / 20 / 15`、feature counts `273 / 74 / 74 / 23 = 444`を保持する。
- pre-push pullは403となり、同じqueryのkernel listにはtrainだけが表示されたため、
  inference canonical IDは未作成と判断した。slugを変更せず初回pushへ進んだ。
- 2026-07-25 09:07:35 UTC（18:07:36 JST）にcanonical CPU inference IDへ
  version 1をpushした。Kaggle id_noは`128563759`。
- push直後にrootの`run_inference/create_submission`とpush/run approvalをfalseへ戻し、
  重複実行を防止した。pushed embedded configのrun flagはtrueのまま。
- pull-back metadataはcanonical ID/title、private、CPU（machine shape None）、
  internet off、9 kernel sources、1 dataset source、competition sourceを保持した。
- pull-backは19 cells / support files 42件で、embedded config/support ZIP SHAはpush前と一致。
  Kaggle正規化後のSHA:
  - Notebook:
    `ef2a831ab41e6bda33808d70e7aa0cdb741362e8c614bca88827c248672bc290`
  - metadata:
    `93ad64c7239a7ccf2c28b82984d8bcb52e06048fad94d1e42a1c716bb31ca127`
- inference version 1はKaggle log時刻25.549087秒で`ERROR`となった。
  union model manifestの正常値`completed_15_gpu_boosters`を、別artifactで管理する
  科学判定`train_complete_guard_failed_closed`と誤比較した推論ガードのtechnical error。
  feature生成、model predict、model fitには到達していない。
- 完全ログを`kaggle/output/inference_v1/logs.json`へ保存した。SHA256は
  `b1b73c4f946a266d7bb5365f9cded29dfc2920017e2337c8c616a15b9f0e6970`。
- 修正はmodel manifestを`completed_15_gpu_boosters`、train結果を引き続き
  `train_complete_guard_failed_closed`として別々に検証する。15保存modelと444列schemaの
  SHA契約、0 fit、CPU、competition submit falseは変更しない。
- 修正版は専用11 tests、py_compile、Ruff、Jupytext `--test`、strict experiment/project
  validationをPASS。version 2 packageは19 cells / support files 42件で、
  Notebook SHA `6ead654605fc94b242a74aee7ed3282eb6e0386743ed61dd15b390c774247b79`、
  embedded config SHA `06fa9fdd004ab53073a52f363645511cce22feb3a562391729cea149905ab388`、
  support ZIP SHA `a993c7c3778d0bb8d7caa527a5870666bcb1de2fda22b7eb47a5be7b7718e600`。
- 2026-07-25 09:33:06 UTC（18:33:06 JST）に同じcanonical IDへversion 2をpushした。
  push直後にrootのrun/create/push/run approval flagをfalseへ戻した。
- version 2はcandidate 12本、14,151 rows / 3 wellsのraw-test生成を完了後、
  log時刻434.628235秒で`KeyError: selector`となった。exp335 inference由来の
  `model.selector.training.predict_base_row_chunk_size`をexp372 configから読もうとした
  technical errorで、model fitは0。
- 完全ログを`kaggle/output/inference_v2/logs.json`へ保存した。SHA256は
  `1caaa1d541bee7a1d86afcf3331f644671decb337ccba0e5540c1b66bdaf8f3e`。
- source内のroot config参照を全件確認し、同系統で未到達だったsigned top1 parity
  toleranceも含め、chunk size `20000`とtolerance `1e-5`をexp372のinference契約へ移した。
  0 fit、保存model、444列順序、CPU、competition submit falseは変更しない。
- v3 technical fixはJupytext `--test`、py_compile、Ruff、関連29 tests、
  strict experiment/project validationをPASSした。
- v3 packageは19 cells / support files 42件。Notebook SHA
  `5e84ccd8b5289e733a2b212bd99405d21d10b52658b6fea40f02413ba24ea6a0`、
  embedded config SHA `755c791e7461e73ce549bc1116b938885827a3d24c97fc0fb092f6883e6795f0`、
  support ZIP SHA `6be9a531036692b7533c5dd136d00e7f9649460128609df6acd4d61619adc5d9`。
- 2026-07-25 10:20:46 UTC（19:20:46 JST）に同じcanonical IDへversion 3をpushし、
  root実行flagを直ちにfalseへ戻した。
- v3 pull-backは19 cells / support files 42件、embedded config/support ZIP SHA一致。
  Kaggle正規化後Notebook SHAは
  `085a70e2cf5bae59cdce0406cf3ceffa9ca849e103470bf2176168a03c0f923a`、
  metadata SHAは
  `93ad64c7239a7ccf2c28b82984d8bcb52e06048fad94d1e42a1c716bb31ca127`。
- version 3はraw-test生成後、selector missingness guardでlog時刻412.991784秒に停止した。
  exp372の`features`節にcorrected exp264 Stage A v4のraw-context allowlistがなく、
  training-denseな`MD/X/Y/Z`系7列がexpected schemaへのreindexでNaNになったため。
  model predictionとfitには到達していない。
- 完全ログを`kaggle/output/inference_v3/logs.json`へ保存した。SHA256は
  `dfc98eaf752dd8a693e04477927af80215e77a9b9fee8183967aaeccc6e5236b`。
- local `data/raw/test`の3 horizontal CSVは全て`MD,X,Y,Z,GR,TVT_input` header。
  成功済みexp264/exp335 inferenceと同じfixed 88-selector契約として、
  `MD/X/Y/Z/GR` allowlist、delta/typewell、target forbiddenをexp372へ明示した。
  NaN補完やmissingness guard緩和は行わない。
- v4 fixは専用12を含む関連30 tests、Ruff、strict experiment/project validationをPASS。
- v4 packageは19 cells / support files 42件。Notebook SHA
  `3b8e6516b25de70da2b9ab83f1dbb52348a0e6ffcfe54f878ea31d93db232fec`、
  embedded config SHA `8afb33a7fc0c322e888c481e225ec3c4951ea0d9e7294b030d6c44c05892b5c3`、
  support ZIP SHA `d3755d97f529b3f065cec746071517ae21e80df334ff70f778b6f8e93befd10c`。
- 2026-07-25 10:35:31 UTC（19:35:31 JST）にversion 4をpushし、
  root実行flagを直ちにfalseへ戻した。
- v4 pull-backはembedded config/support ZIP SHA一致、Notebook SHA
  `421bf54b305d6d06f12c0c263150c8e69773ae4672fb1f4bbb39ccab12b77b30`、
  metadata SHA
  `93ad64c7239a7ccf2c28b82984d8bcb52e06048fad94d1e42a1c716bb31ca127`。
- version 4は`KernelWorkerStatus.COMPLETE`。notebook runtime `459.376 sec`、
  log最終event `490.494217 sec`。14,151 rows / 3 wells、12 candidates、
  selector 88、`273 + 74 + 74 + 23 = 444`特徴を生成した。
- parent selector 40、signed selector 20、union TVT 15の全slotとmanifest SHA集合が一致。
  booster/model fit 0、formula parity / signed top1 parity最大絶対誤差は`0 / 0`。
- `/tmp`へoutputを取得し、submission、prediction、feature schema、formation、
  outer別parent/signed compactを実ファイル監査した。submissionは実験配下へ常設しない。
- skill checkerとrepository checkerはいずれもsubmit-check PASS。14,151行、
  `id,tvt`、sampleとheader・ID内容/順序一致、重複/NaN/Inf 0、WARN/FAIL 0。
- prediction統計はmin `11591.696289`、max `12239.309570`、
  mean `11905.273438`、std `278.501831`。
- SHA256:
  - logs: `5e405b2015a80a16c8262c54cfadd749d9bcba675e22fab21955359009fbf811`
  - metrics/reproducibility:
    `f5436406513d306fef8e75c63f10b9446fb8cc5ca0204e975eb9f27daa72cd8d`
  - prediction decompressed:
    `5f18bcaf8cdd6952652155c6029c8045272b0b052a69ac8157bbf170aad4bc54`
  - feature schema:
    `bac4d7c539ea6b647c9393c75f00c99349c83887bf644a4a093c3fa89a0116f2`
  - formation logical:
    `cc974f8cc4bd3976b42767fc690a8085389d39d249d73ff3f8e6bdf0c44c9d8c`
  - submission:
    `3688de824db2ae0ff1002fb9c2c9ed8543ed09d4e5bbfdd45d7bbf3c9c7eacdd`
- train科学gate FAILとsame-OOF rescue禁止は維持する。competition submitは未承認・未実行。
- 最終root config/metrics JSON、Jupytext `--test`、py_compile、Ruff、関連30 tests、
  strict experiment/project validationをPASSし、`make update-summary`で台帳を再生成した。
- review-expの全root走査は長時間化したため中断し、同じreviewerの6 evidence categoryを
  exp372のREADME/result/SESSION_NOTES/metrics/config/steeringと横断台帳へ限定して確認。
  purpose、base/change、validation、result、artifact、next actionをすべて確認した。

## 2026-07-26 Kaggle scoring完了

- ユーザーからscoring完了の連絡を受け、`kaggle competitions submissions`と
  submission monitorで最新提出を再確認した。
- `ref=54975325`、submitted `2026-07-25 12:28:12.460000 UTC`、
  status `COMPLETE`、Public LB `7.587`、Private LB未確定。
- inference version 4のterminal観測時刻`2026-07-25 12:15:24 UTC`より後の提出で、
  ユーザー確認済みのexp372 Code submissionとして記録した。
- monitor logは
  `logs/submission_exp372_exp287_exp335_feature_union_on_exp264.log`。
- Public LBはexp335 `7.517`比`+0.070`、exp287 `7.530`比`+0.057`、
  exp264 `7.562`比`+0.025`で、ML Public-LB anchorを更新しない。
- 別routeのexp082 `7.601`は`-0.014`上回るが、routeをまたいだanchor変更は行わない。
- CVでは両単独親を上回ったがPublic LBへ改善は転移しなかった。
  train科学gate FAIL、same-OOF rescue禁止、非昇格判断は維持する。
- 外部competition submitはユーザー確認後に観測したもので、Codexは実行していない。
