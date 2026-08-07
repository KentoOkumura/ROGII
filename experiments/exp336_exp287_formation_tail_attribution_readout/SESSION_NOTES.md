# exp336_exp287_formation_tail_attribution_readout セッションノート

## 目的

exp287のglobal gainとwell-level tail regressionを分けるtarget-free formation信頼性属性が存在するか、保存済みexp287/exp264成果物だけを使う0-booster readoutとして診断する。

## 現在の状態

- Route: `ml_model`診断
- 状態: Kaggle CPU version 2完了、固定gate 0/6 family PASS、branch closed
- risk families: 6
- model / LightGBM config / trained fold / booster / control再学習: `0 / 0 / 0 / 0 / 0`
- CPU/GPU: Kaggle CPU完了 / GPUなし
- CV / LB / prediction / submission: 対象外 / 対象外 / なし / なし

## 2026-07-22 設計

```bash
make new-steering EXP=exp336_exp287_formation_tail_attribution_readout
make new-exp EXP=exp336_exp287_formation_tail_attribution_readout
```

- `kaggle-strategy`でLate phase、ML Public-LB anchor exp287 `7.530`、train-side tail guard FAIL、exp334のwell-balanced loss不十分を確認した。
- `kaggle-review-exp`に従いsteeringを先に作り、その後design-only experiment scaffoldを作成した。
- `docs/06_reproducibility.md`を読み、RNGなし、入力/属性/freeze/metrics content SHA、Kaggle bootstrap確認方針を固定した。
- requirements/design/tasklist、config、README、result、metrics、SESSION_NOTESをdesign-onlyとして確定した。
- scaffoldのtrain/inference Notebookとsettingsはテンプレート生成物のままで、readoutロジックやJupytext sourceは実装していない。
- この設計時点ではKaggle package、push、run、output取得、inference、submissionを実行していなかった。後段の承認・実行記録でStage A/Bだけを実行した。

## 根拠

- exp264 corrected OOF RMSE: `8.460811237612477`、Public LB `7.562`。
- exp287 OOF RMSE: `8.136708220359452`、exp264比`-0.3241030172530248 ft`、5/5 foldsと全scope改善、Public LB `7.530`。
- exp287 worst-well delta: `+8.228409822385604 ft`、`+1/+3/+5 ft`悪化well数`140/40/19`でtrain guard FAIL。
- exp334 OOF RMSE: `8.09349752413077`、exp287比`-0.04321069622868201 ft`、5/5 folds改善。
- exp334はby-well p95`+0.429584617 ft`、exp264比worst`+7.156485377 ft`、`+3/+5 ft`悪化well数`40/19`でtail guard FAIL。well-balanced lossだけではsevere tailを説明できない。

## 固定した科学契約

### Stage A: target-free freeze

- exp287 formation manifestの5つの`role=valid` cacheだけを使い、3,783,989 OOF rows / 773 wellsを1回ずつ構成する。
- plane距離、dense距離、dense不確実性、plane-dense不一致、formation spread、known-prefix calibration errorの6 familyを固定する。
- familyごとのwell scalar、high-risk方向、`numpy linear`四分位境界をtruth/errorなしで作り、manifestとcontent SHAを保存する。
- 四分位境界がstrictに増加しないfamilyはtechnical ineligibleとし、別bucketへ変更しない。
- plane/dense reference availability、feature finite、known-prefix/trajectory、generic signal disagreementはcontext readoutに限定し、primary PASSには使わない。

### Stage B: attribution

- Stage A freeze SHA確認後だけ、SHA固定したexp287/exp264 OOFを読み込む。
- ID/well/fold/actual TVTを照合し、各wellの非加重row RMSE delta `exp287-exp264`を算出する。
- family PASSはglobal Q4-Q1 mean`>=+0.25 ft`、median正、4/5 folds正、hidden-like 2面正、固定coverageのAND。
- いずれか1 familyが全条件PASSした場合だけ、別の単一変更介入実験を設計可能とする。全family FAILならformation attribution枝を閉じる。

## 再現性メモ

- seed policy: RNGなし。well/id/family/scope/fold/quartileをcanonical順にsortする。
- stochastic components: なし。
- CPU/GPU runtime: 設計時点ではKaggle CPU、single worker、BLAS thread 1、internet offを予定した。実績は後段に記録する。
- input SHA: exp287 OOF `8f026c5c...c3913`、formation manifest `25611e28...7772`、model manifest `419dbdf8...7590`、full 421 schema `c1327324...8413`、formation 74 schema `64e8ceb0...914f`、exp264 corrected OOF `b11c5005...9ae2`、hidden-like assignment `5f9ac9fa...6597`を固定。
- attribute/freeze/metrics SHA: 未生成。Stage A freeze SHAをStage B開始前に記録する。
- model/prediction/submission SHA: 非該当。
- deterministic anchor: prediction/submission anchorではない。固定入力に対するdiagnostic reproducibilityだけを主張する。

## 禁止事項

- worst-well ID、truth、prediction、errorをStage A属性や境界へ使わない。
- 同一OOFでthreshold、feature、family、weight、clip、shrink、gateを選ばない。
- formation列削除、救済train、corrected OOF生成、inference、submission、guard緩和を行わない。
- PASSしてもexp287/exp334を自動昇格しない。

## 2026-07-22 実装

- ユーザーの`exp336を実装してください`という明示依頼をimplementationとcompact Notebook候補の承認として記録した。
- `exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_train.py/.ipynb`へ次を実装した。
  - Stage A専用のSHA/schema/identity/finite/outer-valid監査。
  - 6 primary familyのNumPy linear p90またはwell定数max集約。
  - raw horizontal CSVの`MD/X/Y/Z/TVT_input`限定読み込みとcontext readout。
  - target-free属性のcanonical CSV SHA、strict quartile edge、freeze manifest。
  - freeze manifest SHAを明示引数で再検証した後だけOOFを開くStage B barrier。
  - exp287/corrected-exp264のID/well/fold/actual照合、well別RMSE delta。
  - global/fold/hidden-like/coverageの固定AND gateとreport-only指標。
  - 設計固定した11生成物とreproducibility manifest。
- `exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_inference.py/.ipynb`は、model/prediction/submissionを拒否して意図的に停止するfail-closed候補とした。
- `tests/test_exp336_formation_tail_attribution.py`へsynthetic tests 10件を追加した。
- 既存canonical train/inference Notebookは、明示採用前に上書きしないルールに従いscaffoldのまま保持した。
- configは`implementation_approved=true`、`active_stage=implementation_complete_no_run`へ更新した。Kaggle push、Stage A/B run、inference、submission flagはすべてfalseのまま。
- model / LightGBM config / trained fold / booster / control再学習は`0 / 0 / 0 / 0 / 0`。親control再学習なし。

### 実装確認

```bash
.venv/bin/python -m py_compile experiments/exp336_exp287_formation_tail_attribution_readout/*compact_selfcontained*.py tests/test_exp336_formation_tail_attribution.py
.venv/bin/ruff check experiments/exp336_exp287_formation_tail_attribution_readout/*compact_selfcontained*.py tests/test_exp336_formation_tail_attribution.py
.venv/bin/pytest -q tests/test_exp336_formation_tail_attribution.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp336_exp287_formation_tail_attribution_readout/exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp336_exp287_formation_tail_attribution_readout/exp336_exp287_formation_tail_attribution_readout_compact_selfcontained_inference.py
make validate-exp EXP=exp336_exp287_formation_tail_attribution_readout
```

- synthetic tests: `10 passed`。
- Jupytext train/inference round-trip: PASS。
- py_compile / ruff: PASS。
- `make validate-exp EXP=exp336_exp287_formation_tail_attribution_readout`: strict PASS。
- `task` executableは環境にないため、strict experiment validationはMakefile同等コマンドを使う。
- この実装確認時点ではKaggle package/push/run、output取得、inference、submissionを実行していなかった。後段で明示承認されたStage A/Bだけを実行した。

## 次のアクション

完了済み。後段の実行記録を正とする。

## 2026-07-22 Kaggle CPU実行承認とpreflight

- ユーザーの`実行してください`を、compact train候補のcanonical採用とKaggle CPU package/push/run 1回の明示承認として記録した。
- 実行対象はStage A target-free freezeとStage B frozen attributionのみ。
- primary risk family: 6。
- active model variant / LightGBM config / trained fold / booster / parent control再学習: `0 / 0 / 0 / 0 / 0`。
- PF/HMM/Beam run: 0。
- runtime: Kaggle CPU、single worker、BLAS thread 1、GPU off、internet off。
- expected runtime: `1,200-7,200 sec`、計画上限`14,400 sec`（4時間）。
- exp287 kernel sourceにformation valid cache 5件、formation manifest、OOFが存在することを`kaggle kernels files`で確認した。
- corrected exp264 kernel sourceに`stage_d_oof_predictions.parquet`が存在することを`kaggle kernels files`で確認した。
- Kaggle CLI 2.2.3、OAuth credential利用可能。API tokenは未設定だがCLI OAuthとlegacy credentialは利用可能。
- 初回canonical kernel予定: `kentookumura/exp336-exp287-formation-tail-attribution-readout-train`。
- 初回canonical title予定: `exp336 exp287 formation tail attribution readout train`。
- inference、prediction correction、submission、competition submitは承認対象外で、全flag falseを維持する。

### 初回push失敗

- package tests 14件、strict experiment validation、embedded config/input auditをPASS後に初回pushを実施した。
- `kaggle kernels push`はkernel作成前に詳細なしの`SaveKernel 400 Bad Request`で失敗した。
- id/titleのslug文字列は一致し、親kernel source 2件も取得可能だった。
- 初回slugは54文字で、成功済みの近傍canonical slugが48-49文字に収まっていることから、Kaggle kernel slug長制約が最有力原因と判断した。
- kernel作成確認は`pull` 403で不成立。別kernelが作成された証拠はない。
- 実験番号と意味を維持し50文字未満にするため、retry canonicalを`kentookumura/exp336-exp287-formtail-attribution-readout-train`、titleを`exp336 exp287 formtail attribution readout train`へ固定する。
- retryは1回、Kaggle CLI timeoutを計画上限`14,400 sec`として同じpackage内容を再生成後に行う。

### Kaggle CPU version 1 technical ERROR

- retry canonicalへのpushは成功し、kernel version 1、id_no `128221753`、private、CPU、internet off、親kernel source 2件をpullで確認した。
- version 1は約68秒で`KernelWorkerStatus.ERROR`。
- 最初の意味のある例外は`FileNotFoundError: SHA-matched hidden-like assignment was not unique`。
- 同一SHAのassignmentが`/kaggle/working/inputs`と親2 kernelの`inputs`に合計3コピーあり、resolverが内容同一でもpath複数を拒否した。
- Stage A freeze後、Stage B family評価前のtechnical fail-close。model/config/train-fold/booster/control再学習は`0/0/0/0/0`、prediction/submission生成なし。
- scientific contract、family、boundary、gate、入力SHAは変更しない。同一expected SHAに一致する複数コピーはconfig pattern順の先頭を決定的に選ぶresolver修正だけを行う。
- equivalent-copy resolverのsynthetic testを追加し、同じcanonical kernel idへversion 2をpushする。

## 2026-07-22 Kaggle CPU version 2完了

- resolverを、expected SHA一致copyが複数でも内容が同一ならconfig pattern順の先頭を決定的に選ぶよう修正した。科学契約、6 family、四分位、coverage、decision gateは変更していない。
- synthetic testを1件追加し、exp336専用`11 passed`、Notebook共通test込み`15 passed`、py_compile、ruff F821、Jupytext round-trip、strict experiment validationを再確認した。
- 同じcanonical kernel `kentookumura/exp336-exp287-formtail-attribution-readout-train`へversion 2をpushした。
- version 2はprivate CPU、GPU/internet off、id_no `128221753`でCOMPLETE。readout本体runtimeは`92.458 sec`、Notebook全体は`102.406 sec`。
- model / LightGBM config / trained fold / booster / control再学習は`0 / 0 / 0 / 0 / 0`。prediction、inference、submissionは生成していない。

### Technical / leakage audit

- exp287 valid formation cache 5 partition、3,783,989 rows、773 wells、formation非finite値0を確認した。
- Stage A raw value列は`MD/X/Y/Z/TVT_input`だけで、forbidden value列は開いていない。
- 全6 familyがstrict quartile edge eligible、global Q1/Q4は`194/193 wells`、fold/hidden-likeを含むcoverageは全PASS。
- Stage A freeze manifest SHA `e65a9924c11f77008d1574070f71b6cf2d099993e8510eeaf7cc285c5d54979f`をStage B OOF open前に固定し、再検証した。
- exp287 / corrected exp264 OOFはID/well/foldが完全一致し、actual TVT max absolute differenceは`0.0 ft`。
- 11成果物を取得し、reproducibility manifestに記録された10子artifactのSHAを実ファイルと照合して全一致した。

### 固定判定

| family | global Q4-Q1 mean | median | 正fold | hidden-like正scope | PASS |
| --- | ---: | ---: | ---: | ---: | --- |
| plane reference distance | +0.081817 | +0.080864 | 3/5 | 0/2 | FAIL |
| dense reference distance | +0.350471 | +0.314590 | 5/5 | 0/2 | FAIL |
| dense neighbor uncertainty | +0.151014 | +0.016836 | 5/5 | 1/2 | FAIL |
| plane-dense disagreement | +0.229596 | +0.128755 | 4/5 | 0/2 | FAIL |
| formation consensus spread | +0.014517 | +0.080344 | 4/5 | 2/2 | FAIL |
| known-prefix formation calibration error | +0.031585 | -0.002890 | 4/5 | 2/2 | FAIL |

- `dense_reference_distance`はglobal effect、median、5/5 foldsを通過したが、hidden-like spatial/typewell-purgedが`-0.019368/-0.037255 ft`で両方逆方向。
- `formation_consensus_spread`と`known_prefix_formation_calibration_error`はhidden-like 2面を通過したが、global effectが`+0.25 ft`未満。後者はglobal medianも負。
- passed familyは`0/6`、固定statusは`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`。

### 再現性SHA

- freeze manifest: `e65a9924c11f77008d1574070f71b6cf2d099993e8510eeaf7cc285c5d54979f`
- target-free attributes: `a53a537db8eb9416ef4b83e6529d11d8b10233f02926939c349bd679d09f03aa`
- attribution decision: `06b7bfd64405b8f330ac818efe59ca51c9b1b13babc5aff440d72447eea9f99a`
- scientific contract: `b2c8e40c4912abd29277fae96b462d22aaf9826b0a3d4a799902b0f640ee3328`
- pushed train Notebook: `abc6c1cde62dfc04ea416b78e44048b469cc2868485e4a83eecec3bd57b3422c`
- pushed compact train source: `d9d6cfb504e1a642138475ed95bdb850488f3e59c3c9d84174cee25d32130307`
- pushed config: `6f31f387ff217daaa23bbf3126cc6cf34a5264c53c8b525194f2e9dfd8626d87`
- pushed kernel metadata: `189141cf09c7224336bd21be662701b66b4ab081215d78edd23be2e3097a2227`
- family/fold/hidden/context/well-delta/manifest各artifact SHAは`metrics.json`とKaggle outputのreproducibility manifestに記録した。

### 結論

- exp287のglobal improvementは再現したが、6つのtarget-free formation risk familyにglobal・fold・hidden-likeを同時に満たす安定した帰属はなかった。
- 事前契約どおりformation attribution枝を閉じ、同じOOFでfamily、threshold、weight、clip、shrink、gateを追加探索しない。
- exp287/exp334のtrain-side昇格、別介入、inference、submissionを許可しない。記録確定とvalidationまで完了した。
